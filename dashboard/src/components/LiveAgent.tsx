import { useState, useEffect, useRef, useCallback } from 'react'

/*
  SIDE-BY-SIDE: Feature Q-Learning vs Random Baseline
  Same maze, same start, same goal. Watch one learn and one flail.
  Q-table persists to localStorage -- reload page and learning continues.
*/

const GRID = 9
const WALL = 1
const HAZARD = 3
const ACTIONS: [number,number][] = [[0,-1],[1,0],[0,1],[-1,0]]
const ACT_SYMS = ['\u2191','\u2192','\u2193','\u2190']

function mulberry32(seed: number) {
  return function() {
    seed |= 0; seed = seed + 0x6D2B79F5 | 0
    let t = Math.imul(seed ^ seed >>> 15, 1 | seed)
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t
    return ((t ^ t >>> 14) >>> 0) / 4294967296
  }
}

function randomMaze(seed: number): number[][] {
  const g: number[][] = Array.from({length:GRID}, () => Array(GRID).fill(WALL))
  const rng = mulberry32(seed)
  function carve(x: number, y: number) {
    g[y][x] = 0
    const dirs: [number,number][] = [[0,-2],[2,0],[0,2],[-2,0]]
    for (let i = dirs.length-1; i > 0; i--) { const j = Math.floor(rng()*(i+1)); [dirs[i],dirs[j]] = [dirs[j],dirs[i]] }
    for (const [dx,dy] of dirs) {
      const nx = x+dx, ny = y+dy
      if (nx>=1 && nx<=GRID-2 && ny>=1 && ny<=GRID-2 && g[ny][nx]===WALL) { g[y+dy/2][x+dx/2] = 0; carve(nx,ny) }
    }
  }
  carve(1,1)
  for (let i = 0; i < 2; i++) {
    for (let a = 0; a < 30; a++) {
      const hx = 1+Math.floor(rng()*(GRID-2)), hy = 1+Math.floor(rng()*(GRID-2))
      if (g[hy][hx]===0 && !(hx===1&&hy===1) && !(hx===GRID-2&&hy===GRID-2)) { g[hy][hx] = HAZARD; break }
    }
  }
  return g
}

function isSolvable(grid: number[][]): boolean {
  const visited = new Set<string>()
  const queue: [number,number][] = [[1,1]]
  visited.add('1,1')
  while (queue.length > 0) {
    const [x,y] = queue.shift()!
    if (x===GRID-2 && y===GRID-2) return true
    for (const [dx,dy] of ACTIONS) {
      const nx=x+dx, ny=y+dy, k=`${nx},${ny}`
      if (nx>=0&&nx<GRID&&ny>=0&&ny<GRID&&grid[ny][nx]!==WALL&&!visited.has(k)) { visited.add(k); queue.push([nx,ny]) }
    }
  }
  return false
}

function featureKey(grid: number[][], ax: number, ay: number, gx: number, gy: number): string {
  const isOpen = (x: number, y: number) => x>=0&&x<GRID&&y>=0&&y<GRID&&grid[y][x]!==WALL
  const w = ACTIONS.map(([dx,dy]) => isOpen(ax+dx,ay+dy)?0:1)
  const exits = ACTIONS.map(([dx,dy]) => {
    const nx=ax+dx, ny=ay+dy; if (!isOpen(nx,ny)) return 0
    let c = 0; for (const [ddx,ddy] of ACTIONS) { if (isOpen(nx+ddx,ny+ddy)&&!(nx+ddx===ax&&ny+ddy===ay)) c++ }
    return Math.min(c,3)
  })
  const gdx = Math.sign(gx-ax), gdy = Math.sign(gy-ay)
  const dist = Math.abs(gx-ax)+Math.abs(gy-ay)
  const db = dist<=3?2:dist<=7?1:0
  const hz = ACTIONS.some(([dx,dy])=>{const nx=ax+dx,ny=ay+dy;return nx>=0&&nx<GRID&&ny>=0&&ny<GRID&&grid[ny][nx]===HAZARD})?1:0
  return `${w.join('')}_${exits.join('')}_${gdx}_${gdy}_${db}_${hz}`
}

// Load persisted Q-table from localStorage
function loadQTable(): Map<string,number[]> {
  try {
    const raw = localStorage.getItem('engram_qtable')
    if (raw) {
      const entries: [string,number[]][] = JSON.parse(raw)
      return new Map(entries)
    }
  } catch {}
  return new Map()
}

function saveQTable(qt: Map<string,number[]>) {
  try {
    const entries = Array.from(qt.entries())
    localStorage.setItem('engram_qtable', JSON.stringify(entries))
    localStorage.setItem('engram_stats', JSON.stringify({
      saved: Date.now(),
      features: qt.size,
    }))
  } catch {}
}

interface RunnerState {
  ax: number; ay: number
  step: number; epReward: number
  path: [number,number][]
  solved: boolean
}

interface DualState {
  grid: number[][]
  mazeSeed: number
  episode: number
  // Engram agent
  eng: RunnerState
  engQTable: Map<string,number[]>
  engEpsilon: number
  engRewards: number[]
  engSuccesses: boolean[]
  engSolved: number
  // Random baseline
  rnd: RunnerState
  rndRewards: number[]
  rndSuccesses: boolean[]
  rndSolved: number
  // Shared
  totalMazes: number
  isPersisted: boolean
}

function newRunner(): RunnerState {
  return { ax:1, ay:1, step:0, epReward:0, path:[[1,1]], solved:false }
}

function createDual(): DualState {
  const qt = loadQTable()
  let grid = randomMaze(42)
  while (!isSolvable(grid)) grid = randomMaze(Math.floor(Math.random()*1e9))
  return {
    grid, mazeSeed:42, episode:0,
    eng: newRunner(), engQTable: qt, engEpsilon: qt.size > 0 ? 0.15 : 0.4,
    engRewards:[], engSuccesses:[], engSolved:0,
    rnd: newRunner(), rndRewards:[], rndSuccesses:[], rndSolved:0,
    totalMazes:1, isPersisted: qt.size > 0,
  }
}

function stepRunner(
  s: RunnerState, grid: number[][], action: number
): { runner: RunnerState; reward: number; done: boolean } {
  const r = {...s}
  const [dx,dy] = ACTIONS[action]
  const nx=r.ax+dx, ny=r.ay+dy
  let reward = -0.02
  let done = false
  const prevDist = Math.abs(r.ax-(GRID-2))+Math.abs(r.ay-(GRID-2))

  if (nx>=0&&nx<GRID&&ny>=0&&ny<GRID&&grid[ny][nx]!==WALL) {
    r.ax=nx; r.ay=ny
    const newDist = Math.abs(r.ax-(GRID-2))+Math.abs(r.ay-(GRID-2))
    if (newDist<prevDist) reward+=0.05; else if(newDist>prevDist) reward-=0.03
    if (grid[ny][nx]===HAZARD) reward=-1.0
    if (nx===GRID-2&&ny===GRID-2) { reward=10.0; done=true; r.solved=true }
  } else { reward=-0.3 }

  r.epReward += reward
  r.step++
  r.path = [...r.path, [r.ax,r.ay]]
  if (r.step > 400) done = true
  return { runner:r, reward, done }
}

function dualStep(state: DualState): DualState {
  const s = {...state}

  // Engram: feature Q-learning
  const eKey = featureKey(s.grid, s.eng.ax, s.eng.ay, GRID-2, GRID-2)
  if (!s.engQTable.has(eKey)) s.engQTable.set(eKey, [0,0,0,0])
  const eQ = s.engQTable.get(eKey)!
  const eAction = Math.random()<s.engEpsilon ? Math.floor(Math.random()*4) : eQ.indexOf(Math.max(...eQ))
  const eResult = stepRunner(s.eng, s.grid, eAction)

  // Q-update for Engram
  const eNextKey = featureKey(s.grid, eResult.runner.ax, eResult.runner.ay, GRID-2, GRID-2)
  if (!s.engQTable.has(eNextKey)) s.engQTable.set(eNextKey, [0,0,0,0])
  const eNextQ = s.engQTable.get(eNextKey)!
  const eTarget = eResult.reward + (eResult.done?0:0.99*Math.max(...eNextQ))
  const eNewQ = [...eQ]; eNewQ[eAction] += 0.2*(eTarget-eNewQ[eAction])
  s.engQTable.set(eKey, eNewQ)
  s.eng = eResult.runner

  // Random: pure random actions
  const rAction = Math.floor(Math.random()*4)
  const rResult = stepRunner(s.rnd, s.grid, rAction)
  s.rnd = rResult.runner

  // Check episode end for both
  const engDone = eResult.done
  const rndDone = rResult.done

  // If either finishes, end the episode for both (same maze = same episode boundary)
  if (engDone || rndDone || s.eng.step > 400 || s.rnd.step > 400) {
    s.engRewards = [...s.engRewards, s.eng.epReward]
    s.engSuccesses = [...s.engSuccesses, s.eng.solved]
    if (s.eng.solved) s.engSolved++

    s.rndRewards = [...s.rndRewards, s.rnd.epReward]
    s.rndSuccesses = [...s.rndSuccesses, s.rnd.solved]
    if (s.rnd.solved) s.rndSolved++

    s.episode++
    s.eng = newRunner()
    s.rnd = newRunner()

    // New maze
    const seed = Date.now()+s.episode*7919
    let newGrid = randomMaze(seed)
    let att = 0
    while (!isSolvable(newGrid)&&att<10) { newGrid = randomMaze(seed+att*13); att++ }
    s.grid = newGrid
    s.totalMazes++
    s.engEpsilon = Math.max(0.08, s.engEpsilon*0.997)

    // Save Q-table every 5 episodes
    if (s.episode % 5 === 0) { saveQTable(s.engQTable); s.isPersisted = true }
  }

  return s
}

// Render a maze with agent
function MazeCanvas({ grid, ax, ay, path, solved, label, color, qTable }:
  { grid:number[][]; ax:number; ay:number; path:[number,number][]; solved:boolean; label:string; color:string; qTable?:Map<string,number[]> }
) {
  const ref = useRef<HTMLCanvasElement>(null)
  useEffect(() => {
    const c = ref.current; if (!c) return
    const ctx = c.getContext('2d')!
    const sz = 160, cell = sz/GRID
    c.width=sz*2; c.height=sz*2; c.style.width=sz+'px'; c.style.height=sz+'px'
    ctx.scale(2,2)
    ctx.fillStyle='#060810'; ctx.fillRect(0,0,sz,sz)

    for (let y=0;y<GRID;y++) for (let x=0;x<GRID;x++) {
      const v = grid[y][x]
      ctx.fillStyle = v===WALL?'#121828':v===HAZARD?'#1c1218':'#0b0e18'
      ctx.fillRect(x*cell+0.3,y*cell+0.3,cell-0.6,cell-0.6)
      if (v===HAZARD) { ctx.fillStyle='#906060'; ctx.font=`bold ${cell*0.5}px sans-serif`; ctx.textAlign='center'; ctx.textBaseline='middle'; ctx.fillText('!',x*cell+cell/2,y*cell+cell/2+1) }
    }

    // Q-arrows for Engram only
    if (qTable) {
      ctx.globalAlpha = 0.3
      for (let y=0;y<GRID;y++) for (let x=0;x<GRID;x++) {
        if (grid[y][x]!==0) continue
        const fk = featureKey(grid,x,y,GRID-2,GRID-2)
        if (!qTable.has(fk)) continue
        const qv = qTable.get(fk)!
        if (Math.max(...qv)<0.01) continue
        const best = qv.indexOf(Math.max(...qv))
        const [adx,ady] = ACTIONS[best]
        const cx=x*cell+cell/2, cy=y*cell+cell/2, al=cell*0.3
        ctx.strokeStyle=color; ctx.lineWidth=1
        ctx.beginPath(); ctx.moveTo(cx-adx*al*0.5,cy-ady*al*0.5); ctx.lineTo(cx+adx*al*0.5,cy+ady*al*0.5); ctx.stroke()
        ctx.beginPath(); ctx.arc(cx+adx*al*0.5,cy+ady*al*0.5,1.2,0,Math.PI*2); ctx.fillStyle=color; ctx.fill()
      }
      ctx.globalAlpha = 1
    }

    // Path
    for (const [px,py] of path.slice(0,-1)) { ctx.fillStyle=`${color}12`; ctx.fillRect(px*cell+cell*0.2,py*cell+cell*0.2,cell*0.6,cell*0.6) }

    // Goal
    ctx.fillStyle='#508870'; ctx.shadowColor='#508870'; ctx.shadowBlur=4
    ctx.beginPath(); ctx.arc((GRID-2)*cell+cell/2,(GRID-2)*cell+cell/2,cell*0.28,0,Math.PI*2); ctx.fill()
    ctx.shadowBlur=0; ctx.fillStyle='#0a0e16'; ctx.font=`bold ${cell*0.3}px sans-serif`; ctx.textAlign='center'; ctx.textBaseline='middle'
    ctx.fillText('G',(GRID-2)*cell+cell/2,(GRID-2)*cell+cell/2+0.5)

    // Agent
    ctx.fillStyle=solved?'#508870':color; ctx.shadowColor=solved?'#508870':color; ctx.shadowBlur=6
    ctx.beginPath(); ctx.arc(ax*cell+cell/2,ay*cell+cell/2,cell*0.22,0,Math.PI*2); ctx.fill()
    ctx.shadowBlur=0

    // Grid
    ctx.strokeStyle='rgba(48,80,120,0.04)'; ctx.lineWidth=0.3
    for (let i=0;i<=GRID;i++) { ctx.beginPath();ctx.moveTo(i*cell,0);ctx.lineTo(i*cell,sz);ctx.stroke();ctx.beginPath();ctx.moveTo(0,i*cell);ctx.lineTo(sz,i*cell);ctx.stroke() }
  }, [grid,ax,ay,path,solved,qTable])

  return (
    <div style={{ display:'flex', flexDirection:'column', gap:'2px', alignItems:'center' }}>
      <canvas ref={ref} style={{ borderRadius:'3px', border:`1px solid ${color}15` }} />
      <span style={{ fontFamily:'var(--mono)', fontSize:'9px', color, letterSpacing:'1px', fontWeight:600 }}>{label}</span>
    </div>
  )
}

// Comparison bar
function CompBar({ engVal, rndVal, label, suffix }: { engVal:number; rndVal:number; label:string; suffix?:string }) {
  const max = Math.max(Math.abs(engVal), Math.abs(rndVal), 0.01)
  const engW = Math.abs(engVal)/max*100
  const rndW = Math.abs(rndVal)/max*100
  const engWins = engVal > rndVal
  return (
    <div style={{ display:'flex', alignItems:'center', gap:'6px', fontFamily:'var(--mono)', fontSize:'8px' }}>
      <span style={{ width:'60px', textAlign:'right', color:'var(--t-dim)', letterSpacing:'0.5px' }}>{label}</span>
      <div style={{ flex:1, height:'8px', background:'var(--s4)', borderRadius:'2px', display:'flex', gap:'1px', overflow:'hidden' }}>
        <div style={{ width:`${engW}%`, background: engWins?'#3098a8':'#3098a860', borderRadius:'2px 0 0 2px', transition:'width 0.3s' }} />
        <div style={{ width:`${rndW}%`, background: !engWins?'#906060':'#90606060', borderRadius:'0 2px 2px 0', transition:'width 0.3s' }} />
      </div>
      <span style={{ width:'40px', color:'#3098a8', fontWeight:500 }}>{engVal.toFixed(0)}{suffix||''}</span>
      <span style={{ width:'5px', color:'var(--t-ghost)' }}>|</span>
      <span style={{ width:'40px', color:'#906060' }}>{rndVal.toFixed(0)}{suffix||''}</span>
    </div>
  )
}

export default function LiveAgent() {
  const [state, setState] = useState<DualState>(createDual)

  useEffect(() => {
    const interval = setInterval(() => setState(prev => dualStep(prev)), 30)
    return () => clearInterval(interval)
  }, [])

  const engR20 = state.engSuccesses.slice(-20)
  const rndR20 = state.rndSuccesses.slice(-20)
  const engRate = engR20.length>0 ? engR20.filter(Boolean).length/engR20.length : 0
  const rndRate = rndR20.length>0 ? rndR20.filter(Boolean).length/rndR20.length : 0
  const engAvgR = state.engRewards.length>0 ? state.engRewards.slice(-20).reduce((a,b)=>a+b,0)/Math.min(20,state.engRewards.length) : 0
  const rndAvgR = state.rndRewards.length>0 ? state.rndRewards.slice(-20).reduce((a,b)=>a+b,0)/Math.min(20,state.rndRewards.length) : 0

  return (
    <div style={{ display:'flex', gap:'12px', height:'100%', padding:'6px 10px', alignItems:'center' }}>
      {/* Left maze: Engram */}
      <MazeCanvas grid={state.grid} ax={state.eng.ax} ay={state.eng.ay} path={state.eng.path}
        solved={state.eng.solved} label="ENGRAM (FEATURE Q-LEARN)" color="#3098a8" qTable={state.engQTable} />

      {/* Right maze: Random */}
      <MazeCanvas grid={state.grid} ax={state.rnd.ax} ay={state.rnd.ay} path={state.rnd.path}
        solved={state.rnd.solved} label="RANDOM BASELINE" color="#906060" />

      {/* Comparison stats */}
      <div style={{ flex:1, display:'flex', flexDirection:'column', gap:'6px', minWidth:0, justifyContent:'center' }}>
        {/* Episode counter */}
        <div style={{ display:'flex', gap:'20px', fontFamily:'var(--mono)' }}>
          <div><span style={{ fontSize:'7px', color:'var(--t-dim)', letterSpacing:'1px' }}>EPISODE</span><br/>
            <span style={{ fontSize:'18px', color:'var(--t-max)', fontWeight:500 }}>{state.episode}</span></div>
          <div><span style={{ fontSize:'7px', color:'var(--t-dim)', letterSpacing:'1px' }}>MAZES</span><br/>
            <span style={{ fontSize:'18px', color:'var(--t-max)', fontWeight:500 }}>{state.totalMazes}</span></div>
          <div><span style={{ fontSize:'7px', color:'var(--t-dim)', letterSpacing:'1px' }}>Q-FEATURES</span><br/>
            <span style={{ fontSize:'14px', color:'#3098a8', fontWeight:500 }}>{state.engQTable.size}</span></div>
          <div><span style={{ fontSize:'7px', color:'var(--t-dim)', letterSpacing:'1px' }}>EPSILON</span><br/>
            <span style={{ fontSize:'14px', color:'var(--t-sec)', fontWeight:500 }}>{state.engEpsilon.toFixed(3)}</span></div>
          {state.isPersisted && (
            <div><span style={{ fontSize:'7px', color:'var(--t-dim)', letterSpacing:'1px' }}>MODEL</span><br/>
              <span style={{ fontSize:'11px', color:'#508870', fontWeight:500 }}>SAVED</span></div>
          )}
        </div>

        {/* Comparison bars */}
        <div style={{ display:'flex', flexDirection:'column', gap:'3px' }}>
          <div style={{ display:'flex', gap:'6px', fontFamily:'var(--mono)', fontSize:'7px', color:'var(--t-dim)', letterSpacing:'0.5px', paddingLeft:'66px' }}>
            <span style={{ flex:1 }}>ENGRAM vs RANDOM</span>
          </div>
          <CompBar engVal={engRate*100} rndVal={rndRate*100} label="SUCCESS" suffix="%" />
          <CompBar engVal={state.engSolved} rndVal={state.rndSolved} label="SOLVED" />
          <CompBar engVal={engAvgR} rndVal={rndAvgR} label="AVG REWARD" />
        </div>

        {/* Status */}
        <div style={{ display:'flex', alignItems:'center', gap:'8px' }}>
          <div style={{ width:'5px', height:'5px', borderRadius:'50%',
            background: engRate>0.3?'#508870':'#3098a8',
            boxShadow: `0 0 6px ${engRate>0.3?'#508870':'#3098a8'}`,
            animation:'pulse 2s ease-in-out infinite',
          }} />
          <span style={{ fontFamily:'var(--mono)', fontSize:'10px', color: engRate>0.3?'#508870':'#3098a8', letterSpacing:'1.5px', fontWeight:500 }}>
            {engRate>0.5?'CONVERGING':engRate>0.15?'LEARNING':'EXPLORING'}
          </span>
          <span style={{ fontFamily:'var(--mono)', fontSize:'8px', color:'var(--t-dim)' }}>
            SAME MAZE, DIFFERENT STRATEGY
          </span>
        </div>
      </div>
    </div>
  )
}
