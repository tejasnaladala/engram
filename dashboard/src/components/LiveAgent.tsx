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
  // Tabular Q-Learning baseline (standard RL)
  rnd: RunnerState
  rndQTable: Map<string,number[]>  // position-based: "x,y" -> Q-values
  rndEpsilon: number
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
    rnd: newRunner(), rndQTable: new Map(), rndEpsilon: 0.4,
    rndRewards:[], rndSuccesses:[], rndSolved:0,
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
  // Single attempt per maze -- 150 steps max.
  // Real-world scenario: robot enters a new building ONCE.
  if (r.step > 150) done = true
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
  const eNewQ = [...eQ]; eNewQ[eAction] += 0.25*(eTarget-eNewQ[eAction])
  s.engQTable.set(eKey, eNewQ)
  s.eng = eResult.runner

  // Tabular Q-Learning (standard RL): uses exact (x,y) position as state.
  // This is how robots/agents learn in standard ML -- memorizes positions.
  // Works great on ONE maze, fails on new mazes (no generalization).
  const rKey = `${s.rnd.ax},${s.rnd.ay}`
  if (!s.rndQTable.has(rKey)) s.rndQTable.set(rKey, [0,0,0,0])
  const rQ = s.rndQTable.get(rKey)!
  const rAction = Math.random()<s.rndEpsilon ? Math.floor(Math.random()*4) : rQ.indexOf(Math.max(...rQ))
  const rResult = stepRunner(s.rnd, s.grid, rAction)

  // Q-update for tabular agent
  const rNextKey = `${rResult.runner.ax},${rResult.runner.ay}`
  if (!s.rndQTable.has(rNextKey)) s.rndQTable.set(rNextKey, [0,0,0,0])
  const rNextQ = s.rndQTable.get(rNextKey)!
  const rTarget = rResult.reward + (rResult.done?0:0.99*Math.max(...rNextQ))
  const rNewQ = [...rQ]; rNewQ[rAction] += 0.15*(rTarget-rNewQ[rAction])
  s.rndQTable.set(rKey, rNewQ)
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
    // Tabular Q-table is WIPED on new maze -- position entries are useless
    // in a different layout. This is the fundamental limitation of standard RL.
    // The agent starts from ZERO knowledge every single maze.
    s.rndQTable = new Map()
    s.rndEpsilon = 0.5  // high exploration since it knows nothing

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

// Dual-line success rate chart -- THE key visual showing divergence
function SuccessChart({ engSuccesses, rndSuccesses }: { engSuccesses:boolean[]; rndSuccesses:boolean[] }) {
  const containerRef = useRef<HTMLDivElement>(null)
  const ref = useRef<HTMLCanvasElement>(null)
  useEffect(() => {
    const c = ref.current; const cont = containerRef.current; if (!c||!cont) return
    const ctx = c.getContext('2d')!
    const rect = cont.getBoundingClientRect()
    const w = Math.max(200, rect.width), h = 60
    c.width=w*2; c.height=h*2; c.style.width=w+'px'; c.style.height=h+'px'
    ctx.scale(2,2)
    ctx.fillStyle='#060810'; ctx.fillRect(0,0,w,h)

    if (engSuccesses.length < 3) {
      ctx.fillStyle='#2a3450'; ctx.font='9px "JetBrains Mono", monospace'
      ctx.textAlign='center'; ctx.fillText('Collecting data...',w/2,h/2)
      return
    }

    // Compute windowed success rate for both
    const win = 5
    const engRates: number[] = []; const rndRates: number[] = []
    for (let i = 0; i < engSuccesses.length; i++) {
      const start = Math.max(0,i-win+1)
      const eSlice = engSuccesses.slice(start,i+1)
      const rSlice = rndSuccesses.slice(start,i+1)
      engRates.push(eSlice.filter(Boolean).length/eSlice.length)
      rndRates.push(rSlice.filter(Boolean).length/rSlice.length)
    }

    const n = engRates.length
    const maxR = 1.0

    // Area fill for Engram
    ctx.fillStyle = 'rgba(48,152,168,0.08)'
    ctx.beginPath(); ctx.moveTo(0,h)
    for (let i=0;i<n;i++) ctx.lineTo((i/(n-1))*w, h - engRates[i]/maxR*(h-10)-5)
    ctx.lineTo(w,h); ctx.closePath(); ctx.fill()

    // Engram line (teal)
    ctx.strokeStyle='#3098a8'; ctx.lineWidth=2; ctx.beginPath()
    for (let i=0;i<n;i++) { const x=(i/(n-1))*w, y=h-engRates[i]/maxR*(h-10)-5; if(i===0)ctx.moveTo(x,y);else ctx.lineTo(x,y) }
    ctx.stroke()

    // Standard RL line (red/brown)
    ctx.strokeStyle='#906060'; ctx.lineWidth=1.5; ctx.setLineDash([3,2]); ctx.beginPath()
    for (let i=0;i<n;i++) { const x=(i/(n-1))*w, y=h-rndRates[i]/maxR*(h-10)-5; if(i===0)ctx.moveTo(x,y);else ctx.lineTo(x,y) }
    ctx.stroke(); ctx.setLineDash([])

    // Labels
    ctx.font='8px "JetBrains Mono", monospace'
    ctx.fillStyle='#3098a8'; ctx.textAlign='left'; ctx.fillText('ENGRAM',3,10)
    ctx.fillStyle='#906060'; ctx.fillText('STD RL',3,20)
    ctx.fillStyle='#2a3450'; ctx.textAlign='right'; ctx.fillText(`ep ${n}`,w-3,h-3)

    // Current values at right edge
    if (n > 0) {
      const ey = h - engRates[n-1]/maxR*(h-10)-5
      const ry = h - rndRates[n-1]/maxR*(h-10)-5
      ctx.fillStyle='#3098a8'; ctx.textAlign='right'; ctx.fillText(`${(engRates[n-1]*100).toFixed(0)}%`,w-3,ey-3)
      ctx.fillStyle='#906060'; ctx.fillText(`${(rndRates[n-1]*100).toFixed(0)}%`,w-3,ry+12)
    }
  }, [engSuccesses, rndSuccesses])

  return <div ref={containerRef} style={{ width:'100%' }}>
    <canvas ref={ref} style={{ display:'block', borderRadius:'2px', border:'1px solid rgba(48,80,120,0.08)' }} />
  </div>
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

      {/* Right maze: Tabular Q-Learning (standard RL) */}
      <MazeCanvas grid={state.grid} ax={state.rnd.ax} ay={state.rnd.ay} path={state.rnd.path}
        solved={state.rnd.solved} label="TABULAR Q-LEARN (STD RL)" color="#906060" />

      {/* Scoreboard + chart */}
      <div style={{ flex:1, display:'flex', flexDirection:'column', gap:'6px', minWidth:0, justifyContent:'center' }}>
        {/* BIG SCOREBOARD -- unmissable */}
        <div style={{ display:'flex', gap:'2px', alignItems:'stretch', fontFamily:'var(--mono)' }}>
          {/* Engram score */}
          <div style={{
            flex:1, background:'rgba(48,152,168,0.06)', borderRadius:'3px', padding:'6px 10px',
            border:'1px solid rgba(48,152,168,0.1)', display:'flex', flexDirection:'column', gap:'2px',
          }}>
            <span style={{ fontSize:'7px', color:'#3098a8', letterSpacing:'1.5px' }}>ENGRAM</span>
            <div style={{ display:'flex', alignItems:'baseline', gap:'8px' }}>
              <span style={{ fontSize:'28px', fontWeight:700, color:'#3098a8', fontVariantNumeric:'tabular-nums' }}>
                {state.engSolved}
              </span>
              <span style={{ fontSize:'10px', color:'var(--t-sec)' }}>solved</span>
              <span style={{ fontSize:'14px', fontWeight:600, color: engRate>0.3?'#508870':'#3098a8' }}>
                {(engRate*100).toFixed(0)}%
              </span>
            </div>
            <span style={{ fontSize:'7px', color:'var(--t-dim)' }}>
              features: {state.engQTable.size} | eps: {state.engEpsilon.toFixed(2)}
              {state.isPersisted ? ' | PERSISTENT' : ''}
            </span>
          </div>

          {/* VS divider */}
          <div style={{ display:'flex', alignItems:'center', padding:'0 6px' }}>
            <span style={{ fontSize:'10px', color:'var(--t-dim)', fontFamily:'var(--mono)', letterSpacing:'1px' }}>VS</span>
          </div>

          {/* Standard RL score */}
          <div style={{
            flex:1, background:'rgba(144,96,96,0.06)', borderRadius:'3px', padding:'6px 10px',
            border:'1px solid rgba(144,96,96,0.1)', display:'flex', flexDirection:'column', gap:'2px',
          }}>
            <span style={{ fontSize:'7px', color:'#906060', letterSpacing:'1.5px' }}>TABULAR Q-LEARN</span>
            <div style={{ display:'flex', alignItems:'baseline', gap:'8px' }}>
              <span style={{ fontSize:'28px', fontWeight:700, color:'#906060', fontVariantNumeric:'tabular-nums' }}>
                {state.rndSolved}
              </span>
              <span style={{ fontSize:'10px', color:'var(--t-sec)' }}>solved</span>
              <span style={{ fontSize:'14px', fontWeight:600, color:'#906060' }}>
                {(rndRate*100).toFixed(0)}%
              </span>
            </div>
            <span style={{ fontSize:'7px', color:'var(--t-dim)' }}>
              positions: {state.rndQTable.size} | WIPED EACH MAZE | NO TRANSFER
            </span>
          </div>
        </div>

        {/* Dual success rate chart -- shows the gap growing over time */}
        <SuccessChart engSuccesses={state.engSuccesses} rndSuccesses={state.rndSuccesses} />

        {/* Episode + status line */}
        <div style={{ display:'flex', alignItems:'center', gap:'10px', fontFamily:'var(--mono)' }}>
          <span style={{ fontSize:'10px', color:'var(--t-sec)' }}>EP {state.episode}</span>
          <span style={{ fontSize:'8px', color:'var(--t-dim)' }}>MAZE #{state.totalMazes}</span>
          <div style={{ width:'5px', height:'5px', borderRadius:'50%',
            background: engRate>rndRate+0.1?'#508870':'#3098a8',
            boxShadow: `0 0 6px ${engRate>rndRate+0.1?'#508870':'#3098a8'}`,
            animation:'pulse 2s ease-in-out infinite',
          }} />
          <span style={{ fontSize:'9px', color: engRate>rndRate+0.1?'#508870':'#3098a8', letterSpacing:'1px', fontWeight:500 }}>
            {engRate>rndRate+0.1 ? 'ENGRAM OUTPERFORMING' : engRate>rndRate ? 'ENGRAM LEADING' : 'COMPETING'}
          </span>
        </div>
      </div>
    </div>
  )
}
