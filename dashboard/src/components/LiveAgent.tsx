import { useState, useEffect, useRef } from 'react'

/*
  Live learning agent -- random maze every episode.
  Uses odd-sized grid so recursive backtracking produces valid mazes.
  Feature-based Q-learning generalizes across unseen layouts.
*/

const GRID = 9  // MUST be odd for recursive backtracking to work
const WALL = 1
const HAZARD = 3
const ACTIONS: [number,number][] = [[0,-1],[1,0],[0,1],[-1,0]]
const ACT_SYMS = ['\u2191','\u2192','\u2193','\u2190']

interface AgentState {
  grid: number[][]
  ax: number; ay: number
  gx: number; gy: number
  episode: number; step: number
  epReward: number
  rewardHist: number[]
  successHist: boolean[]
  path: [number,number][]
  qTable: Map<string, number[]>
  epsilon: number
  lastAction: number
  solved: boolean
  bestReward: number
  mazesSolved: number
  uniqueMazes: number
}

// Seedable PRNG
function mulberry32(seed: number) {
  return function() {
    seed |= 0; seed = seed + 0x6D2B79F5 | 0
    let t = Math.imul(seed ^ seed >>> 15, 1 | seed)
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t
    return ((t ^ t >>> 14) >>> 0) / 4294967296
  }
}

// Generate a valid random maze using recursive backtracking.
// Grid MUST be odd-sized. Carves passages at odd coordinates.
// Start (1,1) and goal (GRID-2, GRID-2) are always reachable
// because the algorithm creates a spanning tree over ALL odd cells.
function randomMaze(seed: number): number[][] {
  const g: number[][] = Array.from({length:GRID}, () => Array(GRID).fill(WALL))
  const rng = mulberry32(seed)

  function carve(x: number, y: number) {
    g[y][x] = 0
    // Shuffle directions
    const dirs: [number,number][] = [[0,-2],[2,0],[0,2],[-2,0]]
    for (let i = dirs.length - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1))
      ;[dirs[i], dirs[j]] = [dirs[j], dirs[i]]
    }
    for (const [dx,dy] of dirs) {
      const nx = x+dx, ny = y+dy
      // Check bounds: must stay within 1..GRID-2 (the carveable interior)
      if (nx >= 1 && nx <= GRID-2 && ny >= 1 && ny <= GRID-2 && g[ny][nx] === WALL) {
        // Knock out wall between current and neighbor
        g[y + dy/2][x + dx/2] = 0
        carve(nx, ny)
      }
    }
  }

  carve(1, 1)

  // Verify goal is open (it should be since GRID-2 is odd when GRID is odd)
  // GRID=9: GRID-2=7, which is odd -- carving visits (1,1),(1,3),(1,5),(1,7),(3,1),...,(7,7)
  // Goal at (7,7) is guaranteed reachable.

  // Add 1-2 hazards in open cells (not start or goal)
  for (let i = 0; i < 2; i++) {
    for (let attempt = 0; attempt < 30; attempt++) {
      const hx = 1 + Math.floor(rng() * (GRID-2))
      const hy = 1 + Math.floor(rng() * (GRID-2))
      if (g[hy][hx] === 0 && !(hx===1&&hy===1) && !(hx===GRID-2&&hy===GRID-2)) {
        g[hy][hx] = HAZARD; break
      }
    }
  }
  return g
}

// Verify maze is solvable using BFS
function isSolvable(grid: number[][], sx: number, sy: number, gx: number, gy: number): boolean {
  const visited = new Set<string>()
  const queue: [number,number][] = [[sx,sy]]
  visited.add(`${sx},${sy}`)
  while (queue.length > 0) {
    const [x,y] = queue.shift()!
    if (x === gx && y === gy) return true
    for (const [dx,dy] of ACTIONS) {
      const nx = x+dx, ny = y+dy
      const key = `${nx},${ny}`
      if (nx>=0 && nx<GRID && ny>=0 && ny<GRID && grid[ny][nx] !== WALL && !visited.has(key)) {
        visited.add(key)
        queue.push([nx,ny])
      }
    }
  }
  return false
}

// Feature-based state key for GENERALIZATION
function featureKey(grid: number[][], ax: number, ay: number, gx: number, gy: number): string {
  const wall = (dx: number, dy: number) => {
    const nx = ax+dx, ny = ay+dy
    return (nx<0||nx>=GRID||ny<0||ny>=GRID||grid[ny][nx]===WALL) ? 1 : 0
  }
  const haz = (dx: number, dy: number) => {
    const nx = ax+dx, ny = ay+dy
    return (nx>=0&&nx<GRID&&ny>=0&&ny<GRID&&grid[ny][nx]===HAZARD) ? 1 : 0
  }
  const gdx = Math.sign(gx - ax)
  const gdy = Math.sign(gy - ay)
  const hazNear = haz(0,-1)+haz(1,0)+haz(0,1)+haz(-1,0) > 0 ? 1 : 0
  // Also encode 2-step wall lookahead for better navigation
  const wall2 = (dx: number, dy: number) => {
    const nx = ax+dx*2, ny = ay+dy*2
    return (nx<0||nx>=GRID||ny<0||ny>=GRID||grid[ny][nx]===WALL) ? 1 : 0
  }
  return `${wall(0,-1)}${wall(1,0)}${wall(0,1)}${wall(-1,0)}_${wall2(0,-1)}${wall2(1,0)}${wall2(0,1)}${wall2(-1,0)}_${gdx}_${gdy}_${hazNear}`
}

function createAgent(): AgentState {
  let grid = randomMaze(42)
  // Guarantee solvability
  while (!isSolvable(grid, 1, 1, GRID-2, GRID-2)) {
    grid = randomMaze(Math.floor(Math.random() * 1e9))
  }
  return {
    grid,
    ax:1, ay:1, gx:GRID-2, gy:GRID-2,
    episode:0, step:0, epReward:0,
    rewardHist:[], successHist:[],
    path:[[1,1]],
    qTable: new Map(),
    epsilon:0.4, lastAction:0,
    solved:false, bestReward:-999,
    mazesSolved:0, uniqueMazes:1,
  }
}

function getQ(st: AgentState, key: string): number[] {
  if (!st.qTable.has(key)) st.qTable.set(key, [0,0,0,0])
  return st.qTable.get(key)!
}

function agentStep(state: AgentState): AgentState {
  const s = {...state}
  const key = featureKey(s.grid, s.ax, s.ay, s.gx, s.gy)
  const qvals = getQ(s, key)

  const action = Math.random() < s.epsilon
    ? Math.floor(Math.random()*4)
    : qvals.indexOf(Math.max(...qvals))
  s.lastAction = action

  const [dx,dy] = ACTIONS[action]
  const nx = s.ax+dx, ny = s.ay+dy
  let reward = -0.02
  let done = false

  if (nx>=0 && nx<GRID && ny>=0 && ny<GRID && s.grid[ny][nx] !== WALL) {
    s.ax = nx; s.ay = ny
    if (s.grid[ny][nx] === HAZARD) reward = -1.0
    if (nx===s.gx && ny===s.gy) { reward = 10.0; done = true; s.solved = true; s.mazesSolved++ }
  } else { reward = -0.3 }

  s.epReward += reward
  s.step++
  s.path = [...s.path, [s.ax, s.ay]]

  // Q-learning update
  const nextKey = featureKey(s.grid, s.ax, s.ay, s.gx, s.gy)
  const nextQ = getQ(s, nextKey)
  const target = reward + (done ? 0 : 0.99 * Math.max(...nextQ))
  const newQ = [...qvals]
  newQ[action] += 0.2 * (target - newQ[action])
  s.qTable.set(key, newQ)

  // Episode end
  if (done || s.step > GRID*GRID*3) {
    s.rewardHist = [...s.rewardHist, s.epReward]
    s.successHist = [...s.successHist, done]
    if (s.epReward > s.bestReward) s.bestReward = s.epReward
    s.episode++
    s.epReward = 0; s.step = 0
    s.ax = 1; s.ay = 1
    // New random solvable maze every episode
    let newGrid = randomMaze(Date.now() + s.episode * 7919)
    let attempts = 0
    while (!isSolvable(newGrid, 1, 1, GRID-2, GRID-2) && attempts < 10) {
      newGrid = randomMaze(Date.now() + s.episode * 7919 + attempts * 13)
      attempts++
    }
    s.grid = newGrid
    s.uniqueMazes++
    s.path = [[1,1]]
    s.epsilon = Math.max(0.08, s.epsilon * 0.997)
    s.solved = false
  }

  return s
}

// Reward curve chart
function RewardCurve({ history }: { history: number[] }) {
  const ref = useRef<HTMLCanvasElement>(null)
  useEffect(() => {
    const c = ref.current; if (!c) return
    const ctx = c.getContext('2d')!
    const w = 240, h = 64
    c.width = w*2; c.height = h*2; c.style.width = w+'px'; c.style.height = h+'px'
    ctx.scale(2,2)
    ctx.fillStyle = '#060810'; ctx.fillRect(0,0,w,h)
    if (history.length < 2) return

    const win = 5
    const avg: number[] = []
    for (let i = 0; i < history.length; i++) {
      const start = Math.max(0, i-win+1)
      const sl = history.slice(start, i+1)
      avg.push(sl.reduce((a,b)=>a+b,0)/sl.length)
    }
    const mn = Math.min(...avg), mx = Math.max(...avg), rng = mx-mn||1

    // Area fill
    ctx.fillStyle = 'rgba(48,152,168,0.06)'
    ctx.beginPath(); ctx.moveTo(0, h)
    for (let i = 0; i < avg.length; i++) {
      ctx.lineTo((i/(avg.length-1))*w, h - ((avg[i]-mn)/rng)*(h-12) - 6)
    }
    ctx.lineTo(w, h); ctx.closePath(); ctx.fill()

    // Line
    ctx.strokeStyle = '#3098a8'; ctx.lineWidth = 1.5; ctx.beginPath()
    for (let i = 0; i < avg.length; i++) {
      const x = (i/(avg.length-1))*w, y = h - ((avg[i]-mn)/rng)*(h-12) - 6
      if (i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y)
    }
    ctx.stroke()

    // Zero reference
    if (mn < 0 && mx > 0) {
      const zy = h - ((0-mn)/rng)*(h-12) - 6
      ctx.strokeStyle = 'rgba(80,136,112,0.2)'; ctx.lineWidth = 0.5
      ctx.setLineDash([2,3]); ctx.beginPath(); ctx.moveTo(0,zy); ctx.lineTo(w,zy); ctx.stroke(); ctx.setLineDash([])
    }

    // Labels
    ctx.fillStyle = '#3a4868'; ctx.font = '9px "JetBrains Mono", monospace'
    ctx.textAlign = 'left'; ctx.fillText(mx.toFixed(1), 3, 11)
    ctx.fillText(mn.toFixed(1), 3, h-3)
    ctx.textAlign = 'right'; ctx.fillText(`ep ${history.length}`, w-3, h-3)
  }, [history])
  return <canvas ref={ref} style={{ borderRadius:'2px', border:'1px solid rgba(48,80,120,0.08)' }} />
}

export default function LiveAgent() {
  const [state, setState] = useState<AgentState>(createAgent)
  const gridRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const interval = setInterval(() => setState(prev => agentStep(prev)), 10)
    return () => clearInterval(interval)
  }, [])

  // Render grid
  useEffect(() => {
    const c = gridRef.current; if (!c) return
    const ctx = c.getContext('2d')!
    const sz = 200, cell = sz/GRID
    c.width = sz*2; c.height = sz*2; c.style.width = sz+'px'; c.style.height = sz+'px'
    ctx.scale(2,2)
    ctx.fillStyle = '#060810'; ctx.fillRect(0,0,sz,sz)

    for (let y = 0; y < GRID; y++) for (let x = 0; x < GRID; x++) {
      const v = state.grid[y][x]
      ctx.fillStyle = v===WALL ? '#121828' : v===HAZARD ? '#1c1218' : '#0b0e18'
      ctx.fillRect(x*cell+0.3, y*cell+0.3, cell-0.6, cell-0.6)
      if (v===HAZARD) {
        ctx.fillStyle = '#906060'; ctx.font = `bold ${cell*0.5}px sans-serif`
        ctx.textAlign = 'center'; ctx.textBaseline = 'middle'
        ctx.fillText('!', x*cell+cell/2, y*cell+cell/2+1)
      }
    }

    // Path trace
    for (const [px,py] of state.path.slice(0,-1)) {
      ctx.fillStyle = 'rgba(48,152,168,0.08)'
      ctx.fillRect(px*cell+cell*0.2, py*cell+cell*0.2, cell*0.6, cell*0.6)
    }

    // Goal
    ctx.fillStyle = '#508870'; ctx.shadowColor = '#508870'; ctx.shadowBlur = 6
    ctx.beginPath(); ctx.arc(state.gx*cell+cell/2, state.gy*cell+cell/2, cell*0.3, 0, Math.PI*2); ctx.fill()
    ctx.shadowBlur = 0
    // Goal label
    ctx.fillStyle = '#0a0e16'; ctx.font = `bold ${cell*0.35}px sans-serif`
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle'
    ctx.fillText('G', state.gx*cell+cell/2, state.gy*cell+cell/2+0.5)

    // Agent
    ctx.fillStyle = state.solved ? '#508870' : '#3098a8'
    ctx.shadowColor = state.solved ? '#508870' : '#3098a8'; ctx.shadowBlur = 8
    ctx.beginPath(); ctx.arc(state.ax*cell+cell/2, state.ay*cell+cell/2, cell*0.25, 0, Math.PI*2); ctx.fill()
    ctx.shadowBlur = 0
    // Agent label
    ctx.fillStyle = '#0a0e16'; ctx.font = `bold ${cell*0.35}px sans-serif`
    ctx.fillText('A', state.ax*cell+cell/2, state.ay*cell+cell/2+0.5)

    // Grid lines
    ctx.strokeStyle = 'rgba(48,80,120,0.04)'; ctx.lineWidth = 0.3
    for (let i = 0; i <= GRID; i++) {
      ctx.beginPath(); ctx.moveTo(i*cell,0); ctx.lineTo(i*cell,sz); ctx.stroke()
      ctx.beginPath(); ctx.moveTo(0,i*cell); ctx.lineTo(sz,i*cell); ctx.stroke()
    }
  }, [state])

  const r20 = state.successHist.slice(-20)
  const sRate = r20.length > 0 ? r20.filter(Boolean).length / r20.length : 0
  const status = sRate > 0.4 ? 'CONVERGING' : sRate > 0.1 ? 'LEARNING' : 'EXPLORING'
  const statusColor = sRate > 0.4 ? '#508870' : sRate > 0.1 ? '#907858' : '#5098a8'

  return (
    <div style={{
      display:'flex', gap:'16px', height:'100%',
      padding:'8px 14px', alignItems:'center', justifyContent:'center',
    }}>
      {/* Grid */}
      <div style={{ display:'flex', flexDirection:'column', gap:'4px', alignItems:'center', flexShrink:0 }}>
        <canvas ref={gridRef} style={{ borderRadius:'3px', border:'1px solid rgba(48,80,120,0.08)' }} />
        <span style={{ fontFamily:'var(--mono)', fontSize:'8px', color:'var(--t-dim)', letterSpacing:'1.5px' }}>
          MAZE #{state.uniqueMazes}
        </span>
      </div>

      {/* Stats */}
      <div style={{ display:'flex', flexDirection:'column', gap:'8px', flex:1, minWidth:0 }}>
        <div style={{ display:'flex', gap:'16px', flexWrap:'wrap' }}>
          <Stat label="EPISODE" value={String(state.episode)} size={16} />
          <Stat label="SUCCESS" value={`${(sRate*100).toFixed(0)}%`} size={16} color={statusColor} />
          <Stat label="SOLVED" value={String(state.mazesSolved)} size={16} color="#508870" />
          <Stat label="EPSILON" value={state.epsilon.toFixed(3)} size={12} />
          <Stat label="BEST" value={state.bestReward.toFixed(1)} size={12} color="#3098a8" />
          <Stat label="FEATURES" value={String(state.qTable.size)} size={12} />
        </div>

        <div>
          <span style={{ fontFamily:'var(--mono)', fontSize:'8px', color:'var(--t-dim)', letterSpacing:'1.5px' }}>
            REWARD CURVE
          </span>
          <div style={{ marginTop:'3px' }}><RewardCurve history={state.rewardHist} /></div>
        </div>

        <div style={{ display:'flex', alignItems:'center', gap:'8px' }}>
          <div style={{
            width:'5px', height:'5px', borderRadius:'50%',
            background:statusColor, boxShadow:`0 0 6px ${statusColor}`,
            animation:'pulse 2s ease-in-out infinite',
          }} />
          <span style={{ fontFamily:'var(--mono)', fontSize:'11px', color:statusColor, letterSpacing:'1.5px', fontWeight:500 }}>
            {status}
          </span>
          <span style={{ fontFamily:'var(--mono)', fontSize:'9px', color:'var(--t-sec)' }}>
            {ACT_SYMS[state.lastAction]} step {state.step}
          </span>
        </div>

        <span style={{ fontFamily:'var(--mono)', fontSize:'8px', color:'var(--t-dim)', letterSpacing:'0.5px' }}>
          RANDOM SOLVABLE MAZE EVERY EPISODE. FEATURE-BASED Q-LEARNING TRANSFERS ACROSS LAYOUTS.
        </span>
      </div>
    </div>
  )
}

function Stat({ label, value, color, size }: { label:string; value:string; color?:string; size?:number }) {
  return (
    <div style={{ display:'flex', flexDirection:'column', gap:'1px' }}>
      <span style={{ fontFamily:'var(--mono)', fontSize:'7px', color:'var(--t-dim)', letterSpacing:'1px' }}>{label}</span>
      <span style={{ fontFamily:'var(--mono)', fontSize:`${size||12}px`, color:color||'var(--t-max)', fontWeight:500, fontVariantNumeric:'tabular-nums' }}>{value}</span>
    </div>
  )
}
