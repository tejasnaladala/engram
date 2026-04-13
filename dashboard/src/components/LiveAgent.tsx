import { useState, useEffect, useRef, useCallback } from 'react'

/*
  Live learning agent that runs a grid world inside the dashboard.
  The agent learns in real-time -- you watch reward curves improve,
  the path get shorter, and the brain metrics respond to actual task events.
*/

const GRID = 8
const WALL = 1
const HAZARD = 3
const ACTIONS = [
  [0, -1], // up
  [1, 0],  // right
  [0, 1],  // down
  [-1, 0], // left
]
const ACT_NAMES = ['\u2191', '\u2192', '\u2193', '\u2190']

interface AgentState {
  grid: number[][]
  agentX: number
  agentY: number
  goalX: number
  goalY: number
  episode: number
  step: number
  totalReward: number
  episodeReward: number
  rewardHistory: number[]
  successHistory: boolean[]
  path: [number, number][]
  qTable: Map<string, number[]>
  epsilon: number
  lastAction: number
  solved: boolean
  bestReward: number
  learningRate: number
}

function initGrid(): number[][] {
  const g: number[][] = Array.from({ length: GRID }, () => Array(GRID).fill(0))
  // Border walls
  for (let i = 0; i < GRID; i++) { g[0][i] = WALL; g[GRID-1][i] = WALL; g[i][0] = WALL; g[i][GRID-1] = WALL }
  // Interior walls
  g[2][3] = WALL; g[3][3] = WALL; g[4][3] = WALL
  g[2][5] = WALL; g[3][5] = WALL
  g[5][2] = WALL; g[5][6] = WALL
  // Hazards
  g[3][4] = HAZARD; g[5][5] = HAZARD
  return g
}

function stateKey(x: number, y: number): string { return `${x},${y}` }

function createAgent(): AgentState {
  return {
    grid: initGrid(),
    agentX: 1, agentY: 1,
    goalX: GRID - 2, goalY: GRID - 2,
    episode: 0, step: 0,
    totalReward: 0, episodeReward: 0,
    rewardHistory: [], successHistory: [],
    path: [[1, 1]],
    qTable: new Map(),
    epsilon: 0.4,
    learningRate: 0.15,
    lastAction: 0,
    solved: false,
    bestReward: -999,
  }
}

function getQ(state: AgentState, key: string): number[] {
  if (!state.qTable.has(key)) state.qTable.set(key, [0, 0, 0, 0])
  return state.qTable.get(key)!
}

function agentStep(state: AgentState): AgentState {
  const s = { ...state }
  const key = stateKey(s.agentX, s.agentY)
  const qvals = getQ(s, key)

  // Epsilon-greedy
  let action: number
  if (Math.random() < s.epsilon) {
    action = Math.floor(Math.random() * 4)
  } else {
    action = qvals.indexOf(Math.max(...qvals))
  }
  s.lastAction = action

  const [dx, dy] = ACTIONS[action]
  const nx = s.agentX + dx
  const ny = s.agentY + dy

  let reward = -0.02
  let done = false

  if (nx >= 0 && nx < GRID && ny >= 0 && ny < GRID && s.grid[ny][nx] !== WALL) {
    s.agentX = nx
    s.agentY = ny
    if (s.grid[ny][nx] === HAZARD) reward = -1.0
    if (nx === s.goalX && ny === s.goalY) { reward = 10.0; done = true; s.solved = true }
  } else {
    reward = -0.3
  }

  s.episodeReward += reward
  s.totalReward += reward
  s.step += 1
  s.path = [...s.path, [s.agentX, s.agentY]]

  // Q-learning update
  const nextKey = stateKey(s.agentX, s.agentY)
  const nextQ = getQ(s, nextKey)
  const target = reward + (done ? 0 : 0.99 * Math.max(...nextQ))
  const newQ = [...qvals]
  newQ[action] += s.learningRate * (target - newQ[action])
  s.qTable.set(key, newQ)

  // Episode boundary
  if (done || s.step > GRID * GRID * 2) {
    s.rewardHistory = [...s.rewardHistory, s.episodeReward]
    s.successHistory = [...s.successHistory, done]
    if (s.episodeReward > s.bestReward) s.bestReward = s.episodeReward
    s.episode += 1
    s.episodeReward = 0
    s.step = 0
    s.agentX = 1; s.agentY = 1
    s.path = [[1, 1]]
    s.epsilon = Math.max(0.05, s.epsilon * 0.995)
    s.solved = false
  }

  return s
}

// Reward curve mini chart
function RewardCurve({ history }: { history: number[] }) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const c = canvasRef.current
    if (!c) return
    const ctx = c.getContext('2d')!
    const w = 200, h = 50
    c.width = w * 2; c.height = h * 2
    c.style.width = w + 'px'; c.style.height = h + 'px'
    ctx.scale(2, 2)

    ctx.fillStyle = '#060810'
    ctx.fillRect(0, 0, w, h)

    if (history.length < 2) return

    // Windowed average
    const window = 5
    const avg: number[] = []
    for (let i = 0; i < history.length; i++) {
      const start = Math.max(0, i - window + 1)
      const slice = history.slice(start, i + 1)
      avg.push(slice.reduce((a, b) => a + b, 0) / slice.length)
    }

    const min = Math.min(...avg)
    const max = Math.max(...avg)
    const range = max - min || 1

    ctx.strokeStyle = '#3098a8'
    ctx.lineWidth = 1
    ctx.beginPath()
    for (let i = 0; i < avg.length; i++) {
      const x = (i / (avg.length - 1)) * w
      const y = h - ((avg[i] - min) / range) * (h - 8) - 4
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y)
    }
    ctx.stroke()

    // Zero line
    if (min < 0 && max > 0) {
      const zy = h - ((0 - min) / range) * (h - 8) - 4
      ctx.strokeStyle = 'rgba(48,152,168,0.15)'
      ctx.lineWidth = 0.5
      ctx.setLineDash([2, 3])
      ctx.beginPath(); ctx.moveTo(0, zy); ctx.lineTo(w, zy); ctx.stroke()
      ctx.setLineDash([])
    }
  }, [history])

  return <canvas ref={canvasRef} style={{ borderRadius: '2px' }} />
}

export default function LiveAgent({ onMetrics }: {
  onMetrics?: (data: {
    episode: number, reward: number, epsilon: number,
    successRate: number, solved: boolean, action: number
  }) => void
}) {
  const [state, setState] = useState<AgentState>(createAgent)
  const canvasRef = useRef<HTMLCanvasElement>(null)

  // Run the agent
  useEffect(() => {
    const interval = setInterval(() => {
      setState(prev => {
        const next = agentStep(prev)
        onMetrics?.({
          episode: next.episode,
          reward: next.episodeReward,
          epsilon: next.epsilon,
          successRate: next.successHistory.length > 0
            ? next.successHistory.slice(-20).filter(Boolean).length / Math.min(20, next.successHistory.length)
            : 0,
          solved: next.solved,
          action: next.lastAction,
        })
        return next
      })
    }, 16) // ~60 steps/sec for visible learning
    return () => clearInterval(interval)
  }, [onMetrics])

  // Render grid
  useEffect(() => {
    const c = canvasRef.current
    if (!c) return
    const ctx = c.getContext('2d')!
    const size = 160
    const cell = size / GRID
    c.width = size * 2; c.height = size * 2
    c.style.width = size + 'px'; c.style.height = size + 'px'
    ctx.scale(2, 2)

    // Background
    ctx.fillStyle = '#060810'
    ctx.fillRect(0, 0, size, size)

    // Cells
    for (let y = 0; y < GRID; y++) {
      for (let x = 0; x < GRID; x++) {
        const v = state.grid[y][x]
        if (v === WALL) ctx.fillStyle = '#181e2c'
        else if (v === HAZARD) ctx.fillStyle = '#1a1218'
        else ctx.fillStyle = '#0a0d16'
        ctx.fillRect(x * cell + 0.5, y * cell + 0.5, cell - 1, cell - 1)

        if (v === HAZARD) {
          ctx.fillStyle = '#906060'
          ctx.font = `${cell * 0.5}px sans-serif`
          ctx.textAlign = 'center'; ctx.textBaseline = 'middle'
          ctx.fillText('!', x * cell + cell / 2, y * cell + cell / 2)
        }
      }
    }

    // Path trace
    for (let i = 0; i < state.path.length - 1; i++) {
      const [px, py] = state.path[i]
      ctx.fillStyle = 'rgba(48,152,168,0.08)'
      ctx.fillRect(px * cell + cell * 0.3, py * cell + cell * 0.3, cell * 0.4, cell * 0.4)
    }

    // Goal
    ctx.fillStyle = '#508870'
    ctx.beginPath()
    ctx.arc(state.goalX * cell + cell / 2, state.goalY * cell + cell / 2, cell * 0.3, 0, Math.PI * 2)
    ctx.fill()

    // Agent
    ctx.fillStyle = state.solved ? '#508870' : '#3098a8'
    ctx.shadowColor = state.solved ? '#508870' : '#3098a8'
    ctx.shadowBlur = 6
    ctx.beginPath()
    ctx.arc(state.agentX * cell + cell / 2, state.agentY * cell + cell / 2, cell * 0.25, 0, Math.PI * 2)
    ctx.fill()
    ctx.shadowBlur = 0

    // Grid lines
    ctx.strokeStyle = 'rgba(48,80,120,0.06)'
    ctx.lineWidth = 0.5
    for (let i = 0; i <= GRID; i++) {
      ctx.beginPath(); ctx.moveTo(i * cell, 0); ctx.lineTo(i * cell, size); ctx.stroke()
      ctx.beginPath(); ctx.moveTo(0, i * cell); ctx.lineTo(size, i * cell); ctx.stroke()
    }
  }, [state])

  const recent20 = state.successHistory.slice(-20)
  const successRate = recent20.length > 0 ? recent20.filter(Boolean).length / recent20.length : 0

  return (
    <div style={{
      display: 'flex', gap: '10px', height: '100%',
      padding: '6px 8px', alignItems: 'flex-start',
    }}>
      {/* Grid world */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '4px', alignItems: 'center' }}>
        <canvas ref={canvasRef} style={{ borderRadius: '2px' }} />
        <span style={{ fontFamily: 'var(--mono)', fontSize: '7px', color: 'var(--t-dim)', letterSpacing: '1px' }}>
          ENVIRONMENT
        </span>
      </div>

      {/* Stats + reward curve */}
      <div style={{
        display: 'flex', flexDirection: 'column', gap: '6px', flex: 1,
        fontFamily: 'var(--mono)', fontSize: '10px',
      }}>
        {/* Key metrics */}
        <div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap' }}>
          <Stat label="EPISODE" value={String(state.episode)} />
          <Stat label="SUCCESS" value={`${(successRate * 100).toFixed(0)}%`}
            color={successRate > 0.5 ? '#508870' : successRate > 0.2 ? '#907858' : '#906060'} />
          <Stat label="EPSILON" value={state.epsilon.toFixed(3)} />
          <Stat label="BEST" value={state.bestReward.toFixed(1)} color="#3098a8" />
          <Stat label="ACTION" value={ACT_NAMES[state.lastAction]} />
          <Stat label="Q-STATES" value={String(state.qTable.size)} />
        </div>

        {/* Reward curve */}
        <div>
          <span style={{ fontSize: '7px', color: 'var(--t-dim)', letterSpacing: '1.5px' }}>
            REWARD CURVE (EPISODE AVG)
          </span>
          <div style={{ marginTop: '2px' }}>
            <RewardCurve history={state.rewardHistory} />
          </div>
        </div>

        {/* Learning status */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <div style={{
            width: '4px', height: '4px', borderRadius: '50%',
            background: successRate > 0.3 ? '#508870' : '#3098a8',
            boxShadow: `0 0 4px ${successRate > 0.3 ? '#508870' : '#3098a8'}`,
            animation: 'pulse 2s ease-in-out infinite',
          }} />
          <span style={{ fontSize: '8px', color: 'var(--t-sec)', letterSpacing: '1px' }}>
            {successRate > 0.5 ? 'CONVERGING' : successRate > 0.2 ? 'LEARNING' : 'EXPLORING'}
          </span>
        </div>
      </div>
    </div>
  )
}

function Stat({ label, value, color }: { label: string; value: string; color?: string }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1px' }}>
      <span style={{ fontSize: '7px', color: 'var(--t-dim)', letterSpacing: '1px' }}>{label}</span>
      <span style={{ fontSize: '12px', color: color || 'var(--t-max)', fontWeight: 500, fontVariantNumeric: 'tabular-nums' }}>{value}</span>
    </div>
  )
}
