import { useState, useEffect, useRef, useCallback } from 'react'
import { MODULE_NAMES, MODULE_COLOR_ARRAY } from './lib/theme'
import type { RuntimeSnapshot, SpikeEvent, VetoEvent } from './lib/protocol'
import MetricsBar from './components/MetricsBar'
import SpikeRaster from './components/SpikeRaster'
import ModuleActivity from './components/ModuleActivity'
import PredictionError from './components/PredictionError'
import MemoryHeatMap from './components/MemoryHeatMap'
import SafetyLog from './components/SafetyLog'
import BrainVisualization from './components/BrainVisualization'

// Simulated data generator for demo mode (when no server is connected)
function generateDemoSnapshot(tick: number): RuntimeSnapshot {
  const t = tick * 0.001
  const modules = MODULE_NAMES.map((_, i) => ({
    module_id: ['Sensory', 'AssociativeMemory', 'PredictiveError', 'EpisodicMemory', 'ActionSelector', 'SafetyKernel'][i],
    neuron_count: [128, 256, 64, 64, 128, 32][i],
    active_count: Math.floor(Math.random() * [30, 50, 15, 10, 25, 5][i]),
    avg_potential: -65 + Math.random() * 10,
    spike_rate: Math.random() * 500,
    activity_level: 0.1 + Math.random() * 0.7 * (0.5 + 0.5 * Math.sin(t * 2 + i)),
  }))

  const spikes: SpikeEvent[] = []
  for (let m = 0; m < 6; m++) {
    const count = Math.floor(Math.random() * [15, 20, 8, 5, 12, 3][m])
    for (let s = 0; s < count; s++) {
      spikes.push({
        source_module: ['Sensory', 'AssociativeMemory', 'PredictiveError', 'EpisodicMemory', 'ActionSelector', 'SafetyKernel'][m],
        neuron_id: Math.floor(Math.random() * modules[m].neuron_count),
        timestamp: tick + Math.random(),
        strength: 0.5 + Math.random() * 0.5,
      })
    }
  }

  const vetoes: VetoEvent[] = Math.random() < 0.02 ? [{
    timestamp: tick,
    vetoed_action: { action_id: Math.floor(Math.random() * 4), confidence: Math.random(), is_reflex: false },
    reason: Math.random() < 0.5 ? 'HardConstraint: Forbidden cell' : { LearnedInhibition: { confidence: 0.85 } },
  }] : []

  return {
    metrics: {
      tick,
      sim_time: tick,
      ticks_per_second: 950 + Math.random() * 100,
      total_spikes: tick * 27,
      total_vetoes: Math.floor(tick * 0.02),
      active_synapses: 45000 + Math.floor(Math.random() * 5000),
      memory_bytes: 1024 * 1024 * 12,
      energy_units: tick * 0.027,
    },
    modules,
    recent_spikes: spikes,
    recent_vetoes: vetoes,
    prediction_error: 0.3 + 0.2 * Math.sin(t * 0.5) + Math.random() * 0.1,
    memory_formations: Math.random() < 0.1 ? [{
      timestamp: tick,
      location_id: Math.floor(Math.random() * 1000),
      strength: Math.random(),
      memory_type: 'Associative',
    }] : [],
    current_action: Math.floor(Math.random() * 4),
    total_reward: -0.01 * tick + Math.random() * 10,
    agent_position: [5 + Math.floor(Math.random() * 3), 5 + Math.floor(Math.random() * 3)],
  }
}

export default function App() {
  const [snapshot, setSnapshot] = useState<RuntimeSnapshot | null>(null)
  const [connected, setConnected] = useState(false)
  const [spikeHistory, setSpikeHistory] = useState<SpikeEvent[][]>([])
  const [errorHistory, setErrorHistory] = useState<{ tick: number; error: number }[]>([])
  const [vetoLog, setVetoLog] = useState<VetoEvent[]>([])
  const tickRef = useRef(0)

  // Demo mode: generate simulated data
  useEffect(() => {
    const interval = setInterval(() => {
      tickRef.current += 10
      const snap = generateDemoSnapshot(tickRef.current)
      setSnapshot(snap)
      setSpikeHistory(prev => {
        const next = [...prev, snap.recent_spikes]
        return next.length > 300 ? next.slice(-300) : next
      })
      setErrorHistory(prev => {
        const next = [...prev, { tick: snap.metrics.tick, error: snap.prediction_error }]
        return next.length > 300 ? next.slice(-300) : next
      })
      if (snap.recent_vetoes.length > 0) {
        setVetoLog(prev => [...prev, ...snap.recent_vetoes].slice(-50))
      }
    }, 33) // ~30fps

    return () => clearInterval(interval)
  }, [])

  // Try WebSocket connection
  useEffect(() => {
    let ws: WebSocket | null = null
    const connect = () => {
      try {
        ws = new WebSocket('ws://localhost:9000/ws')
        ws.binaryType = 'arraybuffer'
        ws.onopen = () => setConnected(true)
        ws.onclose = () => setConnected(false)
        ws.onerror = () => {} // silent - demo mode handles it
      } catch {
        // demo mode
      }
    }
    connect()
    return () => ws?.close()
  }, [])

  if (!snapshot) {
    return (
      <div className="flex items-center justify-center h-screen" style={{ background: '#0a0a0f' }}>
        <div className="text-center">
          <div className="text-5xl mb-4">🧠</div>
          <h1 className="text-2xl font-bold mb-2" style={{ color: '#e0e0ff' }}>Engram</h1>
          <p style={{ color: '#8888aa' }}>Initializing cognitive runtime...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="h-screen flex flex-col" style={{ background: '#0a0a0f' }}>
      {/* Top Metrics Bar */}
      <MetricsBar metrics={snapshot.metrics} connected={connected} />

      {/* Main Content Grid */}
      <div className="flex-1 grid grid-cols-12 grid-rows-6 gap-1 p-1 min-h-0">
        {/* Brain 3D — top left */}
        <div className="col-span-4 row-span-3 rounded-lg overflow-hidden" style={{ background: '#12121a', border: '1px solid rgba(255,255,255,0.05)' }}>
          <div className="px-3 py-1.5 text-xs font-semibold uppercase tracking-wider" style={{ color: '#8888aa', borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
            Brain Regions
          </div>
          <BrainVisualization modules={snapshot.modules} />
        </div>

        {/* Spike Raster — top right */}
        <div className="col-span-8 row-span-3 rounded-lg overflow-hidden" style={{ background: '#12121a', border: '1px solid rgba(255,255,255,0.05)' }}>
          <div className="px-3 py-1.5 text-xs font-semibold uppercase tracking-wider" style={{ color: '#8888aa', borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
            Spike Raster
          </div>
          <SpikeRaster spikeHistory={spikeHistory} />
        </div>

        {/* Module Activity — bottom left */}
        <div className="col-span-3 row-span-3 rounded-lg overflow-hidden" style={{ background: '#12121a', border: '1px solid rgba(255,255,255,0.05)' }}>
          <div className="px-3 py-1.5 text-xs font-semibold uppercase tracking-wider" style={{ color: '#8888aa', borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
            Module Activity
          </div>
          <ModuleActivity modules={snapshot.modules} />
        </div>

        {/* Prediction Error — bottom center */}
        <div className="col-span-5 row-span-3 rounded-lg overflow-hidden" style={{ background: '#12121a', border: '1px solid rgba(255,255,255,0.05)' }}>
          <div className="px-3 py-1.5 text-xs font-semibold uppercase tracking-wider" style={{ color: '#8888aa', borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
            Prediction Error
          </div>
          <PredictionError history={errorHistory} />
        </div>

        {/* Memory HeatMap + Safety Log — bottom right */}
        <div className="col-span-4 row-span-3 flex flex-col gap-1">
          <div className="flex-1 rounded-lg overflow-hidden" style={{ background: '#12121a', border: '1px solid rgba(255,255,255,0.05)' }}>
            <div className="px-3 py-1.5 text-xs font-semibold uppercase tracking-wider" style={{ color: '#8888aa', borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
              Memory Formation
            </div>
            <MemoryHeatMap formations={snapshot.memory_formations} tick={snapshot.metrics.tick} />
          </div>
          <div className="flex-1 rounded-lg overflow-hidden" style={{ background: '#12121a', border: '1px solid rgba(255,255,255,0.05)' }}>
            <div className="px-3 py-1.5 text-xs font-semibold uppercase tracking-wider" style={{ color: '#8888aa', borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
              Safety Log
            </div>
            <SafetyLog vetoes={vetoLog} />
          </div>
        </div>
      </div>
    </div>
  )
}
