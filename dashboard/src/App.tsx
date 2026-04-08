import { useState, useEffect, useRef } from 'react'
import { MODULE_NAMES, MODULE_COLOR_ARRAY } from './lib/theme'
import type { RuntimeSnapshot, SpikeEvent, VetoEvent } from './lib/protocol'
import MetricsBar from './components/MetricsBar'
import SpikeRaster from './components/SpikeRaster'
import ModuleActivity from './components/ModuleActivity'
import PredictionError from './components/PredictionError'
import MemoryHeatMap from './components/MemoryHeatMap'
import SafetyLog from './components/SafetyLog'
import BrainVisualization from './components/BrainVisualization'

function generateDemoSnapshot(tick: number): RuntimeSnapshot {
  const t = tick * 0.001
  const modules = MODULE_NAMES.map((_, i) => ({
    module_id: ['Sensory','AssociativeMemory','PredictiveError','EpisodicMemory','ActionSelector','SafetyKernel'][i],
    neuron_count: [128,256,64,64,128,32][i],
    active_count: Math.floor(Math.random() * [30,50,15,10,25,5][i]),
    avg_potential: -65 + Math.random() * 10,
    spike_rate: Math.random() * 500,
    activity_level: 0.1 + Math.random() * 0.7 * (0.5 + 0.5 * Math.sin(t * 2 + i)),
  }))

  const spikes: SpikeEvent[] = []
  for (let m = 0; m < 6; m++) {
    const count = Math.floor(Math.random() * [15,20,8,5,12,3][m])
    for (let s = 0; s < count; s++) {
      spikes.push({
        source_module: ['Sensory','AssociativeMemory','PredictiveError','EpisodicMemory','ActionSelector','SafetyKernel'][m],
        neuron_id: Math.floor(Math.random() * modules[m].neuron_count),
        timestamp: tick + Math.random(),
        strength: 0.5 + Math.random() * 0.5,
      })
    }
  }

  const vetoes: VetoEvent[] = Math.random() < 0.025 ? [{
    timestamp: tick,
    vetoed_action: { action_id: Math.floor(Math.random() * 4), confidence: Math.random(), is_reflex: false },
    reason: Math.random() < 0.5 ? 'HardConstraint: Contraindicated' : { LearnedInhibition: { confidence: 0.85 } },
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
    memory_formations: Math.random() < 0.12 ? [{
      timestamp: tick,
      location_id: Math.floor(Math.random() * 1000),
      strength: 0.3 + Math.random() * 0.7,
      memory_type: 'Associative',
    }] : [],
    current_action: Math.floor(Math.random() * 4),
    total_reward: -0.01 * tick + Math.random() * 10,
    agent_position: [5 + Math.floor(Math.random() * 3), 5 + Math.floor(Math.random() * 3)],
  }
}

function PanelHdr({ title, color, tag }: { title: string; color?: string; tag?: string }) {
  return (
    <div className="panel-hdr">
      {color && <div className="panel-hdr__dot" style={{ background: color, boxShadow: `0 0 6px ${color}` }} />}
      <span className="panel-hdr__title">{title}</span>
      {tag && <span className="panel-hdr__tag">{tag}</span>}
    </div>
  )
}

export default function App() {
  const [snapshot, setSnapshot] = useState<RuntimeSnapshot | null>(null)
  const [connected, setConnected] = useState(false)
  const [spikeHistory, setSpikeHistory] = useState<SpikeEvent[][]>([])
  const [errorHistory, setErrorHistory] = useState<{ tick: number; error: number }[]>([])
  const [vetoLog, setVetoLog] = useState<VetoEvent[]>([])
  const tickRef = useRef(0)

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
    }, 33)
    return () => clearInterval(interval)
  }, [])

  useEffect(() => {
    let ws: WebSocket | null = null
    try {
      ws = new WebSocket('ws://localhost:9000/ws')
      ws.binaryType = 'arraybuffer'
      ws.onopen = () => setConnected(true)
      ws.onclose = () => setConnected(false)
      ws.onerror = () => {}
    } catch { /* demo mode */ }
    return () => ws?.close()
  }, [])

  if (!snapshot) {
    return (
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        height: '100vh', background: 'var(--void)', flexDirection: 'column', gap: '16px',
      }}>
        <div style={{
          width: '2px', height: '2px', borderRadius: '50%',
          background: 'var(--cyan)', boxShadow: 'var(--cyan-glow)',
          animation: 'pulse 1.5s ease-in-out infinite',
        }} />
        <span style={{ fontFamily: 'var(--font-mono)', fontSize: '10px', letterSpacing: '6px', color: 'var(--cyan)' }}>
          ENGRAM
        </span>
        <span style={{ fontFamily: 'var(--font-mono)', fontSize: '8px', letterSpacing: '3px', color: 'var(--text-dim)' }}>
          NEURAL INTERFACE LOADING
        </span>
      </div>
    )
  }

  return (
    <div style={{ height: '100vh', display: 'flex', flexDirection: 'column', background: 'var(--void)', position: 'relative' }}>
      {/* Ambient grid overlay across entire interface */}
      <div className="grid-overlay" />

      {/* Diagnostic header */}
      <MetricsBar metrics={snapshot.metrics} connected={connected} />

      {/* Main workspace */}
      <div style={{
        flex: 1, display: 'grid',
        gridTemplateColumns: '280px 1fr 240px',
        gridTemplateRows: '1fr 1fr',
        gap: 'var(--gap)',
        padding: '1px',
        minHeight: 0,
        position: 'relative',
        zIndex: 1,
      }}>
        {/* ══ LEFT COLUMN: Brain + Modules ══ */}
        {/* Volumetric Brain Scan */}
        <div className="scan-panel" style={{ gridRow: '1 / 2' }}>
          <PanelHdr title="VOLUMETRIC SCAN" color="var(--cyan)" tag="3D" />
          <BrainVisualization modules={snapshot.modules} />
        </div>

        {/* Regional Activation */}
        <div className="scan-panel scan-panel--br" style={{ gridRow: '2 / 3' }}>
          <PanelHdr title="REGIONAL ACTIVATION" tag="RT" />
          <ModuleActivity modules={snapshot.modules} />
        </div>

        {/* ══ CENTER: Spike Raster (full height) ══ */}
        <div className="scan-panel" style={{ gridRow: '1 / 3' }}>
          <PanelHdr title="MULTI-ELECTRODE ARRAY" color="var(--mod-s1)" tag={`${snapshot.modules.reduce((s,m)=>s+m.active_count,0)} UNITS`} />
          <SpikeRaster spikeHistory={spikeHistory} />
        </div>

        {/* ══ RIGHT COLUMN: Error + Memory + Safety ══ */}
        <div style={{ gridRow: '1 / 3', display: 'flex', flexDirection: 'column', gap: 'var(--gap)' }}>
          {/* Prediction Error Trace */}
          <div className="scan-panel" style={{ flex: '1' }}>
            <PanelHdr title="PREDICTION ERROR" color="var(--mod-pe)" tag="PE" />
            <PredictionError history={errorHistory} />
          </div>

          {/* Memory Activation Map */}
          <div className="scan-panel scan-panel--br" style={{ flex: '1' }}>
            <PanelHdr title="MEMORY MAP" color="var(--mod-hpc)" tag="fMEM" />
            <MemoryHeatMap formations={snapshot.memory_formations} tick={snapshot.metrics.tick} />
          </div>

          {/* Safety Monitor */}
          <div className="scan-panel" style={{ flex: '0.7' }}>
            <PanelHdr title="SAFETY GOVERNOR" color="var(--mod-bst)" tag={snapshot.metrics.total_vetoes > 0 ? `${snapshot.metrics.total_vetoes}` : 'OK'} />
            <SafetyLog vetoes={vetoLog} />
          </div>
        </div>
      </div>

      {/* Bottom status line */}
      <div style={{
        height: '18px', display: 'flex', alignItems: 'center',
        padding: '0 8px', gap: '16px',
        borderTop: '1px solid var(--border-ghost)',
        background: 'var(--surface-0)',
        flexShrink: 0, zIndex: 1,
      }}>
        <span style={{ fontFamily: 'var(--font-mono)', fontSize: '7px', letterSpacing: '1px', color: 'var(--text-dim)' }}>
          ENGRAM NCS v0.1.0
        </span>
        <span style={{ fontFamily: 'var(--font-mono)', fontSize: '7px', color: 'var(--text-dim)' }}>
          LIF/STDP RUNTIME
        </span>
        <span style={{ fontFamily: 'var(--font-mono)', fontSize: '7px', color: 'var(--text-dim)' }}>
          672 NEURONS | 6 REGIONS | 4 PATHWAYS
        </span>
        <div style={{ flex: 1 }} />
        <span style={{ fontFamily: 'var(--font-mono)', fontSize: '7px', color: 'var(--text-dim)' }}>
          EVENT-DRIVEN COGNITIVE ARCHITECTURE
        </span>
      </div>
    </div>
  )
}
