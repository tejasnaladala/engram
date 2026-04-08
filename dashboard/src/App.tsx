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

// ──────────────────────────────────────────────────────────────────
// Demo data generator — produces physiologically plausible synthetic
// data when no live server is connected
// ──────────────────────────────────────────────────────────────────
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
    reason: Math.random() < 0.5 ? 'HardConstraint: Contraindicated trajectory' : { LearnedInhibition: { confidence: 0.85 } },
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

// ──────────────────────────────────────────────────────────────────
// Scan Panel Header — reusable panel frame with clinical title bar
// ──────────────────────────────────────────────────────────────────
function PanelHeader({ title, badge }: { title: string; badge?: string }) {
  return (
    <div className="panel-header">
      <span className="panel-header__title">{title}</span>
      {badge && <span className="panel-header__badge">{badge}</span>}
    </div>
  )
}

// ──────────────────────────────────────────────────────────────────
// Main Application
// ──────────────────────────────────────────────────────────────────
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
    }, 33)

    return () => clearInterval(interval)
  }, [])

  // WebSocket connection attempt
  useEffect(() => {
    let ws: WebSocket | null = null
    const connect = () => {
      try {
        ws = new WebSocket('ws://localhost:9000/ws')
        ws.binaryType = 'arraybuffer'
        ws.onopen = () => setConnected(true)
        ws.onclose = () => setConnected(false)
        ws.onerror = () => {}
      } catch {
        // demo mode
      }
    }
    connect()
    return () => ws?.close()
  }, [])

  // ── Loading state — scan initialization ──
  if (!snapshot) {
    return (
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        height: '100vh',
        background: 'var(--surface-0)',
        flexDirection: 'column',
        gap: '12px',
      }}>
        <div style={{
          width: '8px',
          height: '8px',
          borderRadius: '50%',
          background: 'var(--accent-primary)',
          boxShadow: 'var(--glow-active)',
          animation: 'pulse-glow 1.5s ease-in-out infinite',
        }} />
        <span style={{
          fontFamily: 'var(--font-clinical)',
          fontSize: '11px',
          letterSpacing: '4px',
          color: 'var(--accent-primary)',
        }}>
          ENGRAM
        </span>
        <span style={{
          fontFamily: 'var(--font-clinical)',
          fontSize: '9px',
          letterSpacing: '2px',
          color: 'var(--text-tertiary)',
        }}>
          INITIALIZING COGNITIVE RUNTIME
        </span>
      </div>
    )
  }

  // ── Main layout — neuronavigation workstation ──
  return (
    <div style={{
      height: '100vh',
      display: 'flex',
      flexDirection: 'column',
      background: 'var(--surface-0)',
    }}>
      {/* Diagnostic header bar */}
      <MetricsBar metrics={snapshot.metrics} connected={connected} />

      {/* Main scan panel grid — modeled after DICOM quad-view layout */}
      <div style={{
        flex: 1,
        display: 'grid',
        gridTemplateColumns: '1fr 2fr',
        gridTemplateRows: '1.2fr 1fr',
        gap: 'var(--panel-gap)',
        padding: 'var(--panel-gap)',
        minHeight: 0,
      }}>
        {/* ═══ Panel 1: Volumetric Brain View (top-left) ═══ */}
        <div className="scan-panel scan-panel--crosshair">
          <PanelHeader title="VOLUMETRIC SCAN" badge="3D" />
          <BrainVisualization modules={snapshot.modules} />
        </div>

        {/* ═══ Panel 2: Electrophysiology Raster (top-right) ═══ */}
        <div className="scan-panel">
          <PanelHeader
            title="MULTI-CHANNEL ELECTROPHYSIOLOGY"
            badge={`${snapshot.modules.reduce((s, m) => s + m.active_count, 0)} ACTIVE`}
          />
          <SpikeRaster spikeHistory={spikeHistory} />
        </div>

        {/* ═══ Panel 3: Bottom-left split — Module Readouts + Safety ═══ */}
        <div style={{
          display: 'grid',
          gridTemplateRows: '1fr 1fr',
          gap: 'var(--panel-gap)',
        }}>
          {/* Module Activity — regional activation readout */}
          <div className="scan-panel">
            <PanelHeader title="REGIONAL ACTIVATION" badge="RT" />
            <ModuleActivity modules={snapshot.modules} />
          </div>

          {/* Safety Kernel Event Log */}
          <div className="scan-panel">
            <PanelHeader
              title="SAFETY MONITOR"
              badge={snapshot.metrics.total_vetoes > 0 ? `${snapshot.metrics.total_vetoes} EVENTS` : 'NOMINAL'}
            />
            <SafetyLog vetoes={vetoLog} />
          </div>
        </div>

        {/* ═══ Panel 4: Bottom-right split — Error Signal + Memory Map ═══ */}
        <div style={{
          display: 'grid',
          gridTemplateColumns: '1.5fr 1fr',
          gap: 'var(--panel-gap)',
        }}>
          {/* Prediction Error — clinical waveform monitor */}
          <div className="scan-panel">
            <PanelHeader title="PREDICTION ERROR SIGNAL" badge="PE" />
            <PredictionError history={errorHistory} />
          </div>

          {/* Memory Formation — functional activation map */}
          <div className="scan-panel scan-panel--crosshair">
            <PanelHeader title="MEMORY ACTIVATION MAP" badge="fMEM" />
            <MemoryHeatMap formations={snapshot.memory_formations} tick={snapshot.metrics.tick} />
          </div>
        </div>
      </div>
    </div>
  )
}
