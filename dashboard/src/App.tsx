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

function genDemo(tick: number): RuntimeSnapshot {
  const t = tick * 0.001
  const mids = ['Sensory','AssociativeMemory','PredictiveError','EpisodicMemory','ActionSelector','SafetyKernel']
  const ncounts = [128,256,64,64,128,32]
  const modules = MODULE_NAMES.map((_,i) => ({
    module_id: mids[i],
    neuron_count: ncounts[i],
    active_count: Math.floor(Math.random() * [30,50,15,10,25,5][i]),
    avg_potential: -65 + Math.random() * 10,
    spike_rate: Math.random() * 500,
    activity_level: 0.08 + Math.random() * 0.72 * (0.5 + 0.5 * Math.sin(t * 2 + i * 1.1)),
  }))
  const spikes: SpikeEvent[] = []
  for (let m = 0; m < 6; m++) {
    const c = Math.floor(Math.random() * [15,20,8,5,12,3][m])
    for (let s = 0; s < c; s++)
      spikes.push({ source_module: mids[m], neuron_id: Math.floor(Math.random()*ncounts[m]), timestamp: tick+Math.random(), strength: 0.5+Math.random()*0.5 })
  }
  const vetoes: VetoEvent[] = Math.random()<0.025 ? [{
    timestamp: tick,
    vetoed_action: { action_id: Math.floor(Math.random()*4), confidence: Math.random(), is_reflex: false },
    reason: Math.random()<0.5 ? 'HardConstraint: Forbidden trajectory' : { LearnedInhibition: { confidence: 0.85 } },
  }] : []

  return {
    metrics: { tick, sim_time: tick, ticks_per_second: 950+Math.random()*100,
      total_spikes: tick*27, total_vetoes: Math.floor(tick*0.02),
      active_synapses: 45000+Math.floor(Math.random()*5000),
      memory_bytes: 1024*1024*12, energy_units: tick*0.027 },
    modules, recent_spikes: spikes, recent_vetoes: vetoes,
    prediction_error: 0.3+0.2*Math.sin(t*0.5)+Math.random()*0.1,
    memory_formations: Math.random()<0.12 ? [{ timestamp:tick, location_id:Math.floor(Math.random()*1000), strength:0.3+Math.random()*0.7, memory_type:'Associative' }] : [],
    current_action: Math.floor(Math.random()*4),
    total_reward: -0.01*tick+Math.random()*10,
    agent_position: [5+Math.floor(Math.random()*3), 5+Math.floor(Math.random()*3)],
  }
}

function PH({ title, tag }: { title: string; tag?: string }) {
  return (
    <div className="phdr">
      <span className="phdr-title">{title}</span>
      {tag && <span className="phdr-tag">{tag}</span>}
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
      const snap = genDemo(tickRef.current)
      setSnapshot(snap)
      setSpikeHistory(prev => { const n=[...prev,snap.recent_spikes]; return n.length>300?n.slice(-300):n })
      setErrorHistory(prev => { const n=[...prev,{tick:snap.metrics.tick,error:snap.prediction_error}]; return n.length>300?n.slice(-300):n })
      if (snap.recent_vetoes.length>0) setVetoLog(prev => [...prev,...snap.recent_vetoes].slice(-50))
    }, 33)
    return () => clearInterval(interval)
  }, [])

  useEffect(() => {
    let ws: WebSocket|null = null
    try { ws = new WebSocket('ws://localhost:9000/ws'); ws.binaryType='arraybuffer'; ws.onopen=()=>setConnected(true); ws.onclose=()=>setConnected(false); ws.onerror=()=>{} } catch {}
    return () => ws?.close()
  }, [])

  if (!snapshot) return (
    <div style={{ display:'flex',alignItems:'center',justifyContent:'center',height:'100vh',background:'var(--void)',flexDirection:'column',gap:'16px' }}>
      <div style={{ width:'3px',height:'3px',borderRadius:'50%',background:'var(--cyan)',boxShadow:'var(--cyan-glow)',animation:'pulse 1.5s ease-in-out infinite' }} />
      <span style={{ fontFamily:'var(--mono)',fontSize:'11px',letterSpacing:'6px',color:'var(--cyan)',fontWeight:300 }}>ENGRAM</span>
    </div>
  )

  const totalActive = snapshot.modules.reduce((s,m)=>s+m.active_count, 0)

  /* LAYOUT: Brain is the hero center.
     ┌──────────────────────────────────────────────────┐
     │                  METRICS BAR                      │
     ├────────┬─────────────────────────────┬────────────┤
     │MODULES │                             │ PREDICTION │
     │activity│      BRAIN HOLOGRAM         │ ERROR      │
     │bars    │      (large, centered)      │ waveform   │
     │        │                             │            │
     ├────────┼──────────────┬──────────────┼────────────┤
     │SAFETY  │ SPIKE RASTER │ MEMORY MAP   │            │
     │log     │ (wide)       │ (square)     │            │
     └────────┴──────────────┴──────────────┴────────────┘
  */

  return (
    <div style={{ height:'100vh', display:'flex', flexDirection:'column', background:'var(--void)', position:'relative' }}>
      <div className="ambient-grid" />

      <MetricsBar metrics={snapshot.metrics} connected={connected} />

      <div style={{
        flex: 1, display: 'grid',
        gridTemplateColumns: '200px 1fr 200px',
        gridTemplateRows: '1.4fr 1fr',
        gap: '2px', padding: '2px',
        minHeight: 0, position: 'relative', zIndex: 1,
      }}>

        {/* LEFT TOP: Module activity */}
        <div className="panel" style={{ gridRow: '1/2' }}>
          <PH title="REGIONS" tag="RT" />
          <ModuleActivity modules={snapshot.modules} />
        </div>

        {/* CENTER TOP: Brain hologram -- the HERO */}
        <div className="panel" style={{ gridRow: '1/2' }}>
          <PH title="NEURAL TOPOLOGY" tag="3D" />
          <BrainVisualization modules={snapshot.modules} />
        </div>

        {/* RIGHT TOP: Prediction error */}
        <div className="panel" style={{ gridRow: '1/2' }}>
          <PH title="PREDICTION ERROR" tag="PE" />
          <PredictionError history={errorHistory} />
        </div>

        {/* LEFT BOTTOM: Safety log */}
        <div className="panel" style={{ gridRow: '2/3' }}>
          <PH title="SAFETY" tag={snapshot.metrics.total_vetoes > 0 ? `${snapshot.metrics.total_vetoes}` : 'OK'} />
          <SafetyLog vetoes={vetoLog} />
        </div>

        {/* CENTER BOTTOM: Spike raster */}
        <div className="panel" style={{ gridRow: '2/3' }}>
          <PH title="NEURAL ACTIVITY" tag={`${totalActive} UNITS`} />
          <SpikeRaster spikeHistory={spikeHistory} />
        </div>

        {/* RIGHT BOTTOM: Memory map */}
        <div className="panel" style={{ gridRow: '2/3' }}>
          <PH title="MEMORY MAP" tag="fMEM" />
          <MemoryHeatMap formations={snapshot.memory_formations} tick={snapshot.metrics.tick} />
        </div>
      </div>

      {/* Footer */}
      <div style={{
        height: '18px', display: 'flex', alignItems: 'center',
        padding: '0 10px', gap: '20px',
        borderTop: '1px solid var(--b-ghost)',
        background: 'var(--s0)', flexShrink: 0, zIndex: 1,
        fontFamily: 'var(--mono)', fontSize: '7px', letterSpacing: '1.5px', color: 'var(--t-dim)',
      }}>
        <span>ENGRAM NCS v0.1.0</span>
        <span style={{ color: 'var(--t-ghost)' }}>|</span>
        <span>672 NEURONS</span>
        <span style={{ color: 'var(--t-ghost)' }}>|</span>
        <span>6 REGIONS</span>
        <span style={{ color: 'var(--t-ghost)' }}>|</span>
        <span>EVENT-DRIVEN</span>
        <div style={{ flex: 1 }} />
        <span>SIGNAL PIPELINE ACTIVE</span>
        <div style={{ width:'4px', height:'4px', borderRadius:'50%', background:'var(--cyan)', boxShadow:'var(--cyan-glow)', animation:'pulse 3s ease-in-out infinite' }} />
      </div>
    </div>
  )
}
