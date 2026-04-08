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

/* ─────────────────────────────────────────────────────────────
   DEMO DATA — synthetic RuntimeSnapshot generator
   Produces physiologically plausible data at 30fps
   ───────────────────────────────────────────────────────────── */
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

/* ─────────────────────────────────────────────────────────────
   PANEL HEADER — consistent across all stations
   ───────────────────────────────────────────────────────────── */
function PH({ title, color, tag, stage }: { title: string; color?: string; tag?: string; stage?: string }) {
  return (
    <div className="phdr">
      {color && <div className="phdr-dot" style={{ background: color, boxShadow: `0 0 6px ${color}` }} />}
      {stage && <span style={{ fontFamily:'var(--mono)', fontSize:'7px', color:'var(--t-dim)', letterSpacing:'1px', marginRight:'2px' }}>{stage}</span>}
      <span className="phdr-title">{title}</span>
      {tag && <span className="phdr-tag">{tag}</span>}
    </div>
  )
}

/* ─────────────────────────────────────────────────────────────
   FLOW STAGE DIVIDER — labels between pipeline sections
   ───────────────────────────────────────────────────────────── */
function FlowDivider({ label }: { label: string }) {
  return (
    <div className="stage-label">
      <span>{label}</span>
    </div>
  )
}

/* ═════════════════════════════════════════════════════════════
   MAIN APP — THE SIGNAL-FLOW NARRATIVE
   Layout tells the story: input → process → remember → act → guard
   ═════════════════════════════════════════════════════════════ */
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
      <span style={{ fontFamily:'var(--mono)',fontSize:'10px',letterSpacing:'6px',color:'var(--cyan)',fontWeight:300 }}>ENGRAM</span>
      <span style={{ fontFamily:'var(--mono)',fontSize:'7px',letterSpacing:'3px',color:'var(--t-dim)' }}>INITIALIZING SIGNAL PIPELINE</span>
    </div>
  )

  const totalActive = snapshot.modules.reduce((s,m)=>s+m.active_count, 0)

  return (
    <div style={{ height:'100vh', display:'flex', flexDirection:'column', background:'var(--void)', position:'relative' }}>
      {/* Ambient engineering grid */}
      <div className="ambient-grid" />

      {/* ═══ SYSTEM STATUS BAR ═══ — the "mission control" strip */}
      <MetricsBar metrics={snapshot.metrics} connected={connected} />

      {/* ═══ MAIN WORKSPACE ═══ — the signal-flow pipeline
          Layout narrative:
          ┌────────────┬──────────────────────────┬───────────┐
          │ PERCEPTION │   NEURAL PROCESSING      │ COGNITION │
          │ 3D brain   │   spike raster (hero)    │ PE trace  │
          │ scan view  │   672 channels flowing   │ live err  │
          ├────────────┤                          ├───────────┤
          │ ACTIVATION │                          │ MEMORY    │
          │ per-region │                          │ activation│
          │ readouts   │                          ├───────────┤
          │            │                          │ SAFETY    │
          │            │                          │ governor  │
          └────────────┴──────────────────────────┴───────────┘
          Data flows left→right: sense→process→remember→act→guard
      */}
      <div style={{
        flex: 1, display: 'grid',
        gridTemplateColumns: '240px 1fr 220px',
        gridTemplateRows: '1fr 1fr',
        gap: '2px', padding: '2px',
        minHeight: 0, position: 'relative', zIndex: 1,
      }}>

        {/* ═══ COLUMN 1: PERCEPTION ═══ */}
        {/* Volumetric brain — the "what are we looking at" anchor */}
        <div className="panel" style={{ gridRow: '1/2' }}>
          <PH title="NEURAL TOPOLOGY" color="var(--c-input)" tag="3D" stage="01" />
          <BrainVisualization modules={snapshot.modules} />
        </div>

        {/* Regional readouts — per-region activation levels */}
        <div className="panel" style={{ gridRow: '2/3' }}>
          <PH title="REGIONAL ACTIVATION" tag="RT" stage="02" />
          <ModuleActivity modules={snapshot.modules} />
        </div>

        {/* ═══ COLUMN 2: PROCESSING ═══ — the hero, full height */}
        <div className="panel" style={{ gridRow: '1/3' }}>
          <PH title="NEURAL ACTIVITY" color="var(--c-input)" tag={`${totalActive} ACTIVE`} stage="03" />
          <SpikeRaster spikeHistory={spikeHistory} />
        </div>

        {/* ═══ COLUMN 3: COGNITION + MEMORY + SAFETY ═══ */}
        <div style={{ gridRow: '1/3', display: 'flex', flexDirection: 'column', gap: '2px' }}>
          {/* Prediction error — the learning signal */}
          <div className="panel" style={{ flex: 1 }}>
            <PH title="PREDICTION ERROR" color="var(--c-predict)" tag="PE" stage="04" />
            <PredictionError history={errorHistory} />
          </div>

          {/* Memory map — where experiences are stored */}
          <div className="panel" style={{ flex: 1 }}>
            <PH title="MEMORY MAP" color="var(--c-memory)" tag="fMEM" stage="05" />
            <MemoryHeatMap formations={snapshot.memory_formations} tick={snapshot.metrics.tick} />
          </div>

          {/* Safety governor — the final gate before action */}
          <div className="panel" style={{ flex: 0.6 }}>
            <PH title="SAFETY GOVERNOR" color="var(--c-guard)" tag={snapshot.metrics.total_vetoes > 0 ? `${snapshot.metrics.total_vetoes}` : 'OK'} stage="06" />
            <SafetyLog vetoes={vetoLog} />
          </div>
        </div>
      </div>

      {/* ═══ STATUS LINE ═══ — system identification footer */}
      <div style={{
        height: '18px', display: 'flex', alignItems: 'center',
        padding: '0 10px', gap: '20px',
        borderTop: '1px solid var(--b-ghost)',
        background: 'var(--s0)', flexShrink: 0, zIndex: 1,
      }}>
        <span style={{ fontFamily:'var(--mono)', fontSize:'7px', letterSpacing:'1px', color:'var(--t-dim)' }}>
          ENGRAM NCS v0.1.0
        </span>
        <span style={{ fontFamily:'var(--mono)', fontSize:'7px', color:'var(--t-ghost)' }}>│</span>
        <span style={{ fontFamily:'var(--mono)', fontSize:'7px', color:'var(--t-dim)' }}>
          LIF/STDP &middot; 672 NEURONS &middot; 6 REGIONS &middot; EVENT-DRIVEN
        </span>
        <div style={{ flex: 1 }} />
        <span style={{ fontFamily:'var(--mono)', fontSize:'7px', color:'var(--t-dim)' }}>
          SIGNAL PIPELINE ACTIVE
        </span>
        <div style={{ width:'4px', height:'4px', borderRadius:'50%', background:'var(--cyan)', boxShadow:'var(--cyan-glow)', animation:'breathe 3s ease-in-out infinite' }} />
      </div>
    </div>
  )
}
