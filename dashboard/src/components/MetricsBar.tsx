import type { RuntimeMetrics } from '../lib/protocol'

interface MetricsBarProps {
  metrics: RuntimeMetrics
  connected: boolean
}

function fmt(n: number): string {
  if (n >= 1e6) return (n/1e6).toFixed(1)+'M'
  if (n >= 1e3) return (n/1e3).toFixed(1)+'K'
  return n.toFixed(0)
}

/** Single metric readout — key:value in monospace */
function R({ k, v, c }: { k: string; v: string; c?: string }) {
  return (
    <span style={{ fontFamily:'var(--mono)', fontSize:'10px', letterSpacing:'0.3px' }}>
      <span style={{ color:'var(--t-dim)', fontSize:'9px' }}>{k}</span>
      <span style={{ color:'var(--t-ghost)', margin:'0 3px' }}>:</span>
      <span style={{ color: c || 'var(--t-max)', fontWeight: 500, fontVariantNumeric:'tabular-nums' }}>{v}</span>
    </span>
  )
}

export default function MetricsBar({ metrics, connected }: MetricsBarProps) {
  const hz = metrics.ticks_per_second
  const hzC = hz > 800 ? 'var(--c-output)' : hz > 400 ? 'var(--c-predict)' : 'var(--c-guard)'

  return (
    <div style={{
      display:'flex', alignItems:'center', gap:'14px',
      padding:'0 12px', height:'30px', flexShrink:0,
      background:'var(--s1)', borderBottom:'1px solid var(--b-dim)',
      position:'relative', zIndex:2,
    }}>
      {/* Top signal line */}
      <div style={{
        position:'absolute', top:0, left:0, right:0, height:'1px',
        background:`linear-gradient(90deg, transparent, ${connected?'var(--cyan-10)':'var(--b-sub)'} 50%, transparent)`,
      }} />

      {/* System identity */}
      <div style={{ display:'flex', alignItems:'center', gap:'6px' }}>
        <div style={{
          width:'4px', height:'4px', borderRadius:'1px', background:'var(--cyan)',
          boxShadow:'0 0 6px var(--cyan)', animation:'breathe 3s ease-in-out infinite',
        }} />
        <span style={{ fontFamily:'var(--mono)', fontSize:'10px', fontWeight:700, letterSpacing:'2.5px', color:'var(--cyan)' }}>
          ENGRAM
        </span>
      </div>

      <span style={{ color:'var(--t-ghost)', fontFamily:'var(--mono)', fontSize:'10px' }}>│</span>

      {/* Pipeline vitals — each metric is a signal in the flow */}
      <div style={{ display:'flex', alignItems:'center', gap:'3px' }}>
        <div style={{ width:'4px', height:'4px', borderRadius:'50%', background:hzC, boxShadow:`0 0 4px ${hzC}` }} />
        <R k="Hz" v={hz.toFixed(0)} c={hzC} />
      </div>
      <R k="spk" v={fmt(metrics.total_spikes)} c="var(--c-input)" />
      <R k="vto" v={String(metrics.total_vetoes)} c={metrics.total_vetoes>0?'var(--c-guard)':'var(--t-sec)'} />
      <R k="syn" v={fmt(metrics.active_synapses)} c="var(--c-process)" />
      <R k="E" v={metrics.energy_units.toFixed(1)} c="var(--c-memory)" />

      <span style={{ color:'var(--t-ghost)', fontFamily:'var(--mono)', fontSize:'10px' }}>│</span>

      <R k="t" v={(metrics.sim_time/1000).toFixed(2)+'s'} />
      <R k="tick" v={String(metrics.tick)} />

      <div style={{ flex:1 }} />

      {/* Connection state */}
      <div style={{ display:'flex', alignItems:'center', gap:'5px' }}>
        <div style={{
          width:'4px', height:'4px', borderRadius:'50%',
          background: connected ? 'var(--c-output)' : 'var(--c-predict)',
          boxShadow: connected ? '0 0 6px var(--c-output)' : undefined,
          animation: connected ? undefined : 'pulse 2s ease-in-out infinite',
        }} />
        <span style={{ fontFamily:'var(--mono)', fontSize:'8px', letterSpacing:'1.5px', color: connected?'var(--c-output)':'var(--c-predict)' }}>
          {connected ? 'LIVE' : 'SIM'}
        </span>
      </div>
    </div>
  )
}
