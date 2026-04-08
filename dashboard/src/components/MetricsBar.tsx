import type { RuntimeMetrics } from '../lib/protocol'

interface MetricsBarProps {
  metrics: RuntimeMetrics
  connected: boolean
}

function fmt(n: number): string {
  if (n >= 1e6) return (n / 1e6).toFixed(1) + 'M'
  if (n >= 1e3) return (n / 1e3).toFixed(1) + 'K'
  return n.toFixed(0)
}

function Val({ label, value, color }: { label: string; value: string; color?: string }) {
  return (
    <div style={{ display: 'flex', alignItems: 'baseline', gap: '4px' }}>
      <span style={{ fontFamily: 'var(--font-mono)', fontSize: '7px', letterSpacing: '1px', color: 'var(--text-dim)' }}>
        {label}
      </span>
      <span style={{
        fontFamily: 'var(--font-mono)', fontSize: '11px', fontWeight: 500,
        fontVariantNumeric: 'tabular-nums', color: color || 'var(--text-max)',
        letterSpacing: '0.3px',
      }}>
        {value}
      </span>
    </div>
  )
}

export default function MetricsBar({ metrics, connected }: MetricsBarProps) {
  const hz = metrics.ticks_per_second
  const hzColor = hz > 800 ? 'var(--mod-m1)' : hz > 400 ? 'var(--mod-pe)' : 'var(--mod-bst)'

  return (
    <div style={{
      display: 'flex', alignItems: 'center', gap: '12px',
      padding: '0 10px', height: '32px', flexShrink: 0,
      background: 'var(--surface-1)',
      borderBottom: '1px solid var(--border-dim)',
      position: 'relative', zIndex: 2,
    }}>
      {/* Top accent line */}
      <div style={{
        position: 'absolute', top: 0, left: 0, right: 0, height: '1px',
        background: `linear-gradient(90deg, transparent, var(--cyan-dim) 30%, ${connected ? 'var(--cyan-med)' : 'var(--border-subtle)'} 50%, var(--cyan-dim) 70%, transparent)`,
      }} />

      {/* System ID */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
        <div style={{
          width: '4px', height: '4px', borderRadius: '1px',
          background: 'var(--cyan)',
          boxShadow: '0 0 8px var(--cyan), 0 0 2px var(--cyan)',
        }} />
        <span style={{ fontFamily: 'var(--font-mono)', fontSize: '10px', fontWeight: 600, letterSpacing: '3px', color: 'var(--cyan)' }}>
          ENGRAM
        </span>
      </div>

      <div style={{ width: '1px', height: '16px', background: 'var(--border-dim)' }} />

      {/* Vital readouts */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '3px' }}>
        <div style={{ width: '4px', height: '4px', borderRadius: '50%', background: hzColor, boxShadow: `0 0 4px ${hzColor}` }} />
        <Val label="Hz" value={hz.toFixed(0)} color={hzColor} />
      </div>

      <Val label="SPK" value={fmt(metrics.total_spikes)} color="var(--mod-s1)" />
      <Val label="VTO" value={String(metrics.total_vetoes)} color={metrics.total_vetoes > 0 ? 'var(--mod-bst)' : 'var(--text-sec)'} />
      <Val label="SYN" value={fmt(metrics.active_synapses)} color="var(--mod-asc)" />
      <Val label="E" value={metrics.energy_units.toFixed(1)} color="var(--mod-hpc)" />

      <div style={{ width: '1px', height: '16px', background: 'var(--border-dim)' }} />

      <Val label="T" value={(metrics.sim_time / 1000).toFixed(2) + 's'} />
      <Val label="#" value={String(metrics.tick)} />

      <div style={{ flex: 1 }} />

      {/* Connection */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
        <div style={{
          width: '4px', height: '4px', borderRadius: '50%',
          background: connected ? 'var(--mod-m1)' : 'var(--mod-pe)',
          boxShadow: connected ? '0 0 6px var(--mod-m1)' : undefined,
          animation: connected ? undefined : 'pulse 2s ease-in-out infinite',
        }} />
        <span style={{ fontFamily: 'var(--font-mono)', fontSize: '8px', letterSpacing: '1.5px', color: connected ? 'var(--mod-m1)' : 'var(--mod-pe)' }}>
          {connected ? 'LIVE' : 'SIM'}
        </span>
      </div>
    </div>
  )
}
