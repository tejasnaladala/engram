import type { RuntimeMetrics } from '../lib/protocol'

interface MetricsBarProps {
  metrics: RuntimeMetrics
  connected: boolean
}

function formatNumber(n: number): string {
  if (n >= 1_000_000) return (n / 1_000_000).toFixed(1) + 'M'
  if (n >= 1_000) return (n / 1_000).toFixed(1) + 'K'
  return n.toFixed(0)
}

function formatBytes(b: number): string {
  if (b >= 1024 * 1024) return (b / (1024 * 1024)).toFixed(1) + ' MB'
  if (b >= 1024) return (b / 1024).toFixed(1) + ' KB'
  return b + ' B'
}

/** Clinical diagnostic readout — single metric */
function Readout({ label, value, unit, color }: {
  label: string
  value: string
  unit?: string
  color?: string
}) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '1px' }}>
      <span style={{
        fontFamily: 'var(--font-clinical)',
        fontSize: '8px',
        letterSpacing: '1.2px',
        textTransform: 'uppercase' as const,
        color: 'var(--text-tertiary)',
      }}>
        {label}
      </span>
      <span style={{
        fontFamily: 'var(--font-clinical)',
        fontSize: '12px',
        fontVariantNumeric: 'tabular-nums',
        color: color || 'var(--text-value)',
        letterSpacing: '0.3px',
      }}>
        {value}
        {unit && <span style={{ fontSize: '9px', color: 'var(--text-tertiary)', marginLeft: '2px' }}>{unit}</span>}
      </span>
    </div>
  )
}

/** Vertical divider — scan panel separator */
function Divider() {
  return (
    <div style={{
      width: '1px',
      height: '28px',
      background: 'linear-gradient(180deg, transparent, var(--border-default), transparent)',
    }} />
  )
}

export default function MetricsBar({ metrics, connected }: MetricsBarProps) {
  const perfColor = metrics.ticks_per_second > 800
    ? 'var(--status-active)'
    : metrics.ticks_per_second > 400
      ? 'var(--status-warning)'
      : 'var(--status-critical)'

  return (
    <div style={{
      display: 'flex',
      alignItems: 'center',
      gap: '16px',
      padding: '4px 16px',
      background: 'var(--surface-2)',
      borderBottom: '1px solid var(--border-subtle)',
      height: '44px',
      flexShrink: 0,
      position: 'relative',
    }}>
      {/* Top highlight line — scan panel convention */}
      <div style={{
        position: 'absolute',
        top: 0,
        left: 0,
        right: 0,
        height: '1px',
        background: 'linear-gradient(90deg, transparent, var(--border-default) 30%, var(--accent-primary-dim) 50%, var(--border-default) 70%, transparent)',
      }} />

      {/* System identifier — clinical device ID pattern */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
        <div style={{
          width: '6px',
          height: '6px',
          borderRadius: '1px',
          background: 'var(--accent-primary)',
          boxShadow: '0 0 8px var(--accent-primary)',
        }} />
        <span style={{
          fontFamily: 'var(--font-clinical)',
          fontSize: '11px',
          fontWeight: 600,
          letterSpacing: '3px',
          color: 'var(--accent-primary)',
        }}>
          ENGRAM
        </span>
        <span style={{
          fontFamily: 'var(--font-clinical)',
          fontSize: '8px',
          letterSpacing: '1px',
          color: 'var(--text-tertiary)',
          paddingLeft: '4px',
        }}>
          NCS-1.0
        </span>
      </div>

      <Divider />

      {/* Performance indicator — like a vital sign */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
        <div className="status-dot" style={{
          background: perfColor,
          boxShadow: `0 0 6px ${perfColor}`,
          width: '5px',
          height: '5px',
          borderRadius: '50%',
        }} />
        <Readout label="Hz" value={metrics.ticks_per_second.toFixed(0)} color={perfColor} />
      </div>

      <Divider />

      <Readout label="Spikes" value={formatNumber(metrics.total_spikes)} color="var(--mod-sensory)" />
      <Readout label="Vetoes" value={String(metrics.total_vetoes)} color={metrics.total_vetoes > 0 ? 'var(--mod-safety)' : 'var(--text-value)'} />
      <Readout label="Synapses" value={formatNumber(metrics.active_synapses)} color="var(--mod-association)" />
      <Readout label="Energy" value={metrics.energy_units.toFixed(1)} unit="J" color="var(--mod-episodic)" />

      <Divider />

      <Readout label="Elapsed" value={(metrics.sim_time / 1000).toFixed(2)} unit="s" />
      <Readout label="Tick" value={String(metrics.tick)} />

      {/* Spacer */}
      <div style={{ flex: 1 }} />

      {/* Connection status — clinical monitoring convention */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
        <div style={{
          width: '5px',
          height: '5px',
          borderRadius: '50%',
          background: connected ? 'var(--status-active)' : 'var(--status-warning)',
          boxShadow: connected ? '0 0 6px var(--status-active)' : undefined,
          animation: connected ? undefined : 'pulse-glow 2s ease-in-out infinite',
        }} />
        <span style={{
          fontFamily: 'var(--font-clinical)',
          fontSize: '9px',
          letterSpacing: '1.5px',
          color: connected ? 'var(--status-active)' : 'var(--status-warning)',
        }}>
          {connected ? 'LIVE FEED' : 'SIMULATION'}
        </span>
      </div>
    </div>
  )
}
