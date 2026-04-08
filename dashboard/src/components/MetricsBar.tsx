import type { RuntimeMetrics } from '../lib/protocol'

interface Props {
  metrics: RuntimeMetrics
  connected: boolean
}

function formatNumber(n: number): string {
  if (n >= 1_000_000) return (n / 1_000_000).toFixed(1) + 'M'
  if (n >= 1_000) return (n / 1_000).toFixed(1) + 'K'
  return n.toFixed(0)
}

export default function MetricsBar({ metrics, connected }: Props) {
  const perfColor = metrics.ticks_per_second > 800 ? '#22c55e' : metrics.ticks_per_second > 400 ? '#eab308' : '#ef4444'

  return (
    <div
      className="flex items-center gap-6 px-4 py-2 text-xs font-mono"
      style={{
        background: '#12121a',
        borderBottom: '1px solid rgba(255,255,255,0.05)',
      }}
    >
      {/* Logo */}
      <div className="flex items-center gap-2">
        <span className="text-lg">🧠</span>
        <span className="text-sm font-bold tracking-wider" style={{ color: '#00ffaa' }}>ENGRAM</span>
      </div>

      <div className="w-px h-4" style={{ background: 'rgba(255,255,255,0.1)' }} />

      {/* Ticks/sec */}
      <div className="flex items-center gap-1.5">
        <div className="w-2 h-2 rounded-full" style={{ background: perfColor }} />
        <span style={{ color: '#8888aa' }}>TICKS/S</span>
        <span style={{ color: '#e0e0ff' }}>{metrics.ticks_per_second.toFixed(0)}</span>
      </div>

      {/* Total spikes */}
      <div>
        <span style={{ color: '#8888aa' }}>SPIKES </span>
        <span style={{ color: '#00d4ff' }}>{formatNumber(metrics.total_spikes)}</span>
      </div>

      {/* Vetoes */}
      <div>
        <span style={{ color: '#8888aa' }}>VETOES </span>
        <span style={{ color: metrics.total_vetoes > 0 ? '#ef4444' : '#e0e0ff' }}>
          {metrics.total_vetoes}
        </span>
      </div>

      {/* Synapses */}
      <div>
        <span style={{ color: '#8888aa' }}>SYNAPSES </span>
        <span style={{ color: '#a855f7' }}>{formatNumber(metrics.active_synapses)}</span>
      </div>

      {/* Energy */}
      <div>
        <span style={{ color: '#8888aa' }}>ENERGY </span>
        <span style={{ color: '#22d3ee' }}>{metrics.energy_units.toFixed(1)}</span>
      </div>

      {/* Sim time */}
      <div>
        <span style={{ color: '#8888aa' }}>T= </span>
        <span style={{ color: '#e0e0ff' }}>{(metrics.sim_time / 1000).toFixed(1)}s</span>
      </div>

      {/* Spacer */}
      <div className="flex-1" />

      {/* Connection status */}
      <div className="flex items-center gap-1.5">
        <div
          className="w-2 h-2 rounded-full"
          style={{
            background: connected ? '#22c55e' : '#ff8c00',
            animation: connected ? undefined : 'pulse-glow 2s ease-in-out infinite',
          }}
        />
        <span style={{ color: '#8888aa' }}>
          {connected ? 'LIVE' : 'DEMO'}
        </span>
      </div>
    </div>
  )
}
