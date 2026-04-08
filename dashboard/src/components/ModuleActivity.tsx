import type { ModuleSnapshot } from '../lib/protocol'
import { MODULE_NAMES, MODULE_COLOR_ARRAY, MODULE_ABBREVS, MODULE_COLOR_DIM } from '../lib/theme'

interface ModuleActivityProps {
  modules: ModuleSnapshot[]
}

export default function ModuleActivity({ modules }: ModuleActivityProps) {
  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      gap: '3px',
      padding: '8px 10px',
      height: '100%',
      justifyContent: 'center',
    }}>
      {modules.map((mod, i) => {
        const pct = Math.min(100, mod.activity_level * 100)
        const color = MODULE_COLOR_ARRAY[i]
        const isActive = pct > 30

        return (
          <div key={i} style={{
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
            padding: '3px 0',
          }}>
            {/* Module abbreviation badge */}
            <div style={{
              fontFamily: 'var(--font-clinical)',
              fontSize: '8px',
              fontWeight: 600,
              letterSpacing: '0.5px',
              color: color,
              width: '26px',
              textAlign: 'center',
              padding: '2px 0',
              background: MODULE_COLOR_DIM[i],
              borderRadius: '2px',
              border: `1px solid ${color}22`,
            }}>
              {MODULE_ABBREVS[i]}
            </div>

            {/* Region name */}
            <div style={{
              fontFamily: 'var(--font-label)',
              fontSize: '9px',
              color: 'var(--text-secondary)',
              width: '90px',
              whiteSpace: 'nowrap' as const,
              overflow: 'hidden',
              textOverflow: 'ellipsis',
            }}>
              {MODULE_NAMES[i]}
            </div>

            {/* Activity bar — electrode amplitude indicator style */}
            <div style={{
              flex: 1,
              height: '4px',
              background: 'var(--surface-5)',
              borderRadius: '1px',
              overflow: 'hidden',
              position: 'relative',
            }}>
              <div style={{
                height: '100%',
                width: `${pct}%`,
                background: `linear-gradient(90deg, ${color}60, ${color})`,
                borderRadius: '1px',
                transition: 'width 120ms ease-out',
                boxShadow: isActive ? `0 0 6px ${color}40` : undefined,
              }} />
            </div>

            {/* Numeric readout */}
            <div style={{
              fontFamily: 'var(--font-clinical)',
              fontSize: '9px',
              fontVariantNumeric: 'tabular-nums',
              color: isActive ? color : 'var(--text-tertiary)',
              width: '32px',
              textAlign: 'right',
            }}>
              {pct.toFixed(0)}%
            </div>

            {/* Neuron count */}
            <div style={{
              fontFamily: 'var(--font-clinical)',
              fontSize: '8px',
              color: 'var(--text-tertiary)',
              width: '24px',
              textAlign: 'right',
            }}>
              {mod.active_count}/{mod.neuron_count}
            </div>
          </div>
        )
      })}
    </div>
  )
}
