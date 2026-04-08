import type { ModuleSnapshot } from '../lib/protocol'
import { MODULE_NAMES, MODULE_COLOR_ARRAY, MODULE_ABBREVS } from '../lib/theme'

interface ModuleActivityProps {
  modules: ModuleSnapshot[]
}

export default function ModuleActivity({ modules }: ModuleActivityProps) {
  return (
    <div style={{
      display: 'flex', flexDirection: 'column', gap: '1px',
      padding: '6px 8px', height: 'calc(100% - 22px)', justifyContent: 'center',
    }}>
      {modules.map((mod, i) => {
        const pct = Math.min(100, mod.activity_level * 100)
        const c = MODULE_COLOR_ARRAY[i]
        const active = pct > 25

        return (
          <div key={i} style={{ display: 'flex', alignItems: 'center', gap: '4px', padding: '2px 0' }}>
            {/* Badge */}
            <div style={{
              fontFamily: 'var(--font-mono)', fontSize: '7px', fontWeight: 600,
              letterSpacing: '0.3px', color: c, width: '22px', textAlign: 'center',
              padding: '1px 0', background: `${c}10`, borderRadius: '1px',
              border: `1px solid ${c}18`,
            }}>
              {MODULE_ABBREVS[i]}
            </div>

            {/* Name */}
            <div style={{
              fontFamily: 'var(--font-sans)', fontSize: '8px', color: 'var(--text-sec)',
              width: '72px', whiteSpace: 'nowrap' as const, overflow: 'hidden', textOverflow: 'ellipsis',
            }}>
              {MODULE_NAMES[i]}
            </div>

            {/* Bar */}
            <div style={{
              flex: 1, height: '3px', background: 'var(--surface-4)',
              borderRadius: '1px', overflow: 'hidden',
            }}>
              <div style={{
                height: '100%', width: `${pct}%`,
                background: `linear-gradient(90deg, ${c}50, ${c})`,
                borderRadius: '1px',
                transition: 'width 100ms ease-out',
                boxShadow: active ? `0 0 6px ${c}30` : undefined,
              }} />
            </div>

            {/* Value */}
            <span style={{
              fontFamily: 'var(--font-mono)', fontSize: '8px',
              fontVariantNumeric: 'tabular-nums',
              color: active ? c : 'var(--text-dim)',
              width: '26px', textAlign: 'right',
            }}>
              {pct.toFixed(0)}%
            </span>

            {/* Active/total neuron count */}
            <span style={{
              fontFamily: 'var(--font-mono)', fontSize: '7px',
              color: 'var(--text-dim)', width: '28px', textAlign: 'right',
            }}>
              {mod.active_count}/{mod.neuron_count}
            </span>
          </div>
        )
      })}
    </div>
  )
}
