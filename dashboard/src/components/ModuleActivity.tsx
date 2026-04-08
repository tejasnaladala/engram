import { useState } from 'react'
import type { ModuleSnapshot } from '../lib/protocol'
import { MODULE_NAMES, MODULE_COLOR_ARRAY, MODULE_ABBREVS } from '../lib/theme'

interface ModuleActivityProps {
  modules: ModuleSnapshot[]
}

export default function ModuleActivity({ modules }: ModuleActivityProps) {
  const [hovered, setHovered] = useState<number|null>(null)

  return (
    <div style={{
      display:'flex', flexDirection:'column', gap:'1px',
      padding:'6px 8px', height:'calc(100% - 22px)', justifyContent:'center',
    }}>
      {modules.map((mod, i) => {
        const pct = Math.min(100, mod.activity_level * 100)
        const c = MODULE_COLOR_ARRAY[i]
        const active = pct > 20
        const isHov = hovered === i

        return (
          <div
            key={i}
            onMouseEnter={() => setHovered(i)}
            onMouseLeave={() => setHovered(null)}
            style={{
              display:'flex', alignItems:'center', gap:'4px', padding:'3px 2px',
              borderRadius:'2px', cursor:'default',
              background: isHov ? 'var(--s3)' : 'transparent',
              transition: 'background 0.15s ease',
            }}
          >
            {/* Pipeline stage indicator */}
            <span style={{
              fontFamily:'var(--mono)', fontSize:'7px', color:'var(--t-dim)',
              width:'14px', textAlign:'right',
            }}>
              {String(i+1).padStart(2,'0')}
            </span>

            {/* Module badge */}
            <div style={{
              fontFamily:'var(--mono)', fontSize:'7px', fontWeight:600,
              color: c, width:'22px', textAlign:'center',
              padding:'1px 0', background:`${c}0D`, borderRadius:'1px',
              border:`1px solid ${c}15`,
              transition: 'border-color 0.15s',
              borderColor: isHov ? `${c}40` : `${c}15`,
            }}>
              {MODULE_ABBREVS[i]}
            </div>

            {/* Name */}
            <span style={{
              fontFamily:'var(--sans)', fontSize:'8px', color: isHov ? 'var(--t-pri)' : 'var(--t-sec)',
              width:'68px', whiteSpace:'nowrap' as const, overflow:'hidden', textOverflow:'ellipsis',
              transition: 'color 0.15s',
            }}>
              {MODULE_NAMES[i]}
            </span>

            {/* Activity bar */}
            <div style={{
              flex:1, height:'3px', background:'var(--s4)',
              borderRadius:'1px', overflow:'hidden',
            }}>
              <div style={{
                height:'100%', width:`${pct}%`,
                background:`linear-gradient(90deg, ${c}40, ${c})`,
                borderRadius:'1px', transition:'width 100ms ease-out',
                boxShadow: active ? `0 0 8px ${c}25` : undefined,
              }} />
            </div>

            {/* Value */}
            <span style={{
              fontFamily:'var(--mono)', fontSize:'8px', fontVariantNumeric:'tabular-nums',
              color: active ? c : 'var(--t-dim)', width:'24px', textAlign:'right',
            }}>
              {pct.toFixed(0)}%
            </span>

            {/* Neuron count — visible on hover */}
            <span style={{
              fontFamily:'var(--mono)', fontSize:'7px', color:'var(--t-dim)',
              width:'28px', textAlign:'right',
              opacity: isHov ? 1 : 0.4,
              transition: 'opacity 0.15s',
            }}>
              {mod.active_count}/{mod.neuron_count}
            </span>
          </div>
        )
      })}
    </div>
  )
}
