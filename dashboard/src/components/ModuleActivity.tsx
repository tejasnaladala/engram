import type { ModuleSnapshot } from '../lib/protocol'
import { MODULE_NAMES, MODULE_COLOR_ARRAY } from '../lib/theme'

interface Props {
  modules: ModuleSnapshot[]
}

export default function ModuleActivity({ modules }: Props) {
  return (
    <div className="p-3 flex flex-col gap-2 h-full justify-center">
      {modules.map((mod, i) => (
        <div key={i} className="flex items-center gap-2">
          <div className="w-20 text-[10px] truncate" style={{ color: MODULE_COLOR_ARRAY[i] }}>
            {MODULE_NAMES[i]}
          </div>
          <div className="flex-1 h-3 rounded-full overflow-hidden" style={{ background: '#1a1a2e' }}>
            <div
              className="h-full rounded-full transition-all duration-150"
              style={{
                width: `${Math.min(100, mod.activity_level * 100)}%`,
                background: `linear-gradient(90deg, ${MODULE_COLOR_ARRAY[i]}88, ${MODULE_COLOR_ARRAY[i]})`,
                boxShadow: mod.activity_level > 0.5
                  ? `0 0 8px ${MODULE_COLOR_ARRAY[i]}44`
                  : undefined,
              }}
            />
          </div>
          <div className="w-8 text-[10px] text-right" style={{ color: '#8888aa' }}>
            {(mod.activity_level * 100).toFixed(0)}%
          </div>
        </div>
      ))}
    </div>
  )
}
