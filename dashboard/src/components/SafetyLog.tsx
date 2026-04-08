import type { VetoEvent } from '../lib/protocol'
import { useEffect, useRef } from 'react'

interface Props {
  vetoes: VetoEvent[]
}

const ACTION_NAMES = ['UP', 'RIGHT', 'DOWN', 'LEFT']

export default function SafetyLog({ vetoes }: Props) {
  const scrollRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [vetoes])

  if (vetoes.length === 0) {
    return (
      <div className="flex items-center justify-center h-full text-xs" style={{ color: '#8888aa' }}>
        <div className="text-center">
          <div className="text-lg mb-1">🛡️</div>
          <div>No vetoes — all actions safe</div>
        </div>
      </div>
    )
  }

  return (
    <div ref={scrollRef} className="overflow-y-auto text-[10px] font-mono" style={{ height: 'calc(100% - 28px)' }}>
      <table className="w-full">
        <thead>
          <tr style={{ color: '#8888aa' }}>
            <th className="text-left px-2 py-1">Time</th>
            <th className="text-left px-2 py-1">Action</th>
            <th className="text-left px-2 py-1">Reason</th>
          </tr>
        </thead>
        <tbody>
          {vetoes.map((v, i) => {
            const isHard = typeof v.reason === 'string'
            return (
              <tr
                key={i}
                style={{
                  background: isHard ? 'rgba(239,68,68,0.08)' : 'rgba(234,179,8,0.06)',
                  animation: i === vetoes.length - 1 ? 'veto-flash 1s ease-out' : undefined,
                }}
              >
                <td className="px-2 py-0.5" style={{ color: '#8888aa' }}>
                  {v.timestamp.toFixed(0)}
                </td>
                <td className="px-2 py-0.5" style={{ color: '#e0e0ff' }}>
                  {ACTION_NAMES[v.vetoed_action.action_id] ?? `A${v.vetoed_action.action_id}`}
                  <span style={{ color: '#8888aa' }}>
                    {' '}({(v.vetoed_action.confidence * 100).toFixed(0)}%)
                  </span>
                </td>
                <td className="px-2 py-0.5" style={{ color: isHard ? '#ef4444' : '#eab308' }}>
                  {isHard ? (v.reason as string).replace('HardConstraint: ', '') : `Learned (${((v.reason as any).LearnedInhibition.confidence * 100).toFixed(0)}%)`}
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}
