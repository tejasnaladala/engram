import type { VetoEvent } from '../lib/protocol'
import { useEffect, useRef } from 'react'

interface SafetyLogProps {
  vetoes: VetoEvent[]
}

const ACTION_NAMES = ['SUP', 'DXT', 'INF', 'SIN'] // Superior, Dexter, Inferior, Sinister (anatomical directions)

/** Safety veto event log — styled as a clinical alarm/event monitor.
 *  Follows IEC 62366 medical alarm display conventions:
 *  Red = high priority (hard constraint), Amber = medium priority (learned). */
export default function SafetyLog({ vetoes }: SafetyLogProps) {
  const scrollRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [vetoes])

  if (vetoes.length === 0) {
    return (
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        height: 'calc(100% - 24px)',
        flexDirection: 'column',
        gap: '4px',
      }}>
        <div style={{
          width: '6px',
          height: '6px',
          borderRadius: '50%',
          background: 'var(--status-active)',
          boxShadow: '0 0 8px var(--status-active)',
        }} />
        <span style={{
          fontFamily: 'var(--font-clinical)',
          fontSize: '9px',
          color: 'var(--text-tertiary)',
          letterSpacing: '1.5px',
        }}>
          ALL CLEAR
        </span>
      </div>
    )
  }

  return (
    <div
      ref={scrollRef}
      style={{
        overflow: 'auto',
        height: 'calc(100% - 24px)',
        padding: '2px 0',
      }}
    >
      {/* Table header */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: '52px 55px 1fr',
        padding: '2px 8px',
        borderBottom: '1px solid var(--border-dim)',
        position: 'sticky',
        top: 0,
        background: 'var(--surface-2)',
        zIndex: 1,
      }}>
        {['T(ms)', 'ACTION', 'CLASSIFICATION'].map(h => (
          <span key={h} style={{
            fontFamily: 'var(--font-clinical)',
            fontSize: '7px',
            letterSpacing: '1px',
            color: 'var(--text-tertiary)',
          }}>
            {h}
          </span>
        ))}
      </div>

      {/* Event rows */}
      {vetoes.map((v, i) => {
        const isHard = typeof v.reason === 'string'
        const isLatest = i === vetoes.length - 1
        const priorityColor = isHard ? 'var(--mod-safety)' : 'var(--status-warning)'

        return (
          <div
            key={i}
            style={{
              display: 'grid',
              gridTemplateColumns: '52px 55px 1fr',
              padding: '3px 8px',
              borderBottom: '1px solid var(--border-dim)',
              background: isHard
                ? 'rgba(217, 64, 64, 0.06)'
                : 'rgba(232, 148, 58, 0.04)',
              animation: isLatest ? 'veto-flash 1.5s ease-out' : undefined,
              alignItems: 'center',
            }}
          >
            {/* Timestamp */}
            <span style={{
              fontFamily: 'var(--font-clinical)',
              fontSize: '9px',
              color: 'var(--text-tertiary)',
              fontVariantNumeric: 'tabular-nums',
            }}>
              {v.timestamp.toFixed(0)}
            </span>

            {/* Action with priority indicator */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
              <div style={{
                width: '3px',
                height: '3px',
                borderRadius: '50%',
                background: priorityColor,
                boxShadow: `0 0 4px ${priorityColor}`,
              }} />
              <span style={{
                fontFamily: 'var(--font-clinical)',
                fontSize: '9px',
                color: 'var(--text-value)',
              }}>
                {ACTION_NAMES[v.vetoed_action.action_id] ?? `A${v.vetoed_action.action_id}`}
              </span>
            </div>

            {/* Reason */}
            <span style={{
              fontFamily: 'var(--font-clinical)',
              fontSize: '8px',
              color: priorityColor,
              letterSpacing: '0.3px',
            }}>
              {isHard
                ? (v.reason as string).replace('HardConstraint: ', 'HARD: ')
                : `LEARNED p=${((v.reason as { LearnedInhibition: { confidence: number } }).LearnedInhibition.confidence * 100).toFixed(0)}%`
              }
            </span>
          </div>
        )
      })}
    </div>
  )
}
