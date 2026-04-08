import type { VetoEvent } from '../lib/protocol'
import { useEffect, useRef } from 'react'

interface SafetyLogProps { vetoes: VetoEvent[] }

const ACTS = ['SUP','DXT','INF','SIN']

export default function SafetyLog({ vetoes }: SafetyLogProps) {
  const ref = useRef<HTMLDivElement>(null)
  useEffect(() => { ref.current && (ref.current.scrollTop = ref.current.scrollHeight) }, [vetoes])

  if (vetoes.length === 0) {
    return (
      <div style={{
        display:'flex',alignItems:'center',justifyContent:'center',
        height:'calc(100% - 22px)',flexDirection:'column',gap:'3px',
      }}>
        <div style={{ width:'4px',height:'4px',borderRadius:'50%',background:'var(--mod-m1)',boxShadow:'0 0 6px var(--mod-m1)' }} />
        <span style={{ fontFamily:'var(--font-mono)',fontSize:'7px',color:'var(--text-dim)',letterSpacing:'2px' }}>NOMINAL</span>
      </div>
    )
  }

  return (
    <div ref={ref} style={{ overflow:'auto',height:'calc(100% - 22px)',padding:'1px 0' }}>
      <div style={{
        display:'grid',gridTemplateColumns:'40px 40px 1fr',padding:'1px 6px',
        borderBottom:'1px solid var(--border-ghost)',position:'sticky',top:0,
        background:'var(--surface-1)',zIndex:1,
      }}>
        {['T','ACT','CLASS'].map(h => (
          <span key={h} style={{ fontFamily:'var(--font-mono)',fontSize:'6px',letterSpacing:'1px',color:'var(--text-dim)' }}>{h}</span>
        ))}
      </div>
      {vetoes.map((v,i) => {
        const hard = typeof v.reason === 'string'
        const pc = hard ? 'var(--mod-bst)' : 'var(--mod-pe)'
        return (
          <div key={i} style={{
            display:'grid',gridTemplateColumns:'40px 40px 1fr',padding:'2px 6px',
            borderBottom:'1px solid var(--border-ghost)',
            background: hard ? 'rgba(224,64,64,0.04)' : 'rgba(240,144,48,0.03)',
            animation: i===vetoes.length-1 ? 'veto-flash 1.5s ease-out' : undefined,
            alignItems:'center',
          }}>
            <span style={{ fontFamily:'var(--font-mono)',fontSize:'8px',color:'var(--text-dim)',fontVariantNumeric:'tabular-nums' }}>
              {v.timestamp.toFixed(0)}
            </span>
            <div style={{ display:'flex',alignItems:'center',gap:'3px' }}>
              <div style={{ width:'2px',height:'2px',borderRadius:'50%',background:pc,boxShadow:`0 0 3px ${pc}` }} />
              <span style={{ fontFamily:'var(--font-mono)',fontSize:'8px',color:'var(--text-max)' }}>
                {ACTS[v.vetoed_action.action_id]??'?'}
              </span>
            </div>
            <span style={{ fontFamily:'var(--font-mono)',fontSize:'7px',color:pc,letterSpacing:'0.3px' }}>
              {hard
                ? (v.reason as string).replace('HardConstraint: ','HARD/')
                : `LEARNED p=${((v.reason as {LearnedInhibition:{confidence:number}}).LearnedInhibition.confidence*100).toFixed(0)}%`
              }
            </span>
          </div>
        )
      })}
    </div>
  )
}
