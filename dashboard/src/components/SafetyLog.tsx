import type { VetoEvent } from '../lib/protocol'
import { useEffect, useRef, useState } from 'react'

interface SafetyLogProps { vetoes: VetoEvent[] }
const ACTS = ['SUP','DXT','INF','SIN']

export default function SafetyLog({ vetoes }: SafetyLogProps) {
  const ref = useRef<HTMLDivElement>(null)
  const [expanded, setExpanded] = useState<number|null>(null)
  useEffect(()=>{ref.current&&(ref.current.scrollTop=ref.current.scrollHeight)},[vetoes])

  if (vetoes.length === 0) return (
    <div style={{display:'flex',alignItems:'center',justifyContent:'center',height:'calc(100% - 22px)',gap:'6px'}}>
      <div style={{width:'4px',height:'4px',borderRadius:'50%',background:'var(--c-output)',boxShadow:'0 0 6px var(--c-output)'}} />
      <span style={{fontFamily:'var(--mono)',fontSize:'8px',color:'var(--t-dim)',letterSpacing:'2px'}}>ALL SYSTEMS NOMINAL</span>
    </div>
  )

  return (
    <div ref={ref} style={{overflow:'auto',height:'calc(100% - 22px)',padding:'1px 0'}}>
      {vetoes.map((v,i)=>{
        const hard=typeof v.reason==='string'
        const pc=hard?'var(--c-guard)':'var(--c-predict)'
        const isExp = expanded===i
        return (
          <div key={i}
            onClick={()=>setExpanded(isExp?null:i)}
            style={{
              padding:'3px 8px', cursor:'pointer',
              borderBottom:'1px solid var(--b-ghost)',
              background: hard?'rgba(224,72,72,0.03)':'rgba(240,144,48,0.02)',
              animation: i===vetoes.length-1?'veto-flash 1.5s ease-out':undefined,
              transition:'background 0.15s',
            }}
            onMouseEnter={(e)=>(e.currentTarget.style.background=hard?'rgba(224,72,72,0.06)':'rgba(240,144,48,0.04)')}
            onMouseLeave={(e)=>(e.currentTarget.style.background=hard?'rgba(224,72,72,0.03)':'rgba(240,144,48,0.02)')}
          >
            <div style={{display:'flex',alignItems:'center',gap:'6px'}}>
              <div style={{width:'3px',height:'3px',borderRadius:'50%',background:pc,boxShadow:`0 0 3px ${pc}`}} />
              <span style={{fontFamily:'var(--mono)',fontSize:'8px',color:'var(--t-dim)',fontVariantNumeric:'tabular-nums',width:'36px'}}>
                {v.timestamp.toFixed(0)}ms
              </span>
              <span style={{fontFamily:'var(--mono)',fontSize:'8px',color:'var(--t-max)',width:'24px'}}>
                {ACTS[v.vetoed_action.action_id]??'?'}
              </span>
              <span style={{fontFamily:'var(--mono)',fontSize:'7px',color:pc,flex:1}}>
                {hard?(v.reason as string).replace('HardConstraint: ',''):`LEARNED p=${((v.reason as {LearnedInhibition:{confidence:number}}).LearnedInhibition.confidence*100).toFixed(0)}%`}
              </span>
            </div>
            {/* Expanded detail on click */}
            {isExp && (
              <div style={{
                marginTop:'3px', padding:'3px 0 0 15px',
                borderTop:'1px solid var(--b-ghost)',
                fontFamily:'var(--mono)', fontSize:'7px', color:'var(--t-sec)',
                animation:'fade-up 0.15s ease-out',
              }}>
                <div>confidence: {(v.vetoed_action.confidence*100).toFixed(1)}%</div>
                <div>reflex: {v.vetoed_action.is_reflex?'yes':'no'}</div>
                <div>type: {hard?'HARD CONSTRAINT':'LEARNED INHIBITION'}</div>
              </div>
            )}
          </div>
        )
      })}
    </div>
  )
}
