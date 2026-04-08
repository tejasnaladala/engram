import { useRef, useEffect, useState } from 'react'
import type { MemoryFormation } from '../lib/protocol'

interface MemoryHeatMapProps { formations: MemoryFormation[]; tick: number }
const GS = 20

export default function MemoryHeatMap({ formations, tick }: MemoryHeatMapProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [grid, setGrid] = useState(new Float32Array(GS*GS))
  const [hoverCell, setHoverCell] = useState<{x:number,y:number,val:number}|null>(null)

  useEffect(() => {
    setGrid(prev => {
      const n = new Float32Array(prev)
      for (let i=0;i<n.length;i++) n[i]*=0.991
      for (const f of formations) {
        const idx = f.location_id%(GS*GS)
        n[idx] = Math.min(1.0, n[idx]+f.strength*0.6)
        const x=idx%GS, y=Math.floor(idx/GS)
        for (const [nx,ny] of [[x-1,y],[x+1,y],[x,y-1],[x,y+1]])
          if(nx>=0&&nx<GS&&ny>=0&&ny<GS) n[ny*GS+nx]=Math.min(1.0,n[ny*GS+nx]+f.strength*0.12)
      }
      return n
    })
  }, [formations, tick])

  useEffect(() => {
    const canvas = canvasRef.current
    if(!canvas?.parentElement) return
    const rect = canvas.parentElement.getBoundingClientRect()
    const size = Math.min(rect.width-6, rect.height-6)
    const dpr = window.devicePixelRatio
    canvas.width=size*dpr; canvas.height=size*dpr
    canvas.style.width=size+'px'; canvas.style.height=size+'px'
    const ctx = canvas.getContext('2d')!
    ctx.scale(dpr,dpr)
    ctx.fillStyle='#030408'; ctx.fillRect(0,0,size,size)
    const cw=size/GS, ch=size/GS

    for (let y=0;y<GS;y++) for (let x=0;x<GS;x++) {
      const v=grid[y*GS+x]; if(v<0.004)continue
      let r:number,g:number,b:number
      if(v<0.2){r=Math.floor(v/0.2*80);g=0;b=0}
      else if(v<0.4){const t=(v-0.2)/0.2;r=80+Math.floor(t*60);g=0;b=0}
      else if(v<0.6){const t=(v-0.4)/0.2;r=140+Math.floor(t*72);g=Math.floor(t*64);b=0}
      else if(v<0.8){const t=(v-0.6)/0.2;r=212+Math.floor(t*43);g=64+Math.floor(t*144);b=0}
      else{const t=(v-0.8)/0.2;r=255;g=208+Math.floor(t*47);b=Math.floor(t*255)}
      ctx.fillStyle=`rgb(${r},${g},${b})`
      ctx.fillRect(x*cw+0.3,y*ch+0.3,cw-0.6,ch-0.6)
    }

    // Grid
    ctx.strokeStyle='rgba(80,120,200,0.025)'; ctx.lineWidth=0.3
    for(let i=0;i<=GS;i++){ctx.beginPath();ctx.moveTo(i*cw,0);ctx.lineTo(i*cw,size);ctx.stroke();ctx.beginPath();ctx.moveTo(0,i*ch);ctx.lineTo(size,i*ch);ctx.stroke()}

    // Crosshair
    const cx=size/2,cy=size/2
    ctx.strokeStyle='rgba(56,216,232,0.06)'; ctx.lineWidth=0.5; ctx.setLineDash([2,3])
    ctx.beginPath();ctx.moveTo(cx,0);ctx.lineTo(cx,size);ctx.stroke()
    ctx.beginPath();ctx.moveTo(0,cy);ctx.lineTo(size,cy);ctx.stroke()
    ctx.setLineDash([])

    // Corner marks
    const ml=8; ctx.strokeStyle='rgba(80,120,200,0.10)'; ctx.lineWidth=0.8
    ctx.beginPath();ctx.moveTo(2,2+ml);ctx.lineTo(2,2);ctx.lineTo(2+ml,2);ctx.stroke()
    ctx.beginPath();ctx.moveTo(size-2-ml,2);ctx.lineTo(size-2,2);ctx.lineTo(size-2,2+ml);ctx.stroke()
    ctx.beginPath();ctx.moveTo(2,size-2-ml);ctx.lineTo(2,size-2);ctx.lineTo(2+ml,size-2);ctx.stroke()
    ctx.beginPath();ctx.moveTo(size-2-ml,size-2);ctx.lineTo(size-2,size-2);ctx.lineTo(size-2,size-2-ml);ctx.stroke()

    // Highlight hovered cell
    if (hoverCell) {
      ctx.strokeStyle = 'rgba(56,216,232,0.4)'
      ctx.lineWidth = 1
      ctx.strokeRect(hoverCell.x*cw, hoverCell.y*ch, cw, ch)
    }
  }, [grid, hoverCell])

  const handleMouse = (e: React.MouseEvent) => {
    const canvas = canvasRef.current
    if(!canvas) return
    const rect = canvas.getBoundingClientRect()
    const x = Math.floor((e.clientX-rect.left)/rect.width*GS)
    const y = Math.floor((e.clientY-rect.top)/rect.height*GS)
    if(x>=0&&x<GS&&y>=0&&y<GS) setHoverCell({x,y,val:grid[y*GS+x]})
    else setHoverCell(null)
  }

  return (
    <div style={{display:'flex',alignItems:'center',justifyContent:'center',height:'calc(100% - 22px)',padding:'3px',position:'relative'}}
      onMouseMove={handleMouse} onMouseLeave={()=>setHoverCell(null)}>
      <canvas ref={canvasRef} style={{borderRadius:'1px',cursor:'crosshair'}} />
      {hoverCell && hoverCell.val > 0.01 && (
        <div className="tooltip" style={{ position:'absolute', bottom:'6px', right:'6px' }}>
          [{hoverCell.x},{hoverCell.y}] {(hoverCell.val*100).toFixed(0)}%
        </div>
      )}
    </div>
  )
}
