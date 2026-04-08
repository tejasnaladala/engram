import { useRef, useEffect, useState, useCallback } from 'react'
import type { SpikeEvent } from '../lib/protocol'
import { MODULE_COLOR_ARRAY, MODULE_ABBREVS, MODULE_NEURON_COUNTS, TOTAL_NEURONS } from '../lib/theme'

interface SpikeRasterProps {
  spikeHistory: SpikeEvent[][]
}

const MOD_IDX: Record<string,number> = {
  'Sensory':0,'AssociativeMemory':1,'PredictiveError':2,
  'EpisodicMemory':3,'ActionSelector':4,'SafetyKernel':5,
}

/** Multi-electrode array -- the hero visualization.
 *  Each row = one neuron. Each column = one time frame.
 *  Color = module. Brightness = spike strength.
 *  Interactive: hover shows neuron ID and module. */
export default function SpikeRaster({ spikeHistory }: SpikeRasterProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const [hover, setHover] = useState<{x:number,y:number,mod:string,nid:number}|null>(null)

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    const canvas = canvasRef.current
    if (!canvas) return
    const rect = canvas.getBoundingClientRect()
    const dpr = window.devicePixelRatio
    const mx = (e.clientX - rect.left)
    const my = (e.clientY - rect.top)
    const lbl = 28
    if (mx < lbl) { setHover(null); return }

    const h = rect.height
    const rowH = h / TOTAL_NEURONS
    const neuronIdx = Math.floor(my / rowH)

    let modIdx = 0; let offset = 0
    for (let m = 0; m < 6; m++) {
      if (neuronIdx < offset + MODULE_NEURON_COUNTS[m]) { modIdx = m; break }
      offset += MODULE_NEURON_COUNTS[m]
    }
    setHover({ x: e.clientX, y: e.clientY, mod: MODULE_ABBREVS[modIdx], nid: neuronIdx - offset })
  }, [])

  useEffect(() => {
    const canvas = canvasRef.current
    const container = containerRef.current
    if (!canvas || !container) return
    const rect = container.getBoundingClientRect()
    const lbl = 28
    const w = rect.width, h = rect.height
    const dpr = window.devicePixelRatio
    canvas.width = w * dpr; canvas.height = h * dpr
    canvas.style.width = w+'px'; canvas.style.height = h+'px'
    const ctx = canvas.getContext('2d')!
    ctx.scale(dpr, dpr)

    // Void background
    ctx.fillStyle = '#030408'
    ctx.fillRect(0,0,w,h)

    const plotW = w - lbl
    const colW = Math.max(0.5, plotW / Math.max(spikeHistory.length, 1))
    const rowH = h / TOTAL_NEURONS

    // Module bands + labels
    let offset = 0
    for (let m = 0; m < 6; m++) {
      const y = offset * rowH
      const bH = MODULE_NEURON_COUNTS[m] * rowH
      const hex = MODULE_COLOR_ARRAY[m]
      const r=parseInt(hex.slice(1,3),16), g=parseInt(hex.slice(3,5),16), b=parseInt(hex.slice(5,7),16)

      // Subtle band tint
      ctx.fillStyle = `rgba(${r},${g},${b},0.010)`
      ctx.fillRect(lbl, y, plotW, bH)

      // Separator
      if (m > 0) {
        ctx.strokeStyle = 'rgba(80,120,200,0.05)'
        ctx.lineWidth = 0.5
        ctx.beginPath(); ctx.moveTo(lbl,y); ctx.lineTo(w,y); ctx.stroke()
      }

      // Label
      ctx.save()
      ctx.fillStyle = MODULE_COLOR_ARRAY[m]
      ctx.globalAlpha = 0.35
      ctx.font = '600 7px "JetBrains Mono", monospace'
      ctx.textBaseline = 'middle'; ctx.textAlign = 'center'
      ctx.fillText(MODULE_ABBREVS[m], lbl/2, y + bH/2)
      ctx.restore()

      offset += MODULE_NEURON_COUNTS[m]
    }

    // Render spikes
    for (let col = 0; col < spikeHistory.length; col++) {
      for (const spike of spikeHistory[col]) {
        const mi = MOD_IDX[spike.source_module] ?? 0
        const mOff = MODULE_NEURON_COUNTS.slice(0,mi).reduce((a,b)=>a+b,0)
        const row = mOff + (spike.neuron_id % MODULE_NEURON_COUNTS[mi])
        const x = lbl + col * colW
        const y = row * rowH
        const hex = MODULE_COLOR_ARRAY[mi]
        const r=parseInt(hex.slice(1,3),16), g=parseInt(hex.slice(3,5),16), b=parseInt(hex.slice(5,7),16)
        const a = 0.40 + spike.strength * 0.60

        const dw = Math.max(colW, 0.8), dh = Math.max(rowH, 0.8)
        ctx.fillStyle = `rgba(${r},${g},${b},${a})`
        ctx.fillRect(x, y, dw, dh)

        // Phosphor bloom
        if (spike.strength > 0.78) {
          ctx.fillStyle = `rgba(${Math.min(255,r+50)},${Math.min(255,g+50)},${Math.min(255,b+50)},0.10)`
          ctx.fillRect(x-0.5, y-0.5, dw+1, dh+1)
        }
      }
    }

    // Left separator
    ctx.fillStyle = 'rgba(80,120,200,0.06)'
    ctx.fillRect(lbl-0.5, 0, 0.5, h)

    // Time markers
    const interval = Math.max(1, Math.floor(spikeHistory.length/6))
    for (let i = interval; i < spikeHistory.length; i += interval) {
      ctx.strokeStyle = 'rgba(80,120,200,0.025)'
      ctx.lineWidth = 0.5
      ctx.beginPath(); ctx.moveTo(lbl+i*colW,0); ctx.lineTo(lbl+i*colW,h); ctx.stroke()
    }

    // "Now" indicator -- bright line at the right edge
    if (spikeHistory.length > 10) {
      const nowX = lbl + spikeHistory.length * colW
      ctx.strokeStyle = 'rgba(56,216,232,0.15)'
      ctx.lineWidth = 1
      ctx.beginPath(); ctx.moveTo(nowX, 0); ctx.lineTo(nowX, h); ctx.stroke()
    }

  }, [spikeHistory])

  return (
    <div ref={containerRef} style={{ width:'100%', height:'calc(100% - 22px)', position:'relative', cursor:'crosshair' }}
      onMouseMove={handleMouseMove} onMouseLeave={() => setHover(null)}>
      <canvas ref={canvasRef} style={{ display:'block' }} />

      {/* Hover tooltip */}
      {hover && (
        <div className="tooltip" style={{
          left: Math.min(hover.x + 12, (containerRef.current?.getBoundingClientRect().right || 0) - 100),
          top: hover.y - 28,
          position: 'fixed',
        }}>
          <span style={{ color: MODULE_COLOR_ARRAY[Object.values(MODULE_ABBREVS).indexOf(hover.mod)] }}>
            {hover.mod}
          </span>
          <span style={{ color:'var(--t-sec)', margin:'0 4px' }}>:</span>
          <span>N{hover.nid}</span>
        </div>
      )}
    </div>
  )
}
