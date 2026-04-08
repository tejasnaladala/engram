import { useRef, useEffect } from 'react'
import type { SpikeEvent } from '../lib/protocol'
import { MODULE_COLOR_ARRAY, MODULE_ABBREVS, MODULE_NEURON_COUNTS, TOTAL_NEURONS } from '../lib/theme'

interface SpikeRasterProps {
  spikeHistory: SpikeEvent[][]
}

const MOD_IDX: Record<string, number> = {
  'Sensory': 0, 'AssociativeMemory': 1, 'PredictiveError': 2,
  'EpisodicMemory': 3, 'ActionSelector': 4, 'SafetyKernel': 5,
}

export default function SpikeRaster({ spikeHistory }: SpikeRasterProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    const container = containerRef.current
    if (!canvas || !container) return

    const rect = container.getBoundingClientRect()
    const lbl = 28
    const w = rect.width
    const h = rect.height
    canvas.width = w * window.devicePixelRatio
    canvas.height = h * window.devicePixelRatio
    canvas.style.width = w + 'px'
    canvas.style.height = h + 'px'

    const ctx = canvas.getContext('2d')
    if (!ctx) return
    ctx.scale(window.devicePixelRatio, window.devicePixelRatio)

    // Deep void background
    ctx.fillStyle = '#030408'
    ctx.fillRect(0, 0, w, h)

    const plotW = w - lbl
    const colW = Math.max(0.6, plotW / Math.max(spikeHistory.length, 1))
    const rowH = h / TOTAL_NEURONS

    // Module bands and labels
    let offset = 0
    for (let m = 0; m < 6; m++) {
      const y = offset * rowH
      const bandH = MODULE_NEURON_COUNTS[m] * rowH

      // Very subtle band tint
      const hex = MODULE_COLOR_ARRAY[m]
      const r = parseInt(hex.slice(1, 3), 16)
      const g = parseInt(hex.slice(3, 5), 16)
      const b = parseInt(hex.slice(5, 7), 16)
      ctx.fillStyle = `rgba(${r}, ${g}, ${b}, 0.012)`
      ctx.fillRect(lbl, y, plotW, bandH)

      // Separator line
      if (m > 0) {
        ctx.strokeStyle = 'rgba(80, 120, 180, 0.06)'
        ctx.lineWidth = 0.5
        ctx.beginPath()
        ctx.moveTo(lbl, y)
        ctx.lineTo(w, y)
        ctx.stroke()
      }

      // Channel label
      ctx.save()
      ctx.fillStyle = MODULE_COLOR_ARRAY[m]
      ctx.globalAlpha = 0.45
      ctx.font = '600 7px "JetBrains Mono", monospace'
      ctx.textBaseline = 'middle'
      ctx.textAlign = 'center'
      ctx.fillText(MODULE_ABBREVS[m], lbl / 2, y + bandH / 2)
      ctx.restore()

      offset += MODULE_NEURON_COUNTS[m]
    }

    // Render spikes with glow
    for (let col = 0; col < spikeHistory.length; col++) {
      for (const spike of spikeHistory[col]) {
        const mi = MOD_IDX[spike.source_module] ?? 0
        const mOff = MODULE_NEURON_COUNTS.slice(0, mi).reduce((a, b) => a + b, 0)
        const row = mOff + (spike.neuron_id % MODULE_NEURON_COUNTS[mi])
        const x = lbl + col * colW
        const y = row * rowH
        const hex = MODULE_COLOR_ARRAY[mi]
        const r = parseInt(hex.slice(1, 3), 16)
        const g = parseInt(hex.slice(3, 5), 16)
        const b = parseInt(hex.slice(5, 7), 16)
        const a = 0.45 + spike.strength * 0.55

        // Core pixel
        const dw = Math.max(colW, 1)
        const dh = Math.max(rowH, 1)
        ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${a})`
        ctx.fillRect(x, y, dw, dh)

        // Phosphor bloom on strong spikes
        if (spike.strength > 0.75) {
          ctx.fillStyle = `rgba(${Math.min(255,r+40)}, ${Math.min(255,g+40)}, ${Math.min(255,b+40)}, 0.12)`
          ctx.fillRect(x - 0.5, y - 0.5, dw + 1, dh + 1)
        }
      }
    }

    // Left divider
    ctx.fillStyle = 'rgba(80, 120, 180, 0.08)'
    ctx.fillRect(lbl - 0.5, 0, 0.5, h)

    // Subtle vertical time markers
    const interval = Math.max(1, Math.floor(spikeHistory.length / 8))
    for (let i = interval; i < spikeHistory.length; i += interval) {
      ctx.strokeStyle = 'rgba(80, 120, 180, 0.03)'
      ctx.lineWidth = 0.5
      ctx.beginPath()
      ctx.moveTo(lbl + i * colW, 0)
      ctx.lineTo(lbl + i * colW, h)
      ctx.stroke()
    }
  }, [spikeHistory])

  return (
    <div ref={containerRef} style={{ width: '100%', height: 'calc(100% - 22px)' }}>
      <canvas ref={canvasRef} style={{ display: 'block' }} />
    </div>
  )
}
