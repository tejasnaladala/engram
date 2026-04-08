import { useRef, useEffect } from 'react'
import type { SpikeEvent } from '../lib/protocol'
import { MODULE_COLOR_ARRAY } from '../lib/theme'

interface Props {
  spikeHistory: SpikeEvent[][]
}

const MODULE_INDEX: Record<string, number> = {
  'Sensory': 0,
  'AssociativeMemory': 1,
  'PredictiveError': 2,
  'EpisodicMemory': 3,
  'ActionSelector': 4,
  'SafetyKernel': 5,
}

const NEURON_COUNTS = [128, 256, 64, 64, 128, 32]
const TOTAL_NEURONS = NEURON_COUNTS.reduce((a, b) => a + b, 0) // 672

export default function SpikeRaster({ spikeHistory }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    const container = containerRef.current
    if (!canvas || !container) return

    const { width, height } = container.getBoundingClientRect()
    canvas.width = width
    canvas.height = height - 28 // subtract header

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // Clear
    ctx.fillStyle = '#0a0a0f'
    ctx.fillRect(0, 0, canvas.width, canvas.height)

    const colWidth = Math.max(1, canvas.width / Math.max(spikeHistory.length, 1))
    const rowHeight = canvas.height / TOTAL_NEURONS

    // Draw spikes
    for (let col = 0; col < spikeHistory.length; col++) {
      const spikes = spikeHistory[col]
      for (const spike of spikes) {
        const moduleIdx = MODULE_INDEX[spike.source_module] ?? 0
        const moduleOffset = NEURON_COUNTS.slice(0, moduleIdx).reduce((a, b) => a + b, 0)
        const neuronRow = moduleOffset + (spike.neuron_id % NEURON_COUNTS[moduleIdx])

        const x = col * colWidth
        const y = neuronRow * rowHeight

        ctx.fillStyle = MODULE_COLOR_ARRAY[moduleIdx]
        ctx.globalAlpha = 0.6 + spike.strength * 0.4
        ctx.fillRect(x, y, Math.max(colWidth, 1.5), Math.max(rowHeight, 1.5))
      }
    }
    ctx.globalAlpha = 1

    // Draw module separation lines and labels
    let offset = 0
    for (let m = 0; m < 6; m++) {
      const y = offset * rowHeight
      ctx.strokeStyle = 'rgba(255,255,255,0.08)'
      ctx.lineWidth = 0.5
      ctx.beginPath()
      ctx.moveTo(0, y)
      ctx.lineTo(canvas.width, y)
      ctx.stroke()

      // Module label
      ctx.fillStyle = MODULE_COLOR_ARRAY[m]
      ctx.globalAlpha = 0.4
      ctx.font = '9px monospace'
      ctx.fillText(['SEN', 'ASC', 'PRD', 'EPI', 'ACT', 'SAF'][m], 2, y + 10)
      ctx.globalAlpha = 1

      offset += NEURON_COUNTS[m]
    }
  }, [spikeHistory])

  return (
    <div ref={containerRef} className="w-full h-full">
      <canvas ref={canvasRef} className="w-full" style={{ height: 'calc(100% - 28px)' }} />
    </div>
  )
}
