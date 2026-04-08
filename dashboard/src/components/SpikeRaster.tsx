import { useRef, useEffect } from 'react'
import type { SpikeEvent } from '../lib/protocol'
import { MODULE_COLOR_ARRAY, MODULE_ABBREVS, MODULE_NEURON_COUNTS, TOTAL_NEURONS } from '../lib/theme'

interface SpikeRasterProps {
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

/** Electrophysiology-inspired multi-channel spike raster display.
 *  Modeled after EEG monitoring systems and multi-electrode array recordings.
 *  Each row = one neuron. Each column = one time frame. Spikes rendered as
 *  luminous dots with color encoding the source brain region. */
export default function SpikeRaster({ spikeHistory }: SpikeRasterProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    const container = containerRef.current
    if (!canvas || !container) return

    const rect = container.getBoundingClientRect()
    const labelWidth = 30 // space for channel labels
    const w = rect.width
    const h = rect.height
    canvas.width = w
    canvas.height = h

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // Background — near-black like a clinical monitor
    ctx.fillStyle = '#060810'
    ctx.fillRect(0, 0, w, h)

    const plotW = w - labelWidth
    const colWidth = Math.max(0.8, plotW / Math.max(spikeHistory.length, 1))
    const rowHeight = h / TOTAL_NEURONS

    // Draw subtle horizontal grid lines at module boundaries — electrode channel separators
    let offset = 0
    for (let m = 0; m < 6; m++) {
      const y = offset * rowHeight

      // Module separator line
      ctx.strokeStyle = 'rgba(100, 120, 160, 0.08)'
      ctx.lineWidth = 0.5
      ctx.beginPath()
      ctx.moveTo(labelWidth, y)
      ctx.lineTo(w, y)
      ctx.stroke()

      // Module band background — very subtle region tint
      ctx.fillStyle = MODULE_COLOR_ARRAY[m].replace(')', ', 0.015)').replace('rgb', 'rgba').replace('#', '')
      // Use hex-to-rgba: parse the hex color
      const hex = MODULE_COLOR_ARRAY[m]
      const r = parseInt(hex.slice(1, 3), 16)
      const g = parseInt(hex.slice(3, 5), 16)
      const b = parseInt(hex.slice(5, 7), 16)
      ctx.fillStyle = `rgba(${r}, ${g}, ${b}, 0.02)`
      ctx.fillRect(labelWidth, y, plotW, MODULE_NEURON_COUNTS[m] * rowHeight)

      // Channel label — clinical abbreviation
      ctx.save()
      ctx.fillStyle = MODULE_COLOR_ARRAY[m]
      ctx.globalAlpha = 0.5
      ctx.font = '700 8px "JetBrains Mono", monospace'
      ctx.textBaseline = 'middle'
      ctx.textAlign = 'center'
      const labelY = y + (MODULE_NEURON_COUNTS[m] * rowHeight) / 2
      ctx.fillText(MODULE_ABBREVS[m], labelWidth / 2, labelY)
      ctx.restore()

      offset += MODULE_NEURON_COUNTS[m]
    }

    // Draw spikes — luminous dots like electrode recordings
    for (let col = 0; col < spikeHistory.length; col++) {
      const spikes = spikeHistory[col]
      for (const spike of spikes) {
        const moduleIdx = MODULE_INDEX[spike.source_module] ?? 0
        const moduleOffset = MODULE_NEURON_COUNTS.slice(0, moduleIdx).reduce((a, b) => a + b, 0)
        const neuronRow = moduleOffset + (spike.neuron_id % MODULE_NEURON_COUNTS[moduleIdx])

        const x = labelWidth + col * colWidth
        const y = neuronRow * rowHeight

        // Core spike dot
        const hex = MODULE_COLOR_ARRAY[moduleIdx]
        const r = parseInt(hex.slice(1, 3), 16)
        const g = parseInt(hex.slice(3, 5), 16)
        const b = parseInt(hex.slice(5, 7), 16)
        const alpha = 0.5 + spike.strength * 0.5

        ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${alpha})`
        const dotW = Math.max(colWidth, 1.2)
        const dotH = Math.max(rowHeight, 1.2)
        ctx.fillRect(x, y, dotW, dotH)

        // Glow for strong spikes — simulates phosphor persistence
        if (spike.strength > 0.8) {
          ctx.fillStyle = `rgba(${r}, ${g}, ${b}, 0.15)`
          ctx.fillRect(x - 0.5, y - 0.5, dotW + 1, dotH + 1)
        }
      }
    }

    // Time axis reference marks at bottom
    ctx.fillStyle = 'rgba(100, 120, 160, 0.15)'
    ctx.fillRect(labelWidth, h - 1, plotW, 1)

    // Vertical time markers every ~50 frames
    const markerInterval = Math.max(1, Math.floor(spikeHistory.length / 6))
    for (let i = markerInterval; i < spikeHistory.length; i += markerInterval) {
      const x = labelWidth + i * colWidth
      ctx.strokeStyle = 'rgba(100, 120, 160, 0.05)'
      ctx.lineWidth = 0.5
      ctx.beginPath()
      ctx.moveTo(x, 0)
      ctx.lineTo(x, h)
      ctx.stroke()
    }

    // Left border separator
    ctx.fillStyle = 'rgba(100, 120, 160, 0.1)'
    ctx.fillRect(labelWidth - 1, 0, 1, h)

  }, [spikeHistory])

  return (
    <div ref={containerRef} style={{ width: '100%', height: 'calc(100% - 24px)' }}>
      <canvas ref={canvasRef} style={{ width: '100%', height: '100%', display: 'block' }} />
    </div>
  )
}
