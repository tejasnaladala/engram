import { useRef, useEffect, useState } from 'react'
import type { MemoryFormation } from '../lib/protocol'

interface Props {
  formations: MemoryFormation[]
  tick: number
}

const GRID_SIZE = 20 // 20x20 heatmap grid

export default function MemoryHeatMap({ formations, tick }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [grid, setGrid] = useState<Float32Array>(new Float32Array(GRID_SIZE * GRID_SIZE))

  useEffect(() => {
    // Update grid with new formations
    setGrid(prev => {
      const next = new Float32Array(prev)
      // Decay
      for (let i = 0; i < next.length; i++) {
        next[i] *= 0.995
      }
      // Add new formations
      for (const f of formations) {
        const idx = f.location_id % (GRID_SIZE * GRID_SIZE)
        next[idx] = Math.min(1.0, next[idx] + f.strength * 0.5)
      }
      return next
    })
  }, [formations, tick])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const container = canvas.parentElement
    if (!container) return

    const rect = container.getBoundingClientRect()
    const size = Math.min(rect.width, rect.height - 28)
    canvas.width = size
    canvas.height = size

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const cellW = size / GRID_SIZE
    const cellH = size / GRID_SIZE

    ctx.fillStyle = '#0a0a0f'
    ctx.fillRect(0, 0, size, size)

    for (let y = 0; y < GRID_SIZE; y++) {
      for (let x = 0; x < GRID_SIZE; x++) {
        const val = grid[y * GRID_SIZE + x]
        if (val < 0.01) continue

        // Warm color gradient: black -> dark red -> orange -> yellow -> white
        const r = Math.min(255, Math.floor(val * 3 * 255))
        const g = Math.min(255, Math.floor(Math.max(0, val * 2 - 0.3) * 255))
        const b = Math.min(255, Math.floor(Math.max(0, val - 0.7) * 255 * 0.5))

        ctx.fillStyle = `rgb(${r}, ${g}, ${b})`
        ctx.fillRect(x * cellW + 0.5, y * cellH + 0.5, cellW - 1, cellH - 1)
      }
    }

    // Grid lines
    ctx.strokeStyle = 'rgba(255,255,255,0.03)'
    ctx.lineWidth = 0.5
    for (let i = 0; i <= GRID_SIZE; i++) {
      ctx.beginPath()
      ctx.moveTo(i * cellW, 0)
      ctx.lineTo(i * cellW, size)
      ctx.stroke()
      ctx.beginPath()
      ctx.moveTo(0, i * cellH)
      ctx.lineTo(size, i * cellH)
      ctx.stroke()
    }
  }, [grid])

  return (
    <div className="flex items-center justify-center p-2" style={{ height: 'calc(100% - 28px)' }}>
      <canvas ref={canvasRef} />
    </div>
  )
}
