import { useRef, useEffect, useState } from 'react'
import type { MemoryFormation } from '../lib/protocol'

interface MemoryHeatMapProps {
  formations: MemoryFormation[]
  tick: number
}

const GRID_SIZE = 20

/** Memory formation activation map — styled as a functional neuroimaging
 *  overlay (fMRI/PET activation map). Uses a "hot" colormap following
 *  standard neuroimaging conventions:
 *  black → deep red → red → orange → yellow → white
 *  This maps to: no activity → low → moderate → high → peak activation */
export default function MemoryHeatMap({ formations, tick }: MemoryHeatMapProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [grid, setGrid] = useState<Float32Array>(new Float32Array(GRID_SIZE * GRID_SIZE))

  useEffect(() => {
    setGrid(prev => {
      const next = new Float32Array(prev)
      // Exponential decay — simulates hemodynamic response falloff
      for (let i = 0; i < next.length; i++) {
        next[i] *= 0.993
      }
      // Add new formations
      for (const f of formations) {
        const idx = f.location_id % (GRID_SIZE * GRID_SIZE)
        next[idx] = Math.min(1.0, next[idx] + f.strength * 0.6)
        // Spatial spread — adjacent cells get partial activation (point spread function)
        const x = idx % GRID_SIZE
        const y = Math.floor(idx / GRID_SIZE)
        const neighbors = [
          [x - 1, y], [x + 1, y], [x, y - 1], [x, y + 1],
        ]
        for (const [nx, ny] of neighbors) {
          if (nx >= 0 && nx < GRID_SIZE && ny >= 0 && ny < GRID_SIZE) {
            const ni = ny * GRID_SIZE + nx
            next[ni] = Math.min(1.0, next[ni] + f.strength * 0.15)
          }
        }
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
    const size = Math.min(rect.width - 8, rect.height - 8)
    canvas.width = size
    canvas.height = size

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const cellW = size / GRID_SIZE
    const cellH = size / GRID_SIZE

    // Background — scan void
    ctx.fillStyle = '#060810'
    ctx.fillRect(0, 0, size, size)

    // Render activation map with "hot" colormap
    for (let y = 0; y < GRID_SIZE; y++) {
      for (let x = 0; x < GRID_SIZE; x++) {
        const val = grid[y * GRID_SIZE + x]
        if (val < 0.005) continue

        // Hot colormap (neuroimaging standard):
        // 0.0 → black
        // 0.2 → deep red (#400000)
        // 0.4 → red (#8b0000)
        // 0.6 → orange (#d44000)
        // 0.8 → yellow (#ffd000)
        // 1.0 → white
        let r: number, g: number, b: number
        if (val < 0.2) {
          const t = val / 0.2
          r = Math.floor(t * 80); g = 0; b = 0
        } else if (val < 0.4) {
          const t = (val - 0.2) / 0.2
          r = 80 + Math.floor(t * 59); g = 0; b = 0
        } else if (val < 0.6) {
          const t = (val - 0.4) / 0.2
          r = 139 + Math.floor(t * 73); g = Math.floor(t * 64); b = 0
        } else if (val < 0.8) {
          const t = (val - 0.6) / 0.2
          r = 212 + Math.floor(t * 43); g = 64 + Math.floor(t * 144); b = 0
        } else {
          const t = (val - 0.8) / 0.2
          r = 255; g = 208 + Math.floor(t * 47); b = Math.floor(t * 255)
        }

        ctx.fillStyle = `rgb(${r}, ${g}, ${b})`
        ctx.fillRect(
          x * cellW + 0.5,
          y * cellH + 0.5,
          cellW - 1,
          cellH - 1
        )
      }
    }

    // Calibration grid — subtle crosshatch like a scan overlay
    ctx.strokeStyle = 'rgba(100, 120, 160, 0.04)'
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

    // Crosshair center marker — neuronavigation reference
    const cx = size / 2
    const cy = size / 2
    ctx.strokeStyle = 'rgba(61, 216, 224, 0.12)'
    ctx.lineWidth = 0.5
    ctx.setLineDash([3, 3])
    ctx.beginPath()
    ctx.moveTo(cx, 0)
    ctx.lineTo(cx, size)
    ctx.stroke()
    ctx.beginPath()
    ctx.moveTo(0, cy)
    ctx.lineTo(size, cy)
    ctx.stroke()
    ctx.setLineDash([])

    // Corner registration marks
    const markLen = 6
    ctx.strokeStyle = 'rgba(100, 120, 160, 0.15)'
    ctx.lineWidth = 1
    // Top-left
    ctx.beginPath(); ctx.moveTo(2, 2 + markLen); ctx.lineTo(2, 2); ctx.lineTo(2 + markLen, 2); ctx.stroke()
    // Top-right
    ctx.beginPath(); ctx.moveTo(size - 2 - markLen, 2); ctx.lineTo(size - 2, 2); ctx.lineTo(size - 2, 2 + markLen); ctx.stroke()
    // Bottom-left
    ctx.beginPath(); ctx.moveTo(2, size - 2 - markLen); ctx.lineTo(2, size - 2); ctx.lineTo(2 + markLen, size - 2); ctx.stroke()
    // Bottom-right
    ctx.beginPath(); ctx.moveTo(size - 2 - markLen, size - 2); ctx.lineTo(size - 2, size - 2); ctx.lineTo(size - 2, size - 2 - markLen); ctx.stroke()

  }, [grid])

  return (
    <div style={{
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      height: 'calc(100% - 24px)',
      padding: '4px',
    }}>
      <canvas ref={canvasRef} style={{ borderRadius: '2px' }} />
    </div>
  )
}
