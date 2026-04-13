import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { OrbitControls, Text, Float } from '@react-three/drei'
import { useRef, useMemo, useState, useEffect } from 'react'
import * as THREE from 'three'

// Brain region positions (anatomically inspired layout)
const REGIONS = [
  { name: 'Somatosensory Cortex', abbr: 'S1', pos: [-1.0, 0.6, 0.2] as [number,number,number], color: '#38d8e8', size: 0.28 },
  { name: 'Association Cortex', abbr: 'ASC', pos: [0.0, 0.25, 0.45] as [number,number,number], color: '#8870f0', size: 0.36 },
  { name: 'Predictive Engine', abbr: 'PE', pos: [0.75, 0.85, -0.1] as [number,number,number], color: '#f09030', size: 0.26 },
  { name: 'Hippocampal Memory', abbr: 'HPC', pos: [-0.45, -0.15, 0.55] as [number,number,number], color: '#40c8d8', size: 0.22 },
  { name: 'Motor Cortex', abbr: 'M1', pos: [1.0, 0.05, 0.0] as [number,number,number], color: '#40c060', size: 0.30 },
  { name: 'Brainstem Governor', abbr: 'BST', pos: [0.0, -0.7, -0.1] as [number,number,number], color: '#e04848', size: 0.18 },
]

const CONNECTIONS = [
  [0,1],[0,2],[1,2],[1,4],[2,4],[3,1],[5,4],[3,0],[5,1],
]

// Spike particle traveling along a connection
function SpikeParticle({ from, to, color, speed, delay }: {
  from: [number,number,number], to: [number,number,number],
  color: string, speed: number, delay: number
}) {
  const ref = useRef<THREE.Mesh>(null)
  const startTime = useRef(delay)

  useFrame(({ clock }) => {
    if (!ref.current) return
    const t = clock.elapsedTime - startTime.current
    if (t < 0) { ref.current.visible = false; return }
    const progress = (t * speed) % 1
    ref.current.visible = true

    const p1 = new THREE.Vector3(...from)
    const p2 = new THREE.Vector3(...to)
    const mid = p1.clone().add(p2).multiplyScalar(0.5)
    mid.y += 0.2

    const curve = new THREE.QuadraticBezierCurve3(p1, mid, p2)
    const point = curve.getPoint(progress)
    ref.current.position.copy(point)

    const scale = 0.03 + Math.sin(progress * Math.PI) * 0.05
    ref.current.scale.setScalar(scale)

    const mat = ref.current.material as THREE.MeshBasicMaterial
    mat.opacity = 0.4 + Math.sin(progress * Math.PI) * 0.6
  })

  return (
    <mesh ref={ref} visible={false}>
      <sphereGeometry args={[1, 8, 8]} />
      <meshBasicMaterial color={color} transparent depthWrite={false} />
    </mesh>
  )
}

// Glowing brain region node
function BrainNode({ region, index }: { region: typeof REGIONS[0], index: number }) {
  const coreRef = useRef<THREE.Mesh>(null)
  const haloRef = useRef<THREE.Mesh>(null)
  const ringRef = useRef<THREE.Mesh>(null)
  const col = useMemo(() => new THREE.Color(region.color), [region.color])

  useFrame(({ clock }) => {
    const t = clock.elapsedTime
    const activity = 0.3 + 0.7 * (0.5 + 0.5 * Math.sin(t * 1.5 + index * 1.3))

    if (coreRef.current) {
      const pulse = 1 + activity * 0.08 * (
        Math.sin(t * 1.8 + index) * 0.6 + Math.sin(t * 3.1 + index * 0.7) * 0.4
      )
      coreRef.current.scale.setScalar(region.size * pulse)
      const mat = coreRef.current.material as THREE.MeshStandardMaterial
      mat.emissiveIntensity = 0.4 + activity * 1.2
    }

    if (haloRef.current) {
      haloRef.current.scale.setScalar(region.size * (2.0 + activity * 0.5))
      const mat = haloRef.current.material as THREE.MeshBasicMaterial
      mat.opacity = activity * 0.06
    }

    if (ringRef.current) {
      ringRef.current.rotation.z = t * 0.3 + index
      ringRef.current.scale.setScalar(region.size * (1.6 + activity * 0.3))
      const mat = ringRef.current.material as THREE.MeshBasicMaterial
      mat.opacity = activity * 0.12
    }
  })

  return (
    <group position={region.pos}>
      {/* Outer halo */}
      <mesh ref={haloRef}>
        <sphereGeometry args={[1, 24, 24]} />
        <meshBasicMaterial color={col} transparent opacity={0.08} depthWrite={false} />
      </mesh>

      {/* Holographic ring */}
      <mesh ref={ringRef} rotation={[Math.PI / 2, 0, 0]}>
        <ringGeometry args={[0.85, 1.0, 32]} />
        <meshBasicMaterial color={col} transparent opacity={0.15} side={THREE.DoubleSide} depthWrite={false} />
      </mesh>

      {/* Core sphere */}
      <mesh ref={coreRef}>
        <sphereGeometry args={[1, 48, 48]} />
        <meshStandardMaterial
          color={new THREE.Color(region.color).multiplyScalar(0.2)}
          emissive={col}
          emissiveIntensity={0.5}
          roughness={0.4}
          metalness={0.1}
          transparent
          opacity={0.85}
        />
      </mesh>

      {/* Label */}
      <Text
        position={[0, -region.size - 0.08, 0]}
        fontSize={0.06}
        color={region.color}
        anchorX="center"
        anchorY="top"
        font="https://fonts.gstatic.com/s/jetbrainsmono/v18/tDbY2o-flEEny0FZhsfKu5WU4zr3E_BX0PnT8RD8yKxjPVmNsQ.woff2"
        outlineWidth={0}
        fillOpacity={0.6}
      >
        {region.abbr}
      </Text>
    </group>
  )
}

// Fiber tract connections with glow
function FiberTracts() {
  return (
    <>
      {CONNECTIONS.map(([f, t], i) => {
        const p1 = new THREE.Vector3(...REGIONS[f].pos)
        const p2 = new THREE.Vector3(...REGIONS[t].pos)
        const mid = p1.clone().add(p2).multiplyScalar(0.5)
        mid.y += 0.15
        const curve = new THREE.QuadraticBezierCurve3(p1, mid, p2)
        const geo = new THREE.BufferGeometry().setFromPoints(curve.getPoints(24))
        const isSafety = f === 5 || t === 5
        return (
          <line key={i} geometry={geo}>
            <lineBasicMaterial
              color={isSafety ? '#e04848' : '#405878'}
              transparent opacity={0.15}
            />
          </line>
        )
      })}
    </>
  )
}

// Many spike particles flowing through connections
function SpikeFlow() {
  const particles = useMemo(() => {
    const p: { from: [number,number,number], to: [number,number,number], color: string, speed: number, delay: number }[] = []
    for (let wave = 0; wave < 4; wave++) {
      for (const [f, t] of CONNECTIONS) {
        p.push({
          from: REGIONS[f].pos,
          to: REGIONS[t].pos,
          color: REGIONS[f].color,
          speed: 0.3 + Math.random() * 0.4,
          delay: wave * 2.5 + Math.random() * 2,
        })
      }
    }
    return p
  }, [])

  return (
    <>
      {particles.map((p, i) => (
        <SpikeParticle key={i} {...p} />
      ))}
    </>
  )
}

// Holographic grid floor
function HoloGrid() {
  return (
    <group position={[0, -1.1, 0]}>
      <gridHelper args={[5, 30, '#0a1828', '#081420']} />
      <gridHelper args={[5, 6, '#102030', '#0c1828']} />
    </group>
  )
}

// Use OrbitControls with auto-rotate instead of manual camera rig
// (manual camera rig breaks WebGL capture in Playwright)

// CRT scanline overlay
function CRTOverlay() {
  return (
    <div style={{
      position: 'fixed', inset: 0, pointerEvents: 'none', zIndex: 10,
      background: `repeating-linear-gradient(
        0deg,
        transparent,
        transparent 2px,
        rgba(0,0,0,0.03) 2px,
        rgba(0,0,0,0.03) 4px
      )`,
      mixBlendMode: 'multiply',
    }}>
      {/* Vignette */}
      <div style={{
        position: 'absolute', inset: 0,
        background: 'radial-gradient(ellipse at center, transparent 50%, rgba(0,0,0,0.4) 100%)',
      }} />
      {/* CRT curve effect */}
      <div style={{
        position: 'absolute', inset: 0,
        boxShadow: 'inset 0 0 80px rgba(0,0,0,0.3), inset 0 0 200px rgba(0,0,0,0.15)',
      }} />
    </div>
  )
}

// Telemetry readouts
function TelemetryHUD({ tick }: { tick: number }) {
  const t = tick * 0.001
  const hz = (950 + Math.sin(t) * 50).toFixed(0)
  const spk = (tick * 27).toLocaleString()
  const pe = (0.3 + 0.2 * Math.sin(t * 0.5)).toFixed(3)
  const syn = '47.2K'
  const energy = (tick * 0.027).toFixed(1)

  const style = {
    fontFamily: "'JetBrains Mono', monospace",
    letterSpacing: '1px',
  }

  return (
    <>
      {/* Top bar */}
      <div style={{
        position: 'fixed', top: 0, left: 0, right: 0, zIndex: 20,
        height: '36px', display: 'flex', alignItems: 'center',
        padding: '0 16px', gap: '16px',
        background: 'linear-gradient(180deg, rgba(6,8,16,0.95), rgba(6,8,16,0.7))',
        borderBottom: '1px solid rgba(56,216,232,0.08)',
        ...style,
      }}>
        {/* Logo */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <div style={{
            width: '5px', height: '5px', borderRadius: '1px',
            background: '#38d8e8',
            boxShadow: '0 0 10px #38d8e8, 0 0 3px #38d8e8',
            animation: 'pulse 2s ease-in-out infinite',
          }} />
          <span style={{ fontSize: '11px', fontWeight: 700, letterSpacing: '3px', color: '#38d8e8' }}>
            ENGRAM
          </span>
          <span style={{ fontSize: '8px', color: '#3a4060', letterSpacing: '1.5px', marginLeft: '4px' }}>
            NEURAL COGNITIVE SYSTEM
          </span>
        </div>

        <div style={{ width: '1px', height: '18px', background: 'rgba(56,216,232,0.1)' }} />

        {/* Metrics */}
        <Metric label="Hz" value={hz} color="#40c060" />
        <Metric label="SPK" value={spk} color="#38d8e8" />
        <Metric label="PE" value={pe} color="#f09030" />
        <Metric label="SYN" value={syn} color="#8870f0" />
        <Metric label="E" value={energy + 'J'} color="#40c8d8" />

        <div style={{ flex: 1 }} />

        <span style={{ fontSize: '8px', color: '#38d8e8', letterSpacing: '1.5px' }}>
          LIVE
        </span>
        <div style={{
          width: '4px', height: '4px', borderRadius: '50%',
          background: '#40c060', boxShadow: '0 0 6px #40c060',
        }} />
      </div>

      {/* Bottom status */}
      <div style={{
        position: 'fixed', bottom: 0, left: 0, right: 0, zIndex: 20,
        height: '24px', display: 'flex', alignItems: 'center',
        padding: '0 16px', gap: '20px',
        background: 'linear-gradient(0deg, rgba(6,8,16,0.95), rgba(6,8,16,0.7))',
        borderTop: '1px solid rgba(56,216,232,0.06)',
        ...style, fontSize: '7px', color: '#2a3450', letterSpacing: '1.5px',
      }}>
        <span>ENGRAM NCS v0.1.0</span>
        <span style={{ color: '#1a2038' }}>|</span>
        <span>LIF/STDP RUNTIME</span>
        <span style={{ color: '#1a2038' }}>|</span>
        <span>672 NEURONS</span>
        <span style={{ color: '#1a2038' }}>|</span>
        <span>6 REGIONS</span>
        <span style={{ color: '#1a2038' }}>|</span>
        <span>SURROGATE GRADIENT TRAINING</span>
        <div style={{ flex: 1 }} />
        <span>55.3% MAZE SUCCESS vs 41.3% Q-LEARNING</span>
        <div style={{
          width: '3px', height: '3px', borderRadius: '50%',
          background: '#38d8e8', boxShadow: '0 0 8px #38d8e8',
          animation: 'pulse 3s ease-in-out infinite',
        }} />
      </div>

      {/* Left region labels */}
      <div style={{
        position: 'fixed', left: '16px', top: '50%', transform: 'translateY(-50%)',
        zIndex: 20, display: 'flex', flexDirection: 'column', gap: '6px',
        ...style, fontSize: '7px',
      }}>
        {REGIONS.map((r, i) => {
          const activity = 0.3 + 0.7 * (0.5 + 0.5 * Math.sin(t * 2 + i * 1.1))
          return (
            <div key={i} style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <div style={{
                width: '3px', height: '3px', borderRadius: '50%',
                background: r.color, boxShadow: `0 0 4px ${r.color}`,
                opacity: 0.4 + activity * 0.6,
              }} />
              <span style={{ color: r.color, opacity: 0.5, letterSpacing: '0.5px', width: '24px' }}>
                {r.abbr}
              </span>
              <div style={{
                width: '40px', height: '2px', background: '#0a0d16',
                borderRadius: '1px', overflow: 'hidden',
              }}>
                <div style={{
                  width: `${activity * 100}%`, height: '100%',
                  background: `linear-gradient(90deg, ${r.color}40, ${r.color})`,
                  transition: 'width 0.3s',
                }} />
              </div>
              <span style={{ color: '#2a3450', width: '20px', textAlign: 'right' }}>
                {(activity * 100).toFixed(0)}%
              </span>
            </div>
          )
        })}
      </div>

      {/* Right side: mini spike visualization */}
      <div style={{
        position: 'fixed', right: '16px', top: '50%', transform: 'translateY(-50%)',
        zIndex: 20, ...style, fontSize: '7px', color: '#2a3450',
        display: 'flex', flexDirection: 'column', gap: '4px', alignItems: 'flex-end',
      }}>
        <span style={{ letterSpacing: '1.5px', marginBottom: '4px' }}>SPIKE TRACE</span>
        <MiniSpikeTrace tick={tick} />
      </div>
    </>
  )
}

function Metric({ label, value, color }: { label: string, value: string, color: string }) {
  return (
    <div style={{ display: 'flex', alignItems: 'baseline', gap: '4px' }}>
      <span style={{ fontSize: '7px', color: '#2a3450', letterSpacing: '1px' }}>{label}</span>
      <span style={{ fontSize: '11px', color, fontWeight: 500 }}>{value}</span>
    </div>
  )
}

function MiniSpikeTrace({ tick }: { tick: number }) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    const w = 120, h = 60
    canvas.width = w * 2; canvas.height = h * 2
    canvas.style.width = w + 'px'; canvas.style.height = h + 'px'
    ctx.scale(2, 2)

    ctx.fillStyle = '#060810'
    ctx.fillRect(0, 0, w, h)

    const colors = ['#38d8e8', '#8870f0', '#f09030', '#40c8d8', '#40c060', '#e04848']
    const t = tick * 0.001

    for (let m = 0; m < 6; m++) {
      const y0 = m * 10
      const activity = 0.3 + 0.7 * (0.5 + 0.5 * Math.sin(t * 2 + m * 1.1))
      const count = Math.floor(activity * 8)
      for (let s = 0; s < count; s++) {
        const x = (tick * 0.3 + s * 15 + m * 7) % w
        ctx.fillStyle = colors[m]
        ctx.globalAlpha = 0.4 + Math.random() * 0.4
        ctx.fillRect(x, y0 + Math.random() * 8, 1.5, 1.5)
      }
    }
    ctx.globalAlpha = 1

    // Separator lines
    for (let m = 1; m < 6; m++) {
      ctx.strokeStyle = 'rgba(56,216,232,0.04)'
      ctx.lineWidth = 0.5
      ctx.beginPath(); ctx.moveTo(0, m * 10); ctx.lineTo(w, m * 10); ctx.stroke()
    }
  }, [tick])

  return <canvas ref={canvasRef} style={{ borderRadius: '2px', border: '1px solid rgba(56,216,232,0.06)' }} />
}

// Main cinematic component
export default function CinematicDemo() {
  const [tick, setTick] = useState(0)

  useEffect(() => {
    const interval = setInterval(() => setTick(t => t + 10), 33)
    return () => clearInterval(interval)
  }, [])

  return (
    <div style={{
      width: '100vw', height: '100vh',
      background: '#020205',
      position: 'relative', overflow: 'hidden',
    }}>
      {/* CRT effect overlay */}
      <CRTOverlay />

      {/* HUD telemetry */}
      <TelemetryHUD tick={tick} />

      {/* 3D Brain scene */}
      <Canvas
        camera={{ position: [0, 0.8, 3.5], fov: 36 }}
        gl={{ antialias: true, alpha: true, powerPreference: 'high-performance' }}
        style={{ background: '#020205' }}
      >
        {/* Cinematic lighting */}
        <ambientLight intensity={0.08} />
        <directionalLight position={[3, 4, 2]} intensity={0.3} color="#8090b0" />
        <directionalLight position={[-2, 1, -1]} intensity={0.12} color="#405080" />
        <pointLight position={[0, 0.3, 0]} intensity={0.25} color="#38d8e8" distance={4} decay={2} />
        <pointLight position={[-1, -0.5, 1]} intensity={0.10} color="#8870f0" distance={3} decay={2} />
        <pointLight position={[1, 0.8, -0.5]} intensity={0.08} color="#f09030" distance={3} decay={2} />

        {/* Brain regions */}
        {REGIONS.map((r, i) => (
          <BrainNode key={i} region={r} index={i} />
        ))}

        {/* Fiber tract connections */}
        <FiberTracts />

        {/* Animated spike particles */}
        <SpikeFlow />

        {/* Holographic floor grid */}
        <HoloGrid />

        {/* Auto-rotating orbit controls */}
        <OrbitControls
          enableZoom={false}
          enablePan={false}
          autoRotate
          autoRotateSpeed={0.4}
          minPolarAngle={Math.PI * 0.3}
          maxPolarAngle={Math.PI * 0.65}
        />
      </Canvas>

      {/* CSS animations */}
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&display=swap');
        @keyframes pulse { 0%,100%{opacity:0.4}50%{opacity:1} }
        * { margin:0; padding:0; box-sizing:border-box; }
        body { overflow:hidden; background:#020205; }
      `}</style>
    </div>
  )
}
