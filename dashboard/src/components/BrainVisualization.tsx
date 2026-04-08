import { Canvas, useFrame } from '@react-three/fiber'
import { OrbitControls, Float } from '@react-three/drei'
import { useRef, useMemo } from 'react'
import * as THREE from 'three'
import type { ModuleSnapshot } from '../lib/protocol'
import { MODULE_COLOR_ARRAY, MODULE_ABBREVS } from '../lib/theme'

interface BrainVisualizationProps {
  modules: ModuleSnapshot[]
}

// Anatomically-informed positions mapped to approximate brain regions:
// Uses radiological convention (left hemisphere on viewer's right)
const POSITIONS: [number, number, number][] = [
  [-0.9, 0.6, 0.2],     // S1: Primary somatosensory — postcentral gyrus, parietal
  [0.0, 0.2, 0.5],      // ASC: Association cortex — temporal/parietal junction
  [0.7, 0.9, -0.1],     // PE: Prefrontal prediction — anterior prefrontal
  [-0.4, -0.2, 0.6],    // HPC: Hippocampus — medial temporal, deep
  [0.9, 0.1, 0.0],      // M1: Primary motor — precentral gyrus
  [0.0, -0.7, -0.2],    // BST: Brainstem — midline, inferior
]

// Size scaling: larger for cortical areas, smaller for subcortical
const BASE_SIZES = [0.22, 0.28, 0.20, 0.16, 0.24, 0.14]

/** Individual neural module node — volumetric sphere with clinical glow */
function NeuralNode({ position, color, activity, size, abbrev, index }: {
  position: [number, number, number]
  color: string
  activity: number
  size: number
  abbrev: string
  index: number
}) {
  const coreRef = useRef<THREE.Mesh>(null)
  const outerRef = useRef<THREE.Mesh>(null)
  const innerGlowRef = useRef<THREE.Mesh>(null)

  const col = useMemo(() => new THREE.Color(color), [color])
  const dimCol = useMemo(() => {
    const c = new THREE.Color(color)
    c.multiplyScalar(0.3)
    return c
  }, [color])

  useFrame((state) => {
    const t = state.clock.elapsedTime

    if (coreRef.current) {
      // Subtle organic pulsation — not mechanical sine wave
      const pulse = 1 + activity * 0.12 * (
        Math.sin(t * 1.8 + index * 1.1) * 0.6 +
        Math.sin(t * 3.2 + index * 0.7) * 0.4
      )
      coreRef.current.scale.setScalar(size * pulse)

      // Emissive intensity tracks activity
      const mat = coreRef.current.material as THREE.MeshStandardMaterial
      mat.emissiveIntensity = 0.15 + activity * 0.6
    }

    if (outerRef.current) {
      outerRef.current.scale.setScalar(size * (1.6 + activity * 0.3))
      const mat = outerRef.current.material as THREE.MeshBasicMaterial
      mat.opacity = activity * 0.08
    }

    if (innerGlowRef.current) {
      innerGlowRef.current.scale.setScalar(size * (1.15 + activity * 0.15))
      const mat = innerGlowRef.current.material as THREE.MeshBasicMaterial
      mat.opacity = 0.03 + activity * 0.12
    }
  })

  return (
    <group position={position}>
      {/* Outer volumetric halo — simulates CT/MRI scan glow */}
      <mesh ref={outerRef}>
        <sphereGeometry args={[1, 24, 24]} />
        <meshBasicMaterial color={col} transparent opacity={0.05} depthWrite={false} />
      </mesh>

      {/* Inner glow layer — subsurface scattering approximation */}
      <mesh ref={innerGlowRef}>
        <sphereGeometry args={[1, 24, 24]} />
        <meshBasicMaterial color={col} transparent opacity={0.08} depthWrite={false} />
      </mesh>

      {/* Core structure — the actual "brain region" */}
      <mesh ref={coreRef}>
        <sphereGeometry args={[1, 48, 48]} />
        <meshStandardMaterial
          color={dimCol}
          emissive={col}
          emissiveIntensity={0.3}
          roughness={0.6}
          metalness={0.05}
          transparent
          opacity={0.9}
        />
      </mesh>
    </group>
  )
}

/** Axonal connection lines between modules — styled as fiber tract pathways */
function FiberTracts({ modules }: { modules: ModuleSnapshot[] }) {
  const connections = [
    [0, 1], // S1 → Association (thalamocortical)
    [0, 2], // S1 → Prefrontal (dorsal stream)
    [1, 2], // Association → Prefrontal (ventral stream)
    [1, 4], // Association → Motor (premotor pathway)
    [2, 4], // Prefrontal → Motor (executive control)
    [3, 1], // Hippocampus → Association (memory retrieval)
    [5, 4], // Brainstem → Motor (autonomic override)
  ]

  return (
    <>
      {connections.map(([from, to], i) => {
        const p1 = new THREE.Vector3(...POSITIONS[from])
        const p2 = new THREE.Vector3(...POSITIONS[to])

        // Create a curved path — fiber tracts are not straight lines
        const mid = new THREE.Vector3().addVectors(p1, p2).multiplyScalar(0.5)
        mid.y += 0.15 // slight arc upward
        const curve = new THREE.QuadraticBezierCurve3(p1, mid, p2)
        const points = curve.getPoints(20)
        const geometry = new THREE.BufferGeometry().setFromPoints(points)

        const activity = Math.max(
          modules[from]?.activity_level || 0,
          modules[to]?.activity_level || 0
        )
        const isSafety = from === 5 || to === 5
        const color = isSafety ? '#d94040' : '#4a6080'

        return (
          <line key={i} geometry={geometry}>
            <lineBasicMaterial
              color={color}
              transparent
              opacity={0.04 + activity * 0.12}
              linewidth={1}
            />
          </line>
        )
      })}
    </>
  )
}

/** Ambient scan grid — calibration reference plane */
function ScanGrid() {
  return (
    <gridHelper
      args={[4, 20, 'rgba(100, 120, 160, 0.03)', 'rgba(100, 120, 160, 0.02)']}
      position={[0, -1.0, 0]}
      rotation={[0, 0, 0]}
    />
  )
}

export default function BrainVisualization({ modules }: BrainVisualizationProps) {
  return (
    <div style={{ height: 'calc(100% - 24px)', position: 'relative' }}>
      {/* Scan orientation labels — radiological convention */}
      <div style={{
        position: 'absolute',
        top: '4px',
        left: '8px',
        zIndex: 2,
        fontFamily: 'var(--font-clinical)',
        fontSize: '8px',
        color: 'rgba(100, 120, 160, 0.3)',
        letterSpacing: '1px',
      }}>
        ANT
      </div>
      <div style={{
        position: 'absolute',
        bottom: '4px',
        right: '8px',
        zIndex: 2,
        fontFamily: 'var(--font-clinical)',
        fontSize: '8px',
        color: 'rgba(100, 120, 160, 0.3)',
        letterSpacing: '1px',
      }}>
        POST
      </div>

      <Canvas
        camera={{ position: [0, 0.8, 3.2], fov: 40 }}
        gl={{ antialias: true, alpha: true }}
        style={{ background: '#060810' }}
      >
        {/* Clinical lighting — subdued, directional */}
        <ambientLight intensity={0.12} />
        <directionalLight position={[2, 3, 2]} intensity={0.3} color="#c0d0e0" />
        <directionalLight position={[-2, 1, -1]} intensity={0.1} color="#6080a0" />

        {/* Point light at center — simulates volumetric scan illumination */}
        <pointLight position={[0, 0.2, 0]} intensity={0.15} color="#3dd8e0" distance={3} decay={2} />

        {modules.map((mod, i) => (
          <NeuralNode
            key={i}
            position={POSITIONS[i]}
            color={MODULE_COLOR_ARRAY[i]}
            activity={mod.activity_level}
            size={BASE_SIZES[i]}
            abbrev={MODULE_ABBREVS[i]}
            index={i}
          />
        ))}

        <FiberTracts modules={modules} />
        <ScanGrid />

        <OrbitControls
          enableZoom={true}
          enablePan={false}
          autoRotate
          autoRotateSpeed={0.3}
          minDistance={2.0}
          maxDistance={5.5}
          maxPolarAngle={Math.PI * 0.75}
          minPolarAngle={Math.PI * 0.25}
        />
      </Canvas>
    </div>
  )
}
