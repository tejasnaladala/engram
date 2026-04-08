import { Canvas, useFrame } from '@react-three/fiber'
import { OrbitControls } from '@react-three/drei'
import { useRef, useMemo } from 'react'
import * as THREE from 'three'
import type { ModuleSnapshot } from '../lib/protocol'
import { MODULE_COLOR_ARRAY, MODULE_NAMES } from '../lib/theme'

interface Props {
  modules: ModuleSnapshot[]
}

// Brain region positions (rough anatomical layout)
const POSITIONS: [number, number, number][] = [
  [-1.2, 0.8, 0],     // Sensory (occipital/parietal)
  [0, 0.3, 0.3],      // Associative (temporal)
  [0.5, 1.0, -0.3],   // Predictive (prefrontal)
  [-0.5, -0.3, 0.5],  // Episodic (hippocampal)
  [1.2, 0.0, 0],      // Action (motor cortex)
  [0, -0.8, 0],       // Safety (brainstem)
]

function BrainNode({ position, color, activity, name, index }: {
  position: [number, number, number]
  color: string
  activity: number
  name: string
  index: number
}) {
  const meshRef = useRef<THREE.Mesh>(null)
  const glowRef = useRef<THREE.Mesh>(null)
  const baseScale = 0.25 + (index === 1 ? 0.1 : 0) // Associative memory slightly larger

  useFrame((state) => {
    if (meshRef.current) {
      const pulse = 1 + activity * 0.3 * Math.sin(state.clock.elapsedTime * 3 + index)
      meshRef.current.scale.setScalar(baseScale * pulse)
    }
    if (glowRef.current) {
      glowRef.current.scale.setScalar(baseScale * (1.5 + activity * 0.5))
      ;(glowRef.current.material as THREE.MeshBasicMaterial).opacity = activity * 0.15
    }
  })

  const col = useMemo(() => new THREE.Color(color), [color])

  return (
    <group position={position}>
      {/* Glow sphere */}
      <mesh ref={glowRef}>
        <sphereGeometry args={[1, 16, 16]} />
        <meshBasicMaterial color={col} transparent opacity={0.1} />
      </mesh>
      {/* Core sphere */}
      <mesh ref={meshRef}>
        <sphereGeometry args={[1, 32, 32]} />
        <meshStandardMaterial
          color={col}
          emissive={col}
          emissiveIntensity={0.3 + activity * 0.7}
          roughness={0.3}
          metalness={0.1}
        />
      </mesh>
    </group>
  )
}

function Connections({ modules }: { modules: ModuleSnapshot[] }) {
  // Draw connections between related modules
  const connections = [
    [0, 1], // Sensory -> Associative
    [0, 2], // Sensory -> Predictive
    [1, 2], // Associative -> Predictive
    [1, 4], // Associative -> Action
    [2, 4], // Predictive -> Action
    [3, 1], // Episodic -> Associative
    [5, 4], // Safety -> Action (inhibitory)
  ]

  return (
    <>
      {connections.map(([from, to], i) => {
        const p1 = POSITIONS[from]
        const p2 = POSITIONS[to]
        const points = [new THREE.Vector3(...p1), new THREE.Vector3(...p2)]
        const geometry = new THREE.BufferGeometry().setFromPoints(points)
        const activity = Math.max(modules[from]?.activity_level || 0, modules[to]?.activity_level || 0)
        const isSafety = from === 5 || to === 5

        return (
          <line key={i} geometry={geometry}>
            <lineBasicMaterial
              color={isSafety ? '#ef4444' : '#ffffff'}
              transparent
              opacity={0.05 + activity * 0.15}
            />
          </line>
        )
      })}
    </>
  )
}

export default function BrainVisualization({ modules }: Props) {
  return (
    <div style={{ height: 'calc(100% - 28px)' }}>
      <Canvas
        camera={{ position: [0, 0.5, 3.5], fov: 45 }}
        gl={{ antialias: true }}
        style={{ background: '#0a0a0f' }}
      >
        <ambientLight intensity={0.2} />
        <pointLight position={[3, 3, 3]} intensity={0.5} color="#ffffff" />

        {modules.map((mod, i) => (
          <BrainNode
            key={i}
            position={POSITIONS[i]}
            color={MODULE_COLOR_ARRAY[i]}
            activity={mod.activity_level}
            name={MODULE_NAMES[i]}
            index={i}
          />
        ))}

        <Connections modules={modules} />

        <OrbitControls
          enableZoom={true}
          enablePan={false}
          autoRotate
          autoRotateSpeed={0.5}
          minDistance={2}
          maxDistance={6}
        />
      </Canvas>
    </div>
  )
}
