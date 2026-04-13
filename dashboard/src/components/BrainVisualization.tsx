import { Canvas, useFrame } from '@react-three/fiber'
import { OrbitControls, Text } from '@react-three/drei'
import { useRef, useMemo } from 'react'
import * as THREE from 'three'
import type { ModuleSnapshot } from '../lib/protocol'
import { MODULE_ABBREVS } from '../lib/theme'

interface BrainVisualizationProps { modules: ModuleSnapshot[] }

// Muted monochrome-ish palette -- NOT rainbow colored circles
// Each region uses the SAME base cyan/teal, just different brightness
const REGION_COLORS = [
  '#3098a8', // S1 -- muted teal
  '#4878b0', // ASC -- steel blue
  '#5070a0', // PE -- slate
  '#388898', // HPC -- sea
  '#408878', // M1 -- sage
  '#706070', // BST -- muted mauve
]

const POS: [number,number,number][] = [
  [-0.9, 0.5, 0.15],   // S1
  [0.0, 0.2, 0.45],    // ASC (center, large)
  [0.7, 0.75, -0.1],   // PE
  [-0.4, -0.15, 0.5],  // HPC
  [0.85, 0.0, 0.0],    // M1
  [0.0, -0.6, -0.1],   // BST
]

const SZ = [0.24, 0.32, 0.22, 0.18, 0.26, 0.15]

function Node({ pos, color, act, size, idx }: {
  pos:[number,number,number]; color:string; act:number; size:number; idx:number
}) {
  const coreRef = useRef<THREE.Mesh>(null)
  const haloRef = useRef<THREE.Mesh>(null)
  const ringRef = useRef<THREE.Mesh>(null)
  const col = useMemo(() => new THREE.Color(color), [color])
  const dim = useMemo(() => new THREE.Color(color).multiplyScalar(0.15), [color])

  useFrame(({ clock }) => {
    const t = clock.elapsedTime
    if (coreRef.current) {
      const p = 1 + act * 0.08 * (Math.sin(t*1.6+idx)*0.7 + Math.sin(t*2.9+idx*0.5)*0.3)
      coreRef.current.scale.setScalar(size * p)
      ;(coreRef.current.material as THREE.MeshStandardMaterial).emissiveIntensity = 0.15 + act * 0.6
    }
    if (haloRef.current) {
      haloRef.current.scale.setScalar(size * (1.8 + act * 0.4))
      ;(haloRef.current.material as THREE.MeshBasicMaterial).opacity = act * 0.04
    }
    if (ringRef.current) {
      ringRef.current.rotation.z = t * 0.2 + idx
      ringRef.current.scale.setScalar(size * (1.5 + act * 0.3))
      ;(ringRef.current.material as THREE.MeshBasicMaterial).opacity = 0.04 + act * 0.08
    }
  })

  return (
    <group position={pos}>
      {/* Outer halo */}
      <mesh ref={haloRef}>
        <sphereGeometry args={[1, 20, 20]} />
        <meshBasicMaterial color={col} transparent opacity={0.03} depthWrite={false} />
      </mesh>

      {/* Holographic ring */}
      <mesh ref={ringRef} rotation={[Math.PI / 2, 0, 0]}>
        <ringGeometry args={[0.85, 1.0, 32]} />
        <meshBasicMaterial color={col} transparent opacity={0.06} side={THREE.DoubleSide} depthWrite={false} />
      </mesh>

      {/* Core */}
      <mesh ref={coreRef}>
        <sphereGeometry args={[1, 48, 48]} />
        <meshStandardMaterial color={dim} emissive={col} emissiveIntensity={0.3}
          roughness={0.5} metalness={0.1} transparent opacity={0.85} />
      </mesh>

      {/* Label */}
      <Text
        position={[0, -size - 0.06, 0]}
        fontSize={0.055}
        color={color}
        anchorX="center"
        anchorY="top"
        fillOpacity={0.5}
      >
        {MODULE_ABBREVS[idx]}
      </Text>
    </group>
  )
}

function Tracts({ modules }: { modules: ModuleSnapshot[] }) {
  const conns = [[0,1],[0,2],[1,2],[1,4],[2,4],[3,1],[5,4]]
  return <>
    {conns.map(([f,t],i) => {
      const p1 = new THREE.Vector3(...POS[f]), p2 = new THREE.Vector3(...POS[t])
      const mid = p1.clone().add(p2).multiplyScalar(0.5); mid.y += 0.1
      const curve = new THREE.QuadraticBezierCurve3(p1, mid, p2)
      const geo = new THREE.BufferGeometry().setFromPoints(curve.getPoints(14))
      const act = Math.max(modules[f]?.activity_level||0, modules[t]?.activity_level||0)
      return <line key={i} geometry={geo}>
        <lineBasicMaterial color={f===5||t===5?'#604050':'#283848'} transparent opacity={0.04+act*0.08} />
      </line>
    })}
  </>
}

export default function BrainVisualization({ modules }: BrainVisualizationProps) {
  return (
    <div style={{ height:'calc(100% - 22px)', position:'relative' }}>
      {/* Orientation markers */}
      <div style={{ position:'absolute',top:'4px',left:'8px',zIndex:2,fontFamily:'var(--mono)',fontSize:'7px',color:'rgba(80,120,180,0.2)',letterSpacing:'1px' }}>A</div>
      <div style={{ position:'absolute',bottom:'4px',right:'8px',zIndex:2,fontFamily:'var(--mono)',fontSize:'7px',color:'rgba(80,120,180,0.2)',letterSpacing:'1px' }}>P</div>

      <Canvas camera={{position:[0,0.5,2.8],fov:40}} gl={{antialias:true,alpha:true}} style={{background:'#030408'}}>
        <ambientLight intensity={0.06} />
        <directionalLight position={[2,3,2]} intensity={0.20} color="#8098b0" />
        <directionalLight position={[-2,1,-1]} intensity={0.08} color="#405060" />
        <pointLight position={[0,0.2,0]} intensity={0.10} color="#3098a8" distance={3} decay={2} />

        {modules.map((m,i) => <Node key={i} pos={POS[i]} color={REGION_COLORS[i]} act={m.activity_level} size={SZ[i]} idx={i} />)}
        <Tracts modules={modules} />

        <gridHelper args={[4,20,'#080c18','#060a14']} position={[0,-0.85,0]} />

        <OrbitControls enableZoom enablePan={false} autoRotate autoRotateSpeed={0.3}
          minDistance={1.8} maxDistance={5} maxPolarAngle={Math.PI*0.72} minPolarAngle={Math.PI*0.28} />
      </Canvas>
    </div>
  )
}
