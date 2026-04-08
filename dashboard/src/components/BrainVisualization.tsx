import { Canvas, useFrame } from '@react-three/fiber'
import { OrbitControls } from '@react-three/drei'
import { useRef, useMemo } from 'react'
import * as THREE from 'three'
import type { ModuleSnapshot } from '../lib/protocol'
import { MODULE_COLOR_ARRAY } from '../lib/theme'

interface BrainVisualizationProps { modules: ModuleSnapshot[] }

const POS: [number,number,number][] = [
  [-0.85,0.55,0.15], [0,0.15,0.45], [0.65,0.8,-0.1],
  [-0.35,-0.2,0.5], [0.85,0.05,0], [0,-0.65,-0.15],
]
const SZ = [0.20,0.26,0.18,0.14,0.22,0.12]

function Node({ pos, color, act, size, idx }: {
  pos:[number,number,number]; color:string; act:number; size:number; idx:number
}) {
  const coreRef = useRef<THREE.Mesh>(null)
  const haloRef = useRef<THREE.Mesh>(null)
  const col = useMemo(() => new THREE.Color(color), [color])
  const dim = useMemo(() => new THREE.Color(color).multiplyScalar(0.22), [color])

  useFrame(({ clock }) => {
    const t = clock.elapsedTime
    if (coreRef.current) {
      const p = 1 + act * 0.10 * (Math.sin(t*1.6+idx)*0.7 + Math.sin(t*2.9+idx*0.5)*0.3)
      coreRef.current.scale.setScalar(size * p)
      ;(coreRef.current.material as THREE.MeshStandardMaterial).emissiveIntensity = 0.08 + act * 0.85
    }
    if (haloRef.current) {
      haloRef.current.scale.setScalar(size * (1.8 + act * 0.4))
      ;(haloRef.current.material as THREE.MeshBasicMaterial).opacity = act * 0.05
    }
  })

  return (
    <group position={pos}>
      <mesh ref={haloRef}>
        <sphereGeometry args={[1,20,20]} />
        <meshBasicMaterial color={col} transparent opacity={0.03} depthWrite={false} />
      </mesh>
      <mesh ref={coreRef}>
        <sphereGeometry args={[1,40,40]} />
        <meshStandardMaterial color={dim} emissive={col} emissiveIntensity={0.2}
          roughness={0.5} metalness={0.05} transparent opacity={0.9} />
      </mesh>
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
        <lineBasicMaterial color={f===5||t===5?'#e04848':'#283848'} transparent opacity={0.03+act*0.10} />
      </line>
    })}
  </>
}

export default function BrainVisualization({ modules }: BrainVisualizationProps) {
  return (
    <div style={{ height:'calc(100% - 22px)', position:'relative' }}>
      <Canvas camera={{position:[0,0.6,3.0],fov:38}} gl={{antialias:true,alpha:true}} style={{background:'#030408'}}>
        <ambientLight intensity={0.06} />
        <directionalLight position={[2,3,2]} intensity={0.2} color="#a0b8d0" />
        <directionalLight position={[-2,1,-1]} intensity={0.06} color="#405060" />
        <pointLight position={[0,0.2,0]} intensity={0.08} color="#38d8e8" distance={2.5} decay={2} />
        {modules.map((m,i) => <Node key={i} pos={POS[i]} color={MODULE_COLOR_ARRAY[i]} act={m.activity_level} size={SZ[i]} idx={i} />)}
        <Tracts modules={modules} />
        <gridHelper args={[3.5,18,'#080c18','#060a14']} position={[0,-0.9,0]} />
        <OrbitControls enableZoom enablePan={false} autoRotate autoRotateSpeed={0.2}
          minDistance={1.8} maxDistance={5} maxPolarAngle={Math.PI*0.72} minPolarAngle={Math.PI*0.28} />
      </Canvas>
    </div>
  )
}
