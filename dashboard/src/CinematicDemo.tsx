import { Canvas, useFrame } from '@react-three/fiber'
import { OrbitControls, Text } from '@react-three/drei'
import { useRef, useMemo, useState, useEffect } from 'react'
import * as THREE from 'three'
import LiveAgent from './components/LiveAgent'

// Muted region colors matching main dashboard
const REGIONS = [
  { abbr: 'S1', pos: [-1.0, 0.6, 0.2] as [number,number,number], color: '#3098a8', size: 0.28 },
  { abbr: 'ASC', pos: [0.0, 0.25, 0.45] as [number,number,number], color: '#4878b0', size: 0.36 },
  { abbr: 'PE', pos: [0.75, 0.85, -0.1] as [number,number,number], color: '#907858', size: 0.26 },
  { abbr: 'HPC', pos: [-0.45, -0.15, 0.55] as [number,number,number], color: '#388898', size: 0.22 },
  { abbr: 'M1', pos: [1.0, 0.05, 0.0] as [number,number,number], color: '#408878', size: 0.30 },
  { abbr: 'BST', pos: [0.0, -0.7, -0.1] as [number,number,number], color: '#706070', size: 0.18 },
]

const CONNS = [[0,1],[0,2],[1,2],[1,4],[2,4],[3,1],[5,4],[3,0]]

function Particle({ from, to, color, speed, delay }: {
  from: [number,number,number]; to: [number,number,number]; color: string; speed: number; delay: number
}) {
  const ref = useRef<THREE.Mesh>(null)
  useFrame(({ clock }) => {
    if (!ref.current) return
    const t = clock.elapsedTime - delay
    if (t < 0) { ref.current.visible = false; return }
    ref.current.visible = true
    const p = (t * speed) % 1
    const p1 = new THREE.Vector3(...from), p2 = new THREE.Vector3(...to)
    const mid = p1.clone().add(p2).multiplyScalar(0.5); mid.y += 0.2
    const point = new THREE.QuadraticBezierCurve3(p1, mid, p2).getPoint(p)
    ref.current.position.copy(point)
    ref.current.scale.setScalar(0.03 + Math.sin(p * Math.PI) * 0.04)
    ;(ref.current.material as THREE.MeshBasicMaterial).opacity = 0.3 + Math.sin(p * Math.PI) * 0.7
  })
  return <mesh ref={ref} visible={false}>
    <sphereGeometry args={[1,8,8]} />
    <meshBasicMaterial color={color} transparent depthWrite={false} />
  </mesh>
}

function Node({ region, idx }: { region: typeof REGIONS[0]; idx: number }) {
  const coreRef = useRef<THREE.Mesh>(null)
  const haloRef = useRef<THREE.Mesh>(null)
  const ringRef = useRef<THREE.Mesh>(null)
  const col = useMemo(() => new THREE.Color(region.color), [region.color])

  useFrame(({ clock }) => {
    const t = clock.elapsedTime
    const act = 0.3 + 0.7 * (0.5 + 0.5 * Math.sin(t * 1.5 + idx * 1.3))
    if (coreRef.current) {
      coreRef.current.scale.setScalar(region.size * (1 + act * 0.08 * Math.sin(t * 1.8 + idx)))
      ;(coreRef.current.material as THREE.MeshStandardMaterial).emissiveIntensity = 0.2 + act * 0.7
    }
    if (haloRef.current) {
      haloRef.current.scale.setScalar(region.size * (2.0 + act * 0.4))
      ;(haloRef.current.material as THREE.MeshBasicMaterial).opacity = act * 0.05
    }
    if (ringRef.current) {
      ringRef.current.rotation.z = t * 0.2 + idx
      ringRef.current.scale.setScalar(region.size * (1.5 + act * 0.3))
      ;(ringRef.current.material as THREE.MeshBasicMaterial).opacity = 0.04 + act * 0.08
    }
  })

  return (
    <group position={region.pos}>
      <mesh ref={haloRef}><sphereGeometry args={[1,20,20]} /><meshBasicMaterial color={col} transparent opacity={0.04} depthWrite={false} /></mesh>
      <mesh ref={ringRef} rotation={[Math.PI/2,0,0]}><ringGeometry args={[0.85,1.0,32]} /><meshBasicMaterial color={col} transparent opacity={0.06} side={THREE.DoubleSide} depthWrite={false} /></mesh>
      <mesh ref={coreRef}><sphereGeometry args={[1,48,48]} /><meshStandardMaterial color={new THREE.Color(region.color).multiplyScalar(0.15)} emissive={col} emissiveIntensity={0.4} roughness={0.5} metalness={0.1} transparent opacity={0.85} /></mesh>
      <Text position={[0,-region.size-0.07,0]} fontSize={0.055} color={region.color} anchorX="center" anchorY="top" fillOpacity={0.5}>{region.abbr}</Text>
    </group>
  )
}

function Tracts() {
  return <>{CONNS.map(([f,t],i) => {
    const p1 = new THREE.Vector3(...REGIONS[f].pos), p2 = new THREE.Vector3(...REGIONS[t].pos)
    const mid = p1.clone().add(p2).multiplyScalar(0.5); mid.y += 0.12
    const geo = new THREE.BufferGeometry().setFromPoints(new THREE.QuadraticBezierCurve3(p1, mid, p2).getPoints(16))
    return <line key={i} geometry={geo}><lineBasicMaterial color={f===5||t===5?'#604050':'#283848'} transparent opacity={0.1} /></line>
  })}</>
}

function Particles() {
  const ps = useMemo(() => {
    const arr: { from: [number,number,number]; to: [number,number,number]; color: string; speed: number; delay: number }[] = []
    for (let w = 0; w < 3; w++) for (const [f,t] of CONNS) {
      arr.push({ from: REGIONS[f].pos, to: REGIONS[t].pos, color: REGIONS[f].color, speed: 0.25 + Math.random()*0.3, delay: w*3 + Math.random()*2 })
    }
    return arr
  }, [])
  return <>{ps.map((p,i) => <Particle key={i} {...p} />)}</>
}

export default function CinematicDemo() {
  const [tick, setTick] = useState(0)
  useEffect(() => { const i = setInterval(() => setTick(t => t+10), 33); return () => clearInterval(i) }, [])

  const t = tick * 0.001
  const hz = (950 + Math.sin(t) * 50).toFixed(0)
  const spk = (tick * 27).toLocaleString()
  const pe = (0.3 + 0.2 * Math.sin(t * 0.5)).toFixed(3)

  return (
    <div style={{ width:'100vw', height:'100vh', background:'#020205', position:'relative', overflow:'hidden' }}>
      {/* CRT scanlines */}
      <div style={{ position:'fixed', inset:0, pointerEvents:'none', zIndex:10,
        background:'repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(0,0,0,0.03) 2px, rgba(0,0,0,0.03) 4px)',
      }}>
        <div style={{ position:'absolute', inset:0, background:'radial-gradient(ellipse at center, transparent 50%, rgba(0,0,0,0.35) 100%)' }} />
      </div>

      {/* Top bar */}
      <div style={{
        position:'fixed', top:0, left:0, right:0, zIndex:20, height:'36px',
        display:'flex', alignItems:'center', padding:'0 16px', gap:'16px',
        background:'linear-gradient(180deg, rgba(6,8,16,0.95), rgba(6,8,16,0.7))',
        borderBottom:'1px solid rgba(48,152,168,0.08)',
        fontFamily:"'JetBrains Mono', monospace",
      }}>
        <div style={{ display:'flex', alignItems:'center', gap:'8px' }}>
          <div style={{ width:'5px', height:'5px', borderRadius:'1px', background:'#3098a8', boxShadow:'0 0 8px #3098a8', animation:'pulse 2s ease-in-out infinite' }} />
          <span style={{ fontSize:'12px', fontWeight:700, letterSpacing:'3px', color:'#3098a8' }}>ENGRAM</span>
          <span style={{ fontSize:'8px', color:'#2a3450', letterSpacing:'1.5px', marginLeft:'4px' }}>NEURAL COGNITIVE SYSTEM</span>
        </div>
        <div style={{ width:'1px', height:'18px', background:'rgba(48,152,168,0.1)' }} />
        <M label="Hz" value={hz} color="#508870" />
        <M label="SPK" value={spk} color="#5098a8" />
        <M label="PE" value={pe} color="#907858" />
        <div style={{ flex:1 }} />
        <span style={{ fontSize:'9px', color:'#3098a8', letterSpacing:'1.5px' }}>LIVE</span>
        <div style={{ width:'4px', height:'4px', borderRadius:'50%', background:'#508870', boxShadow:'0 0 6px #508870' }} />
      </div>

      {/* 3D Brain -- top 55% */}
      <div style={{ position:'absolute', top:'36px', left:0, right:0, height:'55%', zIndex:1 }}>
        <Canvas camera={{position:[0,0.6,3.0],fov:38}} gl={{antialias:true,alpha:true}} style={{background:'#020205'}}>
          <ambientLight intensity={0.06} />
          <directionalLight position={[2,3,2]} intensity={0.25} color="#8098b0" />
          <directionalLight position={[-2,1,-1]} intensity={0.08} color="#405060" />
          <pointLight position={[0,0.2,0]} intensity={0.12} color="#3098a8" distance={3} decay={2} />
          {REGIONS.map((r,i) => <Node key={i} region={r} idx={i} />)}
          <Tracts />
          <Particles />
          <gridHelper args={[4,20,'#080c18','#060a14']} position={[0,-0.85,0]} />
          <OrbitControls enableZoom={false} enablePan={false} autoRotate autoRotateSpeed={0.3}
            minPolarAngle={Math.PI*0.3} maxPolarAngle={Math.PI*0.65} />
        </Canvas>
      </div>

      {/* Live Learning Agent -- bottom 40% */}
      <div style={{
        position:'absolute', bottom:'24px', left:'16px', right:'16px', height:'38%', zIndex:20,
        background:'rgba(6,8,16,0.9)', border:'1px solid rgba(48,152,168,0.08)', borderRadius:'4px',
      }}>
        <div style={{
          height:'22px', display:'flex', alignItems:'center', padding:'0 10px', gap:'6px',
          borderBottom:'1px solid rgba(48,80,120,0.06)',
          fontFamily:"'JetBrains Mono', monospace",
        }}>
          <div style={{ width:'4px', height:'4px', borderRadius:'50%', background:'#508870', boxShadow:'0 0 4px #508870' }} />
          <span style={{ fontSize:'8px', fontWeight:500, letterSpacing:'1.5px', color:'#4a6080', textTransform:'uppercase' }}>
            Live Learning -- Random Maze Generalization
          </span>
          <span style={{ marginLeft:'auto', fontSize:'7px', padding:'1px 5px', borderRadius:'2px', background:'rgba(48,152,168,0.05)', color:'#3098a8', border:'1px solid rgba(48,152,168,0.1)', letterSpacing:'0.5px' }}>
            Q-LEARN
          </span>
        </div>
        <LiveAgent />
      </div>

      {/* Bottom status */}
      <div style={{
        position:'fixed', bottom:0, left:0, right:0, zIndex:20, height:'24px',
        display:'flex', alignItems:'center', padding:'0 16px', gap:'20px',
        background:'linear-gradient(0deg, rgba(6,8,16,0.95), rgba(6,8,16,0.7))',
        borderTop:'1px solid rgba(48,152,168,0.06)',
        fontFamily:"'JetBrains Mono', monospace", fontSize:'8px', color:'#2a3450', letterSpacing:'1.5px',
      }}>
        <span>ENGRAM NCS v0.1.0</span>
        <span style={{ color:'#1a2038' }}>|</span>
        <span>672 NEURONS</span>
        <span style={{ color:'#1a2038' }}>|</span>
        <span>6 REGIONS</span>
        <span style={{ color:'#1a2038' }}>|</span>
        <span>FEATURE-BASED GENERALIZATION</span>
        <div style={{ flex:1 }} />
        <span>55.3% MAZE SUCCESS vs 41.3% Q-LEARNING BASELINE</span>
        <div style={{ width:'3px', height:'3px', borderRadius:'50%', background:'#3098a8', boxShadow:'0 0 6px #3098a8', animation:'pulse 3s ease-in-out infinite' }} />
      </div>

      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&display=swap');
        @keyframes pulse { 0%,100%{opacity:0.4}50%{opacity:1} }
        * { margin:0; padding:0; box-sizing:border-box; }
        body { overflow:hidden; background:#020205; }
      `}</style>
    </div>
  )
}

function M({ label, value, color }: { label:string; value:string; color:string }) {
  return <div style={{ display:'flex', alignItems:'baseline', gap:'4px', fontFamily:"'JetBrains Mono', monospace" }}>
    <span style={{ fontSize:'7px', color:'#2a3450', letterSpacing:'1px' }}>{label}</span>
    <span style={{ fontSize:'11px', color, fontWeight:500 }}>{value}</span>
  </div>
}
