import React, { useRef, useMemo } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Environment, Sphere, Cylinder } from '@react-three/drei';
import { EffectComposer, Bloom } from '@react-three/postprocessing';
import * as THREE from 'three';
import { useMergeStore } from '../store/useMergeStore';

// Individual Node Component
const NeuralNode = ({ name, position, index, setActiveLayer }) => {
  const meshRef = useRef();
  
  // Read specific layer score from Zustand store
  const layerScores = useMergeStore(state => state.layerScores);
  const scoreData = layerScores.find(l => l.layerName === name);
  const similarity = scoreData ? scoreData.similarity : null;

  // Determine color based on similarity score
  const targetColor = useMemo(() => {
    if (similarity === null) return new THREE.Color('#0055ff'); // Default Cyber Blue
    if (similarity > 0.90) return new THREE.Color('#00f3ff'); // High Sim -> Cyan
    if (similarity < 0.70) return new THREE.Color('#ff003c'); // Low Sim (Warning) -> Red
    return new THREE.Color('#ffb000'); // Mid Sim -> Amber
  }, [similarity]);

  useFrame((state) => {
    if (meshRef.current) {
      // Smooth color transition
      meshRef.current.material.color.lerp(targetColor, 0.1);
      // Optional subtle hover rotation or pulse
      meshRef.current.rotation.x = state.clock.elapsedTime * 0.5 + index;
      meshRef.current.rotation.y = state.clock.elapsedTime * 0.3 + index;
    }
  });

  return (
    <group position={position}>
      <mesh 
        ref={meshRef}
        onClick={(e) => {
          e.stopPropagation();
          setActiveLayer(name);
        }}
        onPointerOver={() => document.body.style.cursor = 'pointer'}
        onPointerOut={() => document.body.style.cursor = 'auto'}
      >
        <boxGeometry args={[1.5, 0.4, 1.5]} />
        <meshStandardMaterial 
          color="#0055ff"
          emissive="#0055ff"
          emissiveIntensity={similarity === null ? 0.5 : (similarity < 0.7 ? 2.0 : 1.5)} 
          toneMapped={false}
          metalness={0.8}
          roughness={0.2}
        />
      </mesh>
      
      {/* Node connecting energy conduit (except for very top) */}
      <Cylinder 
        args={[0.05, 0.05, 2]} 
        position={[0, 1.2, 0]} 
      >
        <meshBasicMaterial color="#00f3ff" transparent opacity={0.2} />
      </Cylinder>
    </group>
  );
};

export default function NeuralCanvas({ activeLayer, setActiveLayer }) {
  const topology = useMergeStore(state => state.topology);
  
  // Parse blocks
  const blocks = topology?.blocks || [];

  return (
    <Canvas camera={{ position: [0, 5, 25], fov: 60 }} gl={{ antialias: false }}>
      <color attach="background" args={['#050508']} />
      
      <ambientLight intensity={0.2} />
      <pointLight position={[10, 10, 10]} intensity={1} color="#00f3ff" />
      <pointLight position={[-10, -10, -10]} intensity={0.5} color="#ff003c" />

      <OrbitControls 
        enablePan={true} 
        enableZoom={true} 
        enableRotate={true}
        maxPolarAngle={Math.PI / 1.5}
      />

      <group position={[0, - (blocks.length), 0]}>
        {blocks.map((block, i) => {
          const name = typeof block === 'string' ? block : block.name || `layer_${i}`;
          // Stack them vertically with gap of 2 units
          return (
            <NeuralNode 
              key={name}
              name={name} 
              index={i}
              position={[0, i * 2.4, 0]} 
              setActiveLayer={setActiveLayer} 
            />
          );
        })}
      </group>

      <EffectComposer disableNormalPass>
        <Bloom 
          luminanceThreshold={0.15} 
          luminanceSmoothing={0.9} 
          intensity={1.5} 
          mipmapBlur 
        />
      </EffectComposer>
    </Canvas>
  );
}
