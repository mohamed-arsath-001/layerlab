import React, { useRef, useMemo } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Environment, Sphere, Cylinder } from '@react-three/drei';
import { EffectComposer, Bloom } from '@react-three/postprocessing';
import * as THREE from 'three';
import { useMergeStore } from '../store/useMergeStore';

import { Html } from '@react-three/drei';

// Individual Node Component
const NeuralNode = ({ name, position, index, setActiveLayer, activeLayer }) => {
  const meshRef = useRef();
  const [hovered, setHovered] = React.useState(false);
  
  // Read specific layer score from Zustand store
  const layerScores = useMergeStore(state => state.layerScores);
  const scoreData = layerScores.find(l => l.layerName === name);
  const similarity = scoreData ? scoreData.similarity : null;
  const perLayerAlphas = useMergeStore(state => state.perLayerAlphas);
  const layerAlpha = perLayerAlphas[name];

  // Determine color based on similarity score
  const targetColor = useMemo(() => {
    if (hovered) return new THREE.Color('#ffffff'); // Highlight on hover
    if (similarity === null) return new THREE.Color('#0055ff'); // Default Cyber Blue
    if (similarity > 0.90) return new THREE.Color('#00f3ff'); // High Sim -> Cyan
    if (similarity < 0.70) return new THREE.Color('#ff003c'); // Low Sim (Warning) -> Red
    return new THREE.Color('#ffb000'); // Mid Sim -> Amber
  }, [similarity, hovered]);

  useFrame((state) => {
    if (meshRef.current) {
      // Smooth color transition
      meshRef.current.material.color.lerp(targetColor, 0.1);
      // Optional subtle hover rotation or pulse
      meshRef.current.rotation.x = state.clock.elapsedTime * 0.5 + index;
      meshRef.current.rotation.y = state.clock.elapsedTime * 0.3 + index;
      
      if (hovered) {
        meshRef.current.scale.lerp(new THREE.Vector3(1.2, 1.2, 1.2), 0.1);
      } else {
        meshRef.current.scale.lerp(new THREE.Vector3(1.0, 1.0, 1.0), 0.1);
      }
    }
  });

  const isActive = activeLayer === name;
  const showDetails = hovered || isActive || similarity !== null;

  return (
    <group position={position}>
      <mesh 
        ref={meshRef}
        onClick={(e) => {
          e.stopPropagation();
          setActiveLayer(name);
        }}
        onPointerOver={(e) => {
          e.stopPropagation();
          setHovered(true);
          document.body.style.cursor = 'pointer';
        }}
        onPointerOut={(e) => {
          e.stopPropagation();
          setHovered(false);
          document.body.style.cursor = 'auto';
        }}
      >
        <boxGeometry args={[1.5, 0.4, 1.5]} />
        <meshStandardMaterial 
          color="#0055ff"
          emissive="#0055ff"
          emissiveIntensity={similarity === null ? 0.5 : (similarity < 0.7 ? 2.0 : 1.5)} 
          toneMapped={false}
          metalness={0.8}
          roughness={0.2}
          wireframe={hovered}
        />
      </mesh>
      
      {/* Node connecting energy conduit (except for very top) */}
      <Cylinder 
        args={[0.05, 0.05, 2]} 
        position={[0, 1.2, 0]} 
      >
        <meshBasicMaterial color="#00f3ff" transparent opacity={0.2} />
      </Cylinder>

      {/* Layer Details Label via Html */}
      {showDetails && (
        <Html position={[1.5, 0, 0]} center distanceFactor={15} zIndexRange={[100, 0]}>
          <div className={`transition-all duration-300 pointer-events-none flex flex-col gap-1 rounded-md border backdrop-blur-md p-2 min-w-[140px]
            ${isActive ? 'border-cyber-cyan bg-cyber-dark/90 shadow-[0_0_15px_rgba(0,243,255,0.4)] scale-110 relative z-50' : 
              hovered ? 'border-white/50 bg-cyber-gray/80 scale-105 relative z-40' : 
              'border-cyber-cyan/20 bg-cyber-dark/60 opacity-80 z-10'}`}
          >
            <div className="text-[10px] font-mono text-gray-400 uppercase tracking-widest border-b border-gray-600/50 pb-1 mb-1">
              Layer ID
            </div>
            <div className={`font-mono text-xs truncate ${isActive ? 'text-white font-bold' : 'text-cyber-cyan'}`}>
              {name.split('.').pop() || name} {/* Show shortest name if possible */}
            </div>
            
            {similarity !== null ? (
              <div className="flex justify-between items-center mt-1">
                <span className="text-[9px] font-mono text-gray-500">SIMILARITY</span>
                <span className={`text-[11px] font-mono font-bold ${similarity < 0.7 ? 'text-cyber-red animate-pulse' : similarity > 0.9 ? 'text-cyber-cyan' : 'text-cyber-amber'}`}>
                  {(similarity * 100).toFixed(1)}%
                </span>
              </div>
            ) : (
              <div className="text-[9px] font-mono text-gray-500 italic mt-1">Awaiting Dissection...</div>
            )}

            {layerAlpha !== undefined && (
               <div className="flex justify-between items-center mt-1 border-t border-cyber-amber/30 pt-1">
                 <span className="text-[9px] font-mono text-cyber-amber">ALPHA OVERRIDE</span>
                 <span className="text-[10px] font-mono text-white">{layerAlpha.toFixed(2)}</span>
               </div>
            )}
          </div>
        </Html>
      )}
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
                activeLayer={activeLayer}
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
