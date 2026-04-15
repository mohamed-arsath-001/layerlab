import React, { useState } from 'react';
import { useMergeStore } from '../store/useMergeStore';
import { useMergeSocket } from '../hooks/useMergeSocket';
import { Play, Database, ShieldAlert, Cpu } from 'lucide-react';

export default function Sidebar() {
  const {
    modelA, setModelA,
    modelB, setModelB,
    baseModel, setBaseModel,
    outputPath, setOutputPath,
    algorithm, setAlgorithm,
    globalAlpha, setGlobalAlpha,
    setTopology,
    connectionStatus
  } = useMergeStore();

  const { connectAndMerge, disconnect } = useMergeSocket();
  const [probing, setProbing] = useState(false);

  const handleProbe = async () => {
    setProbing(true);
    try {
      const res = await fetch('http://localhost:8000/api/probe', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path: modelA })
      });
      if (res.ok) {
        const data = await res.json();
        // Assuming backend returns { blocks: [...] } or an array of block names
        setTopology(data);
      } else {
        console.error("Probe failed:", await res.text());
        // For testing/fallback if backend is mocked/down
        setTopology({ blocks: Array.from({length: 32}, (_, i) => `model.layers.${i}`) });
      }
    } catch (err) {
      console.error(err);
      // Fallback for visualizer demo
      setTopology({ blocks: Array.from({length: 32}, (_, i) => `model.layers.${i}`) });
    }
    setProbing(false);
  };

  const isMerging = connectionStatus === 'merging' || connectionStatus === 'connecting';

  return (
    <div className="flex flex-col gap-6 text-sm">
      <div className="flex items-center gap-3 border-b border-cyber-cyan/30 pb-4">
        <Cpu className="text-cyber-cyan" />
        <h1 className="text-xl font-bold tracking-widest text-white">LAYERLAB</h1>
      </div>

      {/* Model Paths */}
      <div className="flex flex-col gap-3">
        <label className="text-cyber-cyan font-mono text-xs uppercase">Model A (Subject)</label>
        <input 
          className="bg-cyber-dark/50 border border-cyber-gray p-2 rounded text-gray-300 focus:outline-none focus:border-cyber-cyan font-mono text-xs"
          value={modelA} onChange={e => setModelA(e.target.value)}
        />

        <label className="text-cyber-cyan font-mono text-xs uppercase mt-2">Model B (Donor)</label>
        <input 
          className="bg-cyber-dark/50 border border-cyber-gray p-2 rounded text-gray-300 focus:outline-none focus:border-cyber-cyan font-mono text-xs"
          value={modelB} onChange={e => setModelB(e.target.value)}
        />

        <label className="text-cyber-cyan font-mono text-xs uppercase mt-2">Base Model (for TIES)</label>
        <input 
          className="bg-cyber-dark/50 border border-cyber-gray p-2 rounded text-gray-300 focus:outline-none focus:border-cyber-cyan font-mono text-xs"
          value={baseModel} onChange={e => setBaseModel(e.target.value)}
        />
        
        <button 
          onClick={handleProbe}
          disabled={probing || isMerging}
          className="mt-2 bg-cyber-dark border border-cyber-cyan py-2 rounded text-cyber-cyan hover:bg-cyber-cyan hover:text-black transition-colors flex items-center justify-center gap-2 font-mono uppercase tracking-widest disabled:opacity-50"
        >
          <Database size={16} />
          {probing ? 'Probing...' : 'Probe Topology'}
        </button>
      </div>

      {/* Configurations */}
      <div className="flex flex-col gap-3 mt-4 border-t border-cyber-cyan/30 pt-6">
        <label className="text-cyber-cyan font-mono text-xs uppercase">Algorithm</label>
        <select 
          className="bg-cyber-dark border border-cyber-gray p-2 rounded text-white focus:outline-none focus:border-cyber-cyan font-mono text-xs uppercase"
          value={algorithm} onChange={(e) => setAlgorithm(e.target.value)}
        >
          <option value="lerp">Linear (LERP)</option>
          <option value="slerp">Spherical (SLERP)</option>
          <option value="ties">TIES Merging</option>
        </select>

        <div className="flex justify-between mt-4">
          <label className="text-cyber-cyan font-mono text-xs uppercase">Global Alpha</label>
          <span className="text-white font-mono text-xs">{globalAlpha.toFixed(2)}</span>
        </div>
        <input 
          type="range" min="0" max="1" step="0.05"
          value={globalAlpha} onChange={e => setGlobalAlpha(parseFloat(e.target.value))}
          className="accent-cyber-cyan"
        />

        <label className="text-cyber-cyan font-mono text-xs uppercase mt-4">Output Path</label>
        <input 
          className="bg-cyber-dark/50 border border-cyber-gray p-2 rounded text-gray-300 focus:outline-none focus:border-cyber-cyan font-mono text-xs"
          value={outputPath} onChange={e => setOutputPath(e.target.value)}
        />
      </div>

      <div className="mt-auto pt-6 flex flex-col gap-2">
        {!isMerging ? (
           <button 
             onClick={connectAndMerge}
             className="bg-cyber-cyan hover:bg-cyber-blue transition-colors text-black font-bold py-3 rounded flex justify-center items-center gap-2 uppercase tracking-widest shadow-[0_0_15px_rgba(0,243,255,0.4)]"
           >
             <Play size={18} fill="currentColor" />
             Execute Merge
           </button>
        ) : (
           <button 
             onClick={disconnect}
             className="bg-cyber-red/20 border border-cyber-red hover:bg-cyber-red transition-colors text-white font-bold py-3 rounded flex justify-center items-center gap-2 uppercase tracking-widest shadow-[0_0_15px_rgba(255,0,60,0.4)]"
           >
             <ShieldAlert size={18} />
             Abort Surgery
           </button>
        )}
      </div>
    </div>
  );
}
