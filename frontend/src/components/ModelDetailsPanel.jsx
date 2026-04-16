import React from 'react';
import { useMergeStore } from '../store/useMergeStore';
import { Database, Zap, CheckCircle } from 'lucide-react';

const getCapabilities = (blocks) => {
  if (!blocks) return [];
  const caps = new Set();
  blocks.forEach(b => {
     const t = typeof b === 'string' ? b : (b.block_type || b.label || '');
     if (t.includes('attention') || t.includes('attn')) caps.add('Context Focus & Cross-Reference');
     if (t.includes('mlp')) caps.add('Concept Recall & Pattern Matching');
     if (t.includes('norm')) caps.add('Signal Stabilization');
     if (t.includes('embed')) caps.add('Token Mapping & Embeddings');
     if (t.includes('head') || t.includes('lm_head')) caps.add('Vocal Projection (Logits)');
  });
  // Default fallbacks if no specific tags found
  if (caps.size === 0) {
    if (blocks.length > 0) {
       caps.add('Pattern Matching');
       caps.add('Feature Transformation');
    }
  }
  return Array.from(caps);
};

export default function ModelDetailsPanel({ isHidden }) {
  const { topologyA, topologyB, connectionStatus } = useMergeStore();

  if (!topologyA || isHidden) return null;

  const isComplete = connectionStatus === 'done';

  // Using inline styles for dynamic colors since generic color classes might be purged by Tailwind
  const renderTopology = (topo, title, colorHex) => {
     if (!topo) return null;
     const capabilities = getCapabilities(topo.blocks);
     const numLayers = topo.num_layers || topo.blocks?.length || 0;
     const totalParams = topo.total_params ? (topo.total_params / 1e6).toFixed(1) + 'M' : 'Unknown';

     return (
       <div className="p-4 bg-cyber-dark/50 rounded-lg flex flex-col gap-3 relative overflow-hidden" style={{ border: `1px solid ${colorHex}50` }}>
          <div className="absolute top-0 left-0 w-1 h-full" style={{ backgroundColor: colorHex }}></div>
          <h3 className="font-mono font-bold text-sm tracking-widest uppercase" style={{ color: colorHex }}>{title}</h3>
          
          <div className="grid grid-cols-2 gap-2 text-xs font-mono">
             <div className="text-gray-400">LAYERS</div>
             <div className="text-white text-right">{numLayers}</div>
             <div className="text-gray-400">PARAMS</div>
             <div className="text-white text-right">{totalParams}</div>
          </div>

          <div className="mt-2 text-left">
            <div className="text-[10px] text-gray-500 font-mono tracking-widest uppercase mb-1">Functional Capabilities</div>
            <div className="flex flex-col gap-1">
              {capabilities.map(cap => (
                 <div key={cap} className="flex items-center gap-2 text-gray-300 text-[10px]">
                   <Zap size={10} style={{ color: colorHex }} />
                   <span className="font-mono">{cap}</span>
                 </div>
              ))}
            </div>
          </div>
       </div>
     );
  };

  return (
    <div className={`absolute top-0 right-0 h-full w-[450px] bg-cyber-dark/95 backdrop-blur-xl border-l border-cyber-cyan/30 shadow-[-10px_0_30px_rgba(0,243,255,0.05)] p-6 z-10 pointer-events-auto flex flex-col overflow-y-auto transition-transform duration-300 translate-x-0`}>
      <div className="flex items-center gap-3 border-b border-cyber-gray pb-4">
        <Database className={isComplete ? "text-green-400" : "text-cyber-cyan"} size={20} />
        <h2 className="text-lg font-bold text-white font-mono uppercase tracking-widest">
          {isComplete ? 'OUTPUT METADATA' : 'DISSECTION METADATA'}
        </h2>
      </div>

      <div className="mt-6 flex flex-col gap-6">
        {!isComplete ? (
          <>
            <div className="text-xs text-gray-400 font-mono text-center mb-[-10px]">PRE-MERGE ANALYSIS</div>
            {renderTopology(topologyA, "Model A: Subject", "#00f3ff")}
            {renderTopology(topologyB, "Model B: Donor", "#ffb000")}
          </>
        ) : (
          <>
            <div className="p-4 border border-green-500/50 bg-green-500/10 rounded border-dashed text-center">
               <CheckCircle className="mx-auto text-green-400 mb-2" size={24} />
               <div className="text-green-400 font-mono text-xs uppercase tracking-widest pt-2">MERGE SUCCESSFUL</div>
            </div>
            
            {renderTopology(topologyA, "Synthesized Neural Matrix", "#4ade80")}
            <div className="text-xs text-gray-400 italic text-center mt-2 font-mono">
               Model is ready for inference block extraction or weight packaging.
            </div>
          </>
        )}
      </div>
    </div>
  );
}
