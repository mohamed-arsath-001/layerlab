import React from 'react';
import { useMergeStore } from '../store/useMergeStore';
import { X, Sliders } from 'lucide-react';

export default function SurgicalOverlay({ activeLayer, onClose }) {
  const { globalAlpha, perLayerAlphas, setLayerAlpha, layerScores } = useMergeStore();

  if (!activeLayer) return null;

  const currentAlpha = perLayerAlphas[activeLayer] !== undefined 
    ? perLayerAlphas[activeLayer] 
    : globalAlpha;

  const scoreData = layerScores.find(l => l.layerName === activeLayer);
  const similarity = scoreData ? scoreData.similarity : null;
  const isWarning = similarity !== null && similarity < 0.70;

  return (
    <div className="absolute top-0 right-0 h-full w-96 bg-cyber-dark/95 backdrop-blur-xl border-l border-cyber-cyan/30 shadow-[-10px_0_30px_rgba(0,243,255,0.05)] p-6 z-20 pointer-events-auto flex flex-col transform transition-transform duration-300 translate-x-0">
      
      <div className="flex justify-between items-start border-b border-cyber-gray pb-4">
        <div>
          <h2 className="text-xl font-bold text-white font-mono uppercase tracking-widest break-all">
            {activeLayer}
          </h2>
          <div className="text-xs text-cyber-cyan font-mono mt-1">Surgical Modification Panel</div>
        </div>
        <button onClick={onClose} className="text-gray-400 hover:text-white transition-colors">
          <X size={24} />
        </button>
      </div>

      <div className="mt-8 flex flex-col gap-6">
        {/* Interference Stats */}
        <div className="bg-cyber-gray/30 border border-cyber-gray rounded p-4 flex flex-col gap-2">
           <div className="text-xs font-mono text-gray-400 uppercase">Cosine Similarity</div>
           <div className={`text-3xl font-mono font-bold ${isWarning ? 'text-cyber-red text-glow-red' : similarity !== null ? 'text-cyber-cyan text-glow-cyan' : 'text-gray-500'}`}>
             {similarity !== null ? similarity.toFixed(4) : '---'}
           </div>
           {isWarning && (
             <div className="text-xs text-cyber-red font-mono animate-pulse mt-1 bg-cyber-red/10 p-2 rounded">
               CRITICAL INTERFERENCE DETECTED. SURGERY RECOMMENDED.
             </div>
           )}
        </div>

        {/* Alpha Override Control */}
        <div className="bg-cyber-gray/30 border border-cyber-gray rounded p-4 flex flex-col gap-4">
          <div className="flex items-center gap-2 text-cyber-amber font-mono text-sm uppercase">
             <Sliders size={16} />
             Layer Alpha Override
          </div>
          
          <div className="flex justify-between">
            <span className="text-gray-400 text-xs font-mono">Model A Base</span>
            <span className="text-white text-lg font-mono">{currentAlpha.toFixed(2)}</span>
            <span className="text-gray-400 text-xs font-mono">Model B Inject</span>
          </div>
          
          <input 
            type="range" min="0" max="1" step="0.05"
            value={currentAlpha} 
            onChange={(e) => setLayerAlpha(activeLayer, parseFloat(e.target.value))}
            className="w-full accent-cyber-amber"
          />
          
          <div className="text-xs text-gray-500 font-mono italic text-center mt-2">
            Overrides global alpha for this block only.
          </div>
        </div>
        
        <div className="mt-auto pt-4">
           <button 
             onClick={() => setLayerAlpha(activeLayer, undefined)}
             className="w-full border border-cyber-gray hover:border-white text-gray-400 hover:text-white py-2 rounded font-mono text-xs uppercase tracking-widest transition-colors"
           >
             Reset to Global ({globalAlpha.toFixed(2)})
           </button>
        </div>
      </div>
    </div>
  );
}
