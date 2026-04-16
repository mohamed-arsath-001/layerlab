import React, { useState } from 'react';
import Sidebar from './components/Sidebar';
import NeuralCanvas from './components/NeuralCanvas';
import SurgicalOverlay from './components/SurgicalOverlay';
import ModelDetailsPanel from './components/ModelDetailsPanel';
import { useMergeStore } from './store/useMergeStore';
import { Activity } from 'lucide-react';

function App() {
  const [activeLayer, setActiveLayer] = useState(null);
  const { progressPercent, connectionStatus, currentLayer } = useMergeStore();

  return (
    <div className="w-screen h-screen flex relative bg-cyber-dark overflow-hidden">
      
      {/* 3D Canvas Layer (Background/Main View) */}
      <div className="absolute inset-0 z-0">
        <NeuralCanvas activeLayer={activeLayer} setActiveLayer={setActiveLayer} />
      </div>

      {/* 2D UI Overlay */}
      <div className="z-10 flex w-full h-full pointer-events-none">
        
        {/* Sidebar Controls */}
        <div className="w-80 pointer-events-auto shrink-0 bg-cyber-gray/80 backdrop-blur-md border-r border-cyber-cyan/30 h-full p-6 overflow-y-auto flex flex-col gap-6 shadow-[4px_0_24px_rgba(0,243,255,0.1)]">
          <Sidebar />
        </div>

        {/* HUD: Stream Progress Overlay */}
        <div className="flex-1 relative">
          
          <div className="absolute top-6 left-1/2 -translate-x-1/2 bg-cyber-gray/80 backdrop-blur-md px-6 py-3 rounded-full border border-cyber-cyan/30 flex items-center gap-4 pointer-events-auto">
            <span className={`h-3 w-3 rounded-full animate-pulse ${
              connectionStatus === 'connected' ? 'bg-cyber-amber' : 
              connectionStatus === 'merging' ? 'bg-cyber-cyan' : 
              connectionStatus === 'error' ? 'bg-cyber-red' : 
              connectionStatus === 'done' ? 'bg-green-400' : 'bg-gray-500'
            }`}></span>
            <span className="font-mono text-sm tracking-widest text-gray-300 uppercase">
              STATUS: <span className="text-white font-bold">{connectionStatus}</span>
            </span>
          </div>

          {(connectionStatus === 'merging' || connectionStatus === 'done') && (
            <div className="absolute bottom-10 left-1/2 -translate-x-1/2 w-[600px] bg-cyber-gray/90 backdrop-blur-lg border border-cyber-cyan/40 p-4 rounded-xl pointer-events-auto shadow-[0_0_30px_rgba(0,243,255,0.15)] flex flex-col gap-3">
              <div className="flex justify-between items-end">
                <div className="text-cyber-cyan font-mono text-sm uppercase tracking-wider flex items-center gap-2">
                  <Activity size={16} className="animate-pulse" />
                  Merging Layer
                </div>
                <div className="font-mono text-xs text-cyber-amber bg-cyber-dark px-3 py-1 rounded text-right">
                  {currentLayer || 'INITIATING...'}
                </div>
              </div>
              
              <div className="h-3 w-full bg-cyber-dark rounded-full overflow-hidden relative">
                <div 
                  className="absolute top-0 left-0 h-full bg-gradient-to-r from-cyber-blue to-cyber-cyan shadow-[0_0_15px_#00f3ff] transition-all duration-300 ease-out"
                  style={{ width: `${progressPercent}%` }}
                ></div>
              </div>
              <div className="text-right text-xs font-mono text-cyber-cyan font-bold">{progressPercent.toFixed(1)}% COMPLETION</div>
            </div>
          )}

          {/* Right Side: Dissection Details Panel (Shows when no active layer is selected) */}
          <ModelDetailsPanel isHidden={!!activeLayer} />

          {/* Right Side: Surgical Overlay Panel (Shows when a distinct layer is clicked) */}
          <SurgicalOverlay activeLayer={activeLayer} onClose={() => setActiveLayer(null)} />
        </div>

      </div>
    </div>
  );
}

export default App;
