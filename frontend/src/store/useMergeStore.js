import { create } from 'zustand';

export const useMergeStore = create((set) => ({
  modelA: 'models/model.safetensors',
  modelB: 'models/model.safetensors',
  baseModel: '',
  outputPath: 'output/merged.safetensors',
  
  setModelA: (path) => set({ modelA: path }),
  setModelB: (path) => set({ modelB: path }),
  setBaseModel: (path) => set({ baseModel: path }),
  setOutputPath: (path) => set({ outputPath: path }),

  algorithm: 'lerp', 
  setAlgorithm: (algo) => set({ algorithm: algo }),
  globalAlpha: 0.5,
  setGlobalAlpha: (val) => set({ globalAlpha: val }),
  
  perLayerAlphas: {}, 
  setLayerAlpha: (layerName, value) => set((state) => ({
    perLayerAlphas: { ...state.perLayerAlphas, [layerName]: value }
  })),

  topology: null,
  topologyA: null,
  topologyB: null,
  mergedTopology: null,
  setTopology: (topo) => set({ topology: topo }),
  setTopologyA: (topo) => set({ topologyA: topo }),
  setTopologyB: (topo) => set({ topologyB: topo }),
  setMergedTopology: (topo) => set({ mergedTopology: topo }),

  progressPercent: 0,
  currentLayer: '',
  peakRam: 0,
  layerScores: [], 
  connectionStatus: 'disconnected', 
  
  setSocketStatus: (status) => set({ connectionStatus: status }),
  
  updateStreamMetrics: (data) => set((state) => {
    // FIXED: Mapping engine.py keys (key, cosine_sim, peak_ram_mb)
    let newScores = [...state.layerScores];
    
    if (data.key && data.cosine_sim !== undefined) {
      const newScore = {
        layerName: data.key,
        similarity: data.cosine_sim,
        warning: data.cosine_sim < 0.70
      };
      const existingIndex = state.layerScores.findIndex(s => s.layerName === data.key);
      if (existingIndex >= 0) newScores[existingIndex] = newScore;
      else newScores.push(newScore);
    }
    
    return {
      progressPercent: data.progress !== undefined ? data.progress : state.progressPercent,
      currentLayer: data.key || state.currentLayer,
      peakRam: data.peak_ram_mb !== undefined ? data.peak_ram_mb : state.peakRam,
      layerScores: newScores
    };
  }),

  resetMetrics: () => set({
    progressPercent: 0,
    currentLayer: '',
    peakRam: 0,
    layerScores: [],
    connectionStatus: 'disconnected'
  })
}));
