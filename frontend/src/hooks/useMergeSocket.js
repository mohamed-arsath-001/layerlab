import { useEffect, useRef } from 'react';
import { useMergeStore } from '../store/useMergeStore';

export function useMergeSocket() {
  const wsRef = useRef(null);
  const {
    modelA, modelB, baseModel, outputPath, algorithm, globalAlpha, perLayerAlphas,
    setSocketStatus, updateStreamMetrics, resetMetrics
  } = useMergeStore();

  const connectAndMerge = () => {
    if (wsRef.current) wsRef.current.close();
    resetMetrics();

    const ws = new WebSocket('ws://localhost:8000/ws/merge');
    wsRef.current = ws;
    setSocketStatus('connecting');

    ws.onopen = () => {
      setSocketStatus('connected');
      // FIXED: Matching the exact Pydantic schema from api.py
      const payload = {
        path_a: modelA,
        path_b: modelB,
        path_base: baseModel || null,
        output_path: outputPath,
        algorithm: algorithm,
        global_alpha: globalAlpha,
        per_layer_alpha: perLayerAlphas
      };
      ws.send(JSON.stringify(payload));
      setSocketStatus('merging');
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        
        // FIXED: Matching event_types from engine.py
        if (data.event_type === 'merge_complete') {
           setSocketStatus('done');
           updateStreamMetrics(data);
           return;
        }
        if (data.event_type === 'error') {
           setSocketStatus('error');
           console.error("Backend Error:", data.error_msg);
           return;
        }
        updateStreamMetrics(data);
      } catch (err) {
        console.error("Failed to parse websocket message", err);
      }
    };

    ws.onerror = (error) => {
      console.error("WebSocket Error:", error);
      setSocketStatus('error');
    };

    ws.onclose = () => {
      const currentState = useMergeStore.getState().connectionStatus;
      if (['merging', 'connected', 'connecting'].includes(currentState)) {
        setSocketStatus('disconnected');
      }
    };
  };

  const disconnect = () => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
      setSocketStatus('disconnected');
    }
  };

  useEffect(() => {
    return () => { if (wsRef.current) wsRef.current.close(); };
  }, []);

  return { connectAndMerge, disconnect };
}
