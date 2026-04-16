import os
import time
from engine import MergeConfig, stream_merge
from tensor_math import MergeAlgorithm
from pathlib import Path
import hub_utils
import asyncio

async def test_big_model():
    print("Pre-fetching models into cache...")
    a_path = hub_utils.resolve_path("hf://Qwen/Qwen2.5-0.5B/model.safetensors")
    b_path = hub_utils.resolve_path("hf://Qwen/Qwen2.5-0.5B-Instruct/model.safetensors")
    
    out_path = os.path.join(os.getcwd(), "output", "Qwen_Merged.safetensors")
    
    config = MergeConfig(
        path_a=a_path,
        path_b=b_path,
        output_path=out_path,
        algorithm=MergeAlgorithm("lerp"),
        global_alpha=0.5,
        path_base=None,
        per_layer_alpha={},
        trim_fraction=0.80,
        warn_threshold=0.70
    )
    
    print(f"Starting merge of massive 1GB models. \nA: {config.path_a} \nB: {config.path_b}")
    print(f"Outputting to {config.output_path}")
    
    start_time = time.time()
    
    async for evt in stream_merge(config):
        event_type = evt.get("event_type")
        if event_type == "layer_warning":
            print(f"WARNING [{evt.get('key')}]: {evt.get('message')}")
        elif event_type == "layer_done":
            pass # Suppress output so it doesn't flood 200+ tensors
        elif event_type == "merge_complete":
            print(f"MERGE COMPLETE! Output file saved to {evt.get('output_path')}")
            
    print(f"Total operation time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    asyncio.run(test_big_model())
