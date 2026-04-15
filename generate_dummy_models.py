import os
import torch
from safetensors.torch import save_file

def generate_dummy_model(filepath, num_layers=10, dim=256):
    print(f"Generating {filepath}...")
    tensors = {}
    
    # Embedding layer
    tensors["model.embed_tokens.weight"] = torch.randn((1000, dim))
    
    # Transformer layers
    for i in range(num_layers):
        tensors[f"model.layers.{i}.self_attn.q_proj.weight"] = torch.randn((dim, dim))
        tensors[f"model.layers.{i}.self_attn.k_proj.weight"] = torch.randn((dim, dim))
        tensors[f"model.layers.{i}.self_attn.v_proj.weight"] = torch.randn((dim, dim))
        tensors[f"model.layers.{i}.self_attn.o_proj.weight"] = torch.randn((dim, dim))
        tensors[f"model.layers.{i}.mlp.gate_proj.weight"] = torch.randn((dim * 4, dim))
        tensors[f"model.layers.{i}.mlp.up_proj.weight"] = torch.randn((dim * 4, dim))
        tensors[f"model.layers.{i}.mlp.down_proj.weight"] = torch.randn((dim, dim * 4))
        
        # Adding a bit of variance for Model B so similarity scores aren't identical
        if "B" in filepath:
            for k in list(tensors.keys()):
                if f"layers.{i}" in k:
                    # Randomly perturb some layers drastically to trigger red/yellow/cyan UI colors
                    if i % 3 == 0:
                        tensors[k] += torch.randn_like(tensors[k]) * 2.0  # High interference (Red)
                    elif i % 2 == 0:
                        tensors[k] += torch.randn_like(tensors[k]) * 0.5  # Mid interference (Amber)
                    else:
                        tensors[k] += torch.randn_like(tensors[k]) * 0.01 # Low interference (Cyan)

    # LM Head
    tensors["lm_head.weight"] = torch.randn((1000, dim))
    
    save_file(tensors, filepath)
    print(f"Saved {filepath} ({os.path.getsize(filepath) / 1e6:.2f} MB)")

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    generate_dummy_model("models/modelA.safetensors")
    generate_dummy_model("models/modelB.safetensors")
    print("Done! You can now use these paths in LayerLab.")
