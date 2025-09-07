import json
import os
from pathlib import Path
from safetensors.torch import load_file
from huggingface_hub import snapshot_download

from qwen3.qwen_torch import Qwen3
from qwen3.config import QwenConfig
from qwen3.load import load_weights

repo_id = "Qwen/Qwen3-0.6B"
local_dir = Path(repo_id).parts[-1]

repo_dir = snapshot_download(repo_id=repo_id, local_dir=local_dir)

if __name__ == "__main__":
    import torch

    config = QwenConfig()
    model = Qwen3(config)
    device = torch.device('cuda')
    
    # Check if model is sharded or single file
    index_path = os.path.join(repo_dir, "model.safetensors.index.json")
    single_file_path = os.path.join(repo_dir, "model.safetensors")
    
    weights_dict = {}
    
    if os.path.exists(index_path):
        # Sharded model
        print("Loading sharded model...")
        with open(index_path, "r") as f:
            index = json.load(f)
        
        for filename in set(index["weight_map"].values()):
            shard_path = os.path.join(repo_dir, filename)
            shard = load_file(shard_path)
            weights_dict.update(shard)
    
    elif os.path.exists(single_file_path):
        # Single file model
        print("Loading single file model...")
        weights_dict = load_file(single_file_path)
    
    else:
        # Try to find any .safetensors files
        safetensors_files = list(Path(repo_dir).glob("*.safetensors"))
        if safetensors_files:
            print(f"Found safetensors files: {[f.name for f in safetensors_files]}")
            for file_path in safetensors_files:
                shard = load_file(str(file_path))
                weights_dict.update(shard)
        else:
            raise FileNotFoundError(f"No safetensors files found in {repo_dir}")

    print(f"Loaded {len(weights_dict)} weight tensors")
    # Add this before calling load_weights to debug:
    print("Debugging weight shapes:")
    print(f"q_proj shape: {weights_dict['model.layers.0.self_attn.q_proj.weight'].shape}")
    print(f"k_proj shape: {weights_dict['model.layers.0.self_attn.k_proj.weight'].shape}")
    print(f"v_proj shape: {weights_dict['model.layers.0.self_attn.v_proj.weight'].shape}")
    print(f"Expected embed_dim: {config.embed_dim}")
    print(f"Expected n_heads: {config.n_heads}")
    print(f"Expected n_kv_grps: {config.n_kv_heads}")
    load_weights(model, config, weights_dict)
    model.to(device)

    print("Model loaded successfully!")