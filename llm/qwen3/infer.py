import torch
import torch.nn as nn

from qwen3.qwen_torch import Qwen3
from qwen3.config import QwenConfig
from utils import model_mem_size
from qwen3.load import load_weights


from pathlib import Path
import os
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

torch.manual_seed(696969)
device = torch.device("cuda")


config = QwenConfig()
model = Qwen3(config).to(device)

#checking the model 
# print(model(torch.tensor([1,2,3]).unsqueeze(0).to(device=device)))

print(f"float32 (PyTorch default): {model_mem_size(model, input_dtype=torch.float32):.2f} GB")
print(f"bfloat16: {model_mem_size(model, input_dtype=torch.bfloat16):.2f} GB")

repo_dir = "/home/aman/code/model_go_brr/llm/Qwen3-0.6B"
single_file_path = os.path.join(repo_dir, "model.safetensors")
weights_dict = load_file(single_file_path)
load_weights(model, config, weights_dict)
model.to(device)
print("Model loaded successfully!")