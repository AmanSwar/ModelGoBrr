import json
import os
from pathlib import Path
from safetensors.torch import load_file
from huggingface_hub import snapshot_download

import torch

repo_id = "CompVis/stable-diffusion-v1-4"
local_dir = Path(repo_id).parts[-1]

repo_dir = snapshot_download(repo_id=repo_id, local_dir=local_dir)
device = torch.device("cuda")

