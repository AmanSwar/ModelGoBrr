import torch
import torch.nn as nn

def model_mem_size(model : nn.Module , input_dtype = torch.float32):

    total_params = 0 
    total_grads = 0 
    
    for param in model.parameters():
        param_size = param.numel()
        total_params += param_size

        if param.requires_grad:
            total_grads += param_size


    total_buffers = sum(buf.numel() for buf in model.buffers())

    element_size = torch.tensor(0 , dtype=input_dtype).element_size()

    total_mem_in_bytes = (total_params + total_grads + total_buffers) * element_size

    total_mem_in_gb = total_mem_in_bytes / (1024**3)

    return total_mem_in_gb

