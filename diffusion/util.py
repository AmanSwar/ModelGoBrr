import torch

from PIL import Image


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def get_total_params(model : torch.nn.Module):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params


def get_gpu_memory():
    current_memory = torch.cuda.memory_allocated() / 1024**2
    print(f"Current GPU memory allocated by PyTorch: {current_memory:.2f} MB")
    peak_memory = torch.cuda.max_memory_allocated() / 1024**2
    print(f"Peak GPU memory allocated by PyTorch: {peak_memory:.2f} MB")

