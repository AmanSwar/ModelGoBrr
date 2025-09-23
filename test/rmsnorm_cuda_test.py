from rmsnorm_kernel_vectorized import rmsnorm_kernel_vectorized as rmsnorm_kernel

import torch.nn as nn
import torch

class RMSNorm_CUDA(nn.Module):
    def __init__(self ,dimension , eps=1e-6):
      super().__init__()
      self.weight = nn.Parameter(torch.rand(size=(dimension,) , dtype=torch.bfloat16))
      self.eps = eps

    def forward(self, x):
      out = rmsnorm_kernel(x , self.weight , self.eps)

      return out


class SimpleLinear(nn.Module):

  def __init__(self):

    super().__init__()
    self.in_dim = 1024
    self.hd = self.in_dim * 4

    self.linea1 = nn.Linear(in_features=self.in_dim , out_features=self.hd , dtype=torch.bfloat16)
    self.rmsnorm1 = RMSNorm_CUDA(self.hd)
    
    self.linea2 = nn.Linear(in_features=self.hd , out_features=self.in_dim , dtype=torch.bfloat16)
    self.rmnsnorm2 = RMSNorm_CUDA(self.in_dim)


  def forward(self, x):

    x = self.linea2(self.rmsnorm1(self.linea1(x)))
    x = self.rmnsnorm2(x)

    return x


class SmallModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.layers = nn.ModuleList(
      [SimpleLinear() for _ in range(10)]
    )

        self.final_layer = nn.Linear(1024 , 1024 , dtype=torch.bfloat16)
        self.rms = RMSNorm_CUDA(1024)

    def forward(self, x):

        for layer in self.layers:
            x = layer(x)

        x = self.rms(self.final_layer(x))

        return x


class SimpleLinearTorch(nn.Module):

    def __init__(self):

        super().__init__()
        self.in_dim = 1024
        self.hd = self.in_dim * 4

        self.linea1 = nn.Linear(
            in_features=self.in_dim, out_features=self.hd, dtype=torch.bfloat16
        )
        self.rmsnorm1 = nn.RMSNorm(self.hd , dtype=torch.bfloat16)

        self.linea2 = nn.Linear(
            in_features=self.hd, out_features=self.in_dim, dtype=torch.bfloat16
        )
        self.rmnsnorm2 = nn.RMSNorm(self.in_dim , dtype=torch.bfloat16)

    def forward(self, x):

        x = self.linea2(self.rmsnorm1(self.linea1(x)))
        x = self.rmnsnorm2(x)

        return x


class SmallModelTorch(nn.Module):

    def __init__(self):
        super().__init__()

        self.layers = nn.ModuleList([SimpleLinearTorch() for _ in range(10)])

        self.final_layer = nn.Linear(1024, 1024, dtype=torch.bfloat16)
        self.rms = nn.RMSNorm(1024 , dtype=torch.bfloat16)

    def forward(self, x):

        for layer in self.layers:
            x = layer(x)

        x = self.rms(self.final_layer(x))

        return x


# ==============================================================

import time

device = torch.device("cuda")
a = torch.rand(size=(1,128,1024) , dtype=torch.bfloat16).to(device)

model = SmallModel().to(device)
model_torch = SmallModelTorch().to(device)


st = time.monotonic()
for _ in range(100):
   _ = model_torch(a)


et = time.monotonic() - st

st1 = time.monotonic()
for _ in range(100):
   _ = model(a)

et1 = time.monotonic() - st1

print(f"Torch time : {et}")
print(f"Custom time : {et1}")



# ====== PROFILING ========
print("Profiling kernel")
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True,
    with_stack=True,
) as prof:
    with torch.profiler.record_function("CUDA model run"):
        for i in range(5):
            _ = model(a)
    with torch.profiler.record_function("PyTorchModel_Run"):
        for i in range(3):
            lm_ = model_torch(a)


print("saving")
prof.export_chrome_trace("test/rmsnorm_profile.json")
print("Profiling complete. Chrome trace saved to profile_trace.json")


    