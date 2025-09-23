from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    # name="rmsnorm_kernel_vec",
    name="rmsnorm_kernel_vectorized",
    ext_modules=[
        CUDAExtension(
            # "rmsnorm_kernel_vec",
            "rmsnorm_kernel_vectorized",
            sources=["kernels/CUDA/csrc/rmsnnorm_binding.cu"],
            extra_compile_args={
                "cxx": [],
                "nvcc": [
                    "-O3",
                    "-gencode",
                    "arch=compute_80,code=sm_80",
                    "--expt-relaxed-constexpr",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
