"""JIT loader for the causal RACE CUDA extension.

Compiles forward_kernel.cu + backward_kernels.cu + race_cuda_binding.cpp into a
single extension named ``race_cuda`` and returns it. The compile happens on first
call and is cached by torch (set TORCH_EXTENSIONS_DIR to control where).

Must be called where ``nvcc`` is available (e.g. after ``module load CUDA/12.x``
inside a GPU job). H200 = sm_90.
"""
import os
from torch.utils.cpp_extension import load

_HERE = os.path.dirname(os.path.abspath(__file__))
_ext = None


def load_ext(verbose=True):
    global _ext
    if _ext is None:
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0")  # H200 / Hopper
        _ext = load(
            name="race_cuda",
            sources=[
                os.path.join(_HERE, "forward_kernel.cu"),
                os.path.join(_HERE, "backward_kernels.cu"),
                os.path.join(_HERE, "race_cuda_binding.cpp"),
            ],
            extra_cflags=["-O3"],
            # No --use_fast_math: keep 1/(A+eps) IEEE for exact gradients.
            extra_cuda_cflags=["-O3", "-gencode=arch=compute_90,code=sm_90"],
            verbose=verbose,
        )
    return _ext


if __name__ == "__main__":
    ext = load_ext(verbose=True)
    print("race_cuda built OK; symbols:", [s for s in dir(ext) if not s.startswith("__")])
