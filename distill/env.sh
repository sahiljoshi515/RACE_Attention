#!/bin/bash
# Source on a GPU (H200) node before running the distillation scripts.
#   source distill/env.sh
# Uses race_vit_env (transformers 5.5.0, datasets 4.4.1, torch 2.10). The RACE CUDA
# extension is JIT-built here against torch 2.10 in a SEPARATE cache dir (the
# swa_env build is torch-2.8 ABI and must not be reused).
module load CUDA/12.8.0 2>/dev/null || module load CUDA/12.9.1 2>/dev/null || true

export PYBIN=/scratch/sj157/race_vit_env/bin/python
export TORCH_CUDA_ARCH_LIST=9.0
export TORCH_EXTENSIONS_DIR=/scratch/sj157/RACE_Attention/.torch_ext_distill
mkdir -p "$TORCH_EXTENSIONS_DIR"

echo "[env] PYBIN=$PYBIN"
echo "[env] python torch: $($PYBIN -c 'import torch;print(torch.__version__)')"
echo "[env] nvcc: $(command -v nvcc || echo MISSING)"
echo "[env] TORCH_EXTENSIONS_DIR=$TORCH_EXTENSIONS_DIR"
