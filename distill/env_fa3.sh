#!/bin/bash
# Source on a GPU (H200) node to run the forward-latency bench with GENUINE FA3.
#   source distill/env_fa3.sh
# Uses swa_env (torch 2.8.0+cu128, transformers 4.57.0) + the prebuilt FA3 overlay
# (torch-2.8 ABI) so attn_implementation="flash_attention_3" routes to the real
# Dao FlashAttention-3. The RACE CUDA kernel JIT-builds against torch 2.8 in its
# own cache (distinct from the torch-2.10 distill cache).
module load CUDA/12.8.0 2>/dev/null || module load CUDA/12.9.1 2>/dev/null || true

export PYBIN=/scratch/sj157/swa_env/bin/python
export PYTHONPATH=/scratch/sj157/race_bench_fa3:${PYTHONPATH}      # prebuilt FA3 (torch-2.8 ABI)
export TORCH_CUDA_ARCH_LIST=9.0
export TORCH_EXTENSIONS_DIR=/scratch/sj157/RACE_Attention/.torch_ext_fa3   # torch-2.8 race_cuda build
mkdir -p "$TORCH_EXTENSIONS_DIR"

echo "[env] PYBIN=$PYBIN  ($($PYBIN -c 'import torch,transformers;print("torch",torch.__version__,"tf",transformers.__version__)'))"
echo "[env] FA3: $($PYBIN -c 'import flash_attn_interface;print("import OK")' 2>&1 | head -1)"
echo "[env] nvcc: $(command -v nvcc || echo MISSING)"
