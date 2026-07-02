#!/usr/bin/env bash
# One-command launcher for the RACE-nanochat d24 base model.
#
#   bash run.sh                         # interactive REPL on a GPU
#   bash run.sh --prompt "..."          # one-shot completion
#   bash run.sh --temperature 0.8 --max-tokens 128
#
# What it does: (1) make sure we're on an NVIDIA GPU (on a SLURM cluster it grabs
# one via srun and re-execs), (2) load a CUDA toolkit for the RACE kernel JIT
# build, (3) create a Python env + install inference deps, (4) launch chat.py,
# which auto-downloads the weights from Hugging Face on first run.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- 1) Ensure a visible NVIDIA GPU. On SLURM, allocate one and re-exec once. ---
if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi -L >/dev/null 2>&1; then
  if [ "${RACE_CHAT_REEXEC:-0}" != "1" ] && command -v srun >/dev/null 2>&1; then
    echo "[run] No GPU on this node; requesting one via srun ..."
    exec env RACE_CHAT_REEXEC=1 srun --gres=gpu:1 --cpus-per-task=8 --mem=48G \
        --time="${RACE_CHAT_TIME:-2:00:00}" --pty bash "$HERE/run.sh" "$@"
  fi
  echo "[run] ERROR: no NVIDIA GPU visible and no srun available." >&2
  echo "[run] Run this on a CUDA-capable GPU node." >&2
  exit 1
fi

# --- 2) CUDA toolkit for the RACE attention kernel (JIT-compiled on first token). ---
if command -v module >/dev/null 2>&1; then
  module load CUDA/12.8.0 2>/dev/null || module load cuda 2>/dev/null || true
fi
command -v nvcc >/dev/null 2>&1 || \
  echo "[run] WARNING: nvcc not on PATH; the RACE CUDA kernel build may fail." >&2
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"   # H200=9.0; set to your GPU's arch
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$HERE/.torch_ext}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "$TORCH_EXTENSIONS_DIR"

# --- 3) Python env + inference deps (uv if available, else venv+pip). ---
VENV="$HERE/.venv"
PT_INDEX="https://download.pytorch.org/whl/cu128"
if command -v uv >/dev/null 2>&1; then
  export UV_CACHE_DIR="${UV_CACHE_DIR:-$HERE/.uv-cache}"
  [ -d "$VENV" ] || uv venv --python 3.10 "$VENV"
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
  python -c "import torch, tiktoken, huggingface_hub" 2>/dev/null || \
    uv pip install -r "$HERE/requirements.txt" --extra-index-url "$PT_INDEX"
else
  [ -d "$VENV" ] || python3 -m venv "$VENV"
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
  python -c "import torch, tiktoken, huggingface_hub" 2>/dev/null || {
    pip install --quiet --upgrade pip
    pip install -r "$HERE/requirements.txt" --extra-index-url "$PT_INDEX"
  }
fi

# --- 4) Launch. ---
cd "$HERE"
exec python chat.py "$@"
