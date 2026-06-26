#!/bin/bash
#SBATCH --job-name=fwd_fa3_srrr
#SBATCH --partition=commons
#SBATCH --account=as143
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/sj157/RACE_Attention/distill/results/fwd_fa3_srrr_%j.out

# Forward-latency: full Llama-3.2-3B (all attn = GENUINE FA3) vs the (Softmax,RACE,RACE,RACE)
# hybrid -- 7 EVEN-of-4 layers (0,4,..,24) stay FA3, the other 21 are causal RACE.
# Sweep T=4K..1M. swa_env (torch 2.8, ABI-compatible with prebuilt FA3 _C.so) + .torch_ext_fa3.
set -e
cd /scratch/sj157/RACE_Attention
source distill/env_fa3.sh       # swa_env (torch 2.8) + FA3 overlay + .torch_ext_fa3

# --- preflight: assert GENUINE FA3 is usable on this GPU (fail fast, never fall back to sdpa) ---
$PYBIN - <<'PY'
import sys, torch
import transformers.utils.import_utils as iu
ok = iu.is_flash_attn_3_available()
print("[preflight] torch", torch.__version__, "| is_flash_attn_3_available:", ok, flush=True)
import flash_attn_interface as fa3
q = torch.randn(1, 256, 4, 128, device="cuda", dtype=torch.bfloat16)  # head_dim=128 = Llama-3.2-3B
o = fa3.flash_attn_func(q, q, q, causal=True)
o = o[0] if isinstance(o, (tuple, list)) else o
print("[preflight] genuine FA3 flash_attn_func OK, out", tuple(o.shape), flush=True)
if not ok:
    print("[preflight] FATAL: transformers will not route attn_implementation=flash_attention_3", flush=True)
    sys.exit(3)
print("[preflight] OK -> genuine FA3 confirmed", flush=True)
PY

$PYBIN distill/bench_forward.py --attn-impl flash_attention_3 \
    --pattern srrr \
    --exps 12 13 14 15 16 17 18 19 20 \
    --configs 2,2 3,3 \
    --out distill/results/fwd_latency_fa3_srrr.csv
echo DONE
