#!/bin/bash
#SBATCH --job-name=smoke_unfreeze
#SBATCH --partition=commons
#SBATCH --account=as143
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=00:40:00
#SBATCH --output=/scratch/sj157/RACE_Attention/distill/results/%x_%j.out
#SBATCH --error=/scratch/sj157/RACE_Attention/distill/results/%x_%j.out
# Smoke: --unfreeze mlp + aug-data, 4 steps, loading Stage C. Verifies MLP grads flow (gated
# assert), RACE still trains, frozen attn/embeds stay frozen, and base_state round-trips.
set -uo pipefail
cd /scratch/sj157/RACE_Attention
source distill/env.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
rm -f distill/checkpoints/ar_smoke_unfreeze_resume.pt distill/checkpoints/ar_smoke_unfreeze.done 2>/dev/null
"$PYBIN" distill/distill_global.py --pattern AR --race-l 3 --race-k 3 \
    --seq-len 4096 --batch-size 2 --grad-accum 1 --max-steps 4 --eval-every 4 \
    --unfreeze mlp --base-lr 1e-5 --lr 5e-5 --warmup-steps 1 \
    --hidden-weight 0.1 --kl-weight 1.0 --ce-weight 0.5 --kl-chunk 1024 --grad-checkpoint \
    --data mixed --lm-frac 0.5 --long-source fineweb_local \
    --task-weights "niah_single:4,variable_tracking:3,common_words:2" --n-distractors 24 \
    --save-every 4 --tag smoke_unfreeze \
    --load-race-checkpoint checkpoints/best/race_hybrid_AR_S24_p1c_ext_step450.pt
echo "########## checkpoint round-trip: base_state present + eval-loadable ##########"
"$PYBIN" - << 'PY'
import torch, sys; sys.path.insert(0,"distill")
ck = torch.load("distill/checkpoints/ar_L3_K3_smoke_unfreeze_step4.pt", map_location="cpu", weights_only=False)
bs = ck.get("base_state", {})
print(f"base_state tensors: {len(bs)} | config.unfreeze={ck['config'].get('unfreeze')}")
print("sample base keys:", list(bs.keys())[:3])
import eval_ruler
m, is_hy, meta = eval_ruler.build_model("ar", "distill/checkpoints/ar_L3_K3_smoke_unfreeze_step4.pt", "sdpa", "cuda", decode_mode="recompute")
print("build_model loaded base:", meta.get("n_base_loaded"), "unfreeze:", meta.get("unfreeze"))
print("ROUNDTRIP_OK" if len(bs) > 0 and meta.get("n_base_loaded") == len(bs) else "ROUNDTRIP_FAIL")
PY
echo "SMOKE_UNFREEZE_DONE"
