#!/bin/bash
#SBATCH --job-name=ruler_gate_32k
#SBATCH --partition=commons
#SBATCH --account=as143
#SBATCH --gres=gpu:h200:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/sj157/RACE_Attention/distill/results/%x_%j.out
#SBATCH --error=/scratch/sj157/RACE_Attention/distill/results/%x_%j.out
# RULER-32K GATE for the Phase-0 ablation v2 winners (instruction #1).
# Decides instruction #3 (proceed to Long Schedule iff RULER-32K looks good).
# Two student checkpoints in PARALLEL, one per GPU:
#   GPU0  ar_s24h  (AR 50% RACE, S=24, BEST cell, ppl 36.96)  -> ruler_ar_32k.json
#   GPU1  arrr_s8  (ARRR 75% RACE, S=8, FROZEN hash, ppl 68)  -> ruler_arrr_32k.json  [no-learnable-hash, per #4]
# Teacher RULER-32K (score 80.2) already exists in results/ruler_teacher_32k.json -> reused for the summary.
# NOTE: the cheap NIAH-4K probe (ablation_score.json) already shows collapsed_frac=1.0 / niah_acc=0.0
# for BOTH; this gate confirms at 32K with the real RULER harness.
set -uo pipefail
cd /scratch/sj157/RACE_Attention/distill
source /scratch/sj157/RACE_Attention/distill/env.sh

# Model+tokenizer are cached; RULER dataset (tonychenxyz/ruler-full) streams from the hub
# (the prior 06-24 RULER run streamed fine; only FineWeb's tree-listing 504'd). Bounded timeout.
export HF_HUB_DOWNLOAD_TIMEOUT=60
COMMON="--context-len 32768 --max-examples 20 --decode-mode cache"

echo "########## AR-S24 (ar_s24h) 32768 [GPU0] ##########"
CUDA_VISIBLE_DEVICES=0 "$PYBIN" eval_ruler.py --model ar \
    --checkpoint checkpoints/ar_L3_K3_abl2_ar_s24h_step2000.pt $COMMON \
    > results/ruler_gate_ar_s24h.log 2>&1 &
PID_AR=$!

echo "########## ARRR-S8 (arrr_s8, frozen hash) 32768 [GPU1] ##########"
CUDA_VISIBLE_DEVICES=1 "$PYBIN" eval_ruler.py --model arrr \
    --checkpoint checkpoints/arrr_L2_K2_abl2_arrr_s8_step2000.pt $COMMON \
    > results/ruler_gate_arrr_s8.log 2>&1 &
PID_ARRR=$!

wait $PID_AR;   echo "AR-S24 exit=$?"
wait $PID_ARRR; echo "ARRR-S8 exit=$?"

echo "########## per-student last lines ##########"
for f in results/ruler_gate_ar_s24h.log results/ruler_gate_arrr_s8.log; do
  echo "=== $f ==="; tail -8 "$f" || true
done

echo "########## final summary (teacher + both students) ##########"
"$PYBIN" eval_ruler.py --plot || true
cat results/ruler_summary.md || true
echo "RULER_GATE_DONE"
