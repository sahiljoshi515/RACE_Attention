#!/bin/bash
#SBATCH --job-name=race_ablation_v2
#SBATCH --partition=commons
#SBATCH --account=as143
#SBATCH --gres=gpu:h200:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/sj157/RACE_Attention/distill/results/%x_%j.out
#SBATCH --error=/scratch/sj157/RACE_Attention/distill/results/%x_%j.out
# Phase-0 ablation v2: corrected 2-factor design (pattern x S) + a learnable-hash
# isolation arm. 6 cheap 4K cells, ALL FRESH (no --load/--resume), run in PARALLEL
# one per GPU (single-GPU each, NO DDP). Each saves a numbered eval-compatible
# checkpoint at step 2000 for the RULER-4K probe + ablation_score.py.
set -uo pipefail
cd /scratch/sj157/RACE_Attention
source distill/env.sh

# FULLY OFFLINE: model+tokenizer are cached and FineWeb is pre-staged locally
# (stage_fineweb.py -> data/fineweb/, read by data_fineweb._open_stream with no hub).
# The hub's dataset tree-listing/etag calls 504'd / silently stalled at job start (that
# was the real hang, NOT the GPUs); offline mode removes all hub access. Streaming shuffle
# stays OFF (FINEWEB_SHUFFLE_BUFFER unset) so there's no shard-tree enumeration.
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DOWNLOAD_TIMEOUT=60

run_cell () {   # $1=gpu  $2=tag  $3=batch_size  $4...=extra distill args
  local gpu=$1; shift; local tag=$1; shift; local bs=$1; shift
  # seed=0 for ALL cells: with local (network-free) reads + shuffle off there is no
  # thundering-herd, so every cell sees IDENTICAL data + identical RACE hash init ->
  # the cleanest possible architecture comparison (differences are purely arch).
  CUDA_VISIBLE_DEVICES=$gpu "$PYBIN" distill/distill_global.py \
      --seq-len 4096 --batch-size "$bs" --max-steps 2000 --eval-every 100 --save-every 2000 \
      --lr 5e-5 --warmup-steps 100 --lr-schedule cosine --total-steps 2000 \
      --hidden-weight 1.0 --kl-weight 0.5 --ce-weight 0.1 --kl-chunk 2048 \
      --grad-checkpoint --seed 0 --data fineweb --tag "$tag" "$@" \
      > distill/results/ablation_v2_${tag}.log 2>&1 &
}

# Memory note (141GB H200, both 3B-bf16 models ~6GB each resident, grad-checkpoint ON):
# MEASURED from the prior abl2 run (results/metrics_global_abl2_*.jsonl): the heaviest
# cell, S=64 ARRR (S = L*2^K = 4*2^4 = 64, 21 RACE layers each materializing probs
# [B*24heads, T=4096, S=64] fp32), peaked at 50.2 GB at B=2 -- comfortably under 141GB.
# AR-S64 peaked 46.7 GB, edges-S64 48.7 GB. So B=2 is safe for every cell; no B=1 needed.

#         gpu  tag             bs  <cell args>
run_cell  0    abl2_arrr_s8    2  --pattern ARRR    --race-l 2 --race-k 2                                    # frozen-hash baseline (~prior config)
run_cell  1    abl2_arrr_s8h   2  --pattern ARRR    --race-l 2 --race-k 2 --train-hash-geometry --hash-lr 1e-3  # isolates learnable hash
run_cell  2    abl2_arrr_s64h  2  --pattern ARRR    --race-l 4 --race-k 4 --train-hash-geometry --hash-lr 1e-3  # isolates S 8->64
run_cell  3    abl2_ar_s64h    2  --pattern AR      --race-l 4 --race-k 4 --train-hash-geometry --hash-lr 1e-3  # pattern AR vs ARRR @S64
run_cell  4    abl2_edges_s64h 2  --pattern edges:4 --race-l 4 --race-k 4 --train-hash-geometry --hash-lr 1e-3  # pattern edges vs ARRR @S64
run_cell  5    abl2_ar_s24h    2  --pattern AR      --race-l 3 --race-k 3 --train-hash-geometry --hash-lr 1e-3  # moderate S=24, AR
wait

echo "########## ablation v2 cells done; per-cell logs in results/ablation_v2_*.log ##########"
# Score every step-2000 checkpoint from this v2 run (ablation_score.py globs *abl2*step2000.pt).
# NOTE: ablation_score.py is a separate artifact; if it is absent this line errors but
# (no set -e) the eval-line summary + DONE marker below still run.
"$PYBIN" distill/ablation_score.py
echo "########## per-cell last eval line ##########"
for f in distill/results/ablation_v2_*.log; do
  echo "=== $f ==="
  grep -E "eval@" "$f" | tail -1 || true
done
echo "ABLATION_V2_DONE"
