#!/bin/bash
#SBATCH --job-name=ruler_multi
#SBATCH --partition=commons
#SBATCH --account=as143
#SBATCH --gres=gpu:h200:3
#SBATCH --cpus-per-task=24
#SBATCH --mem=360G
#SBATCH --time=05:00:00
#SBATCH --output=/scratch/sj157/RACE_Attention/distill/results/%x_%j.out
#SBATCH --error=/scratch/sj157/RACE_Attention/distill/results/%x_%j.out
# RULER at 16K / 32K / 64K for a hybrid checkpoint (env CKPT), one length per GPU.
# Also emits prefill/decode tok/s + peak mem per run (hybrid-vs-teacher throughput).
set -uo pipefail
cd /scratch/sj157/RACE_Attention/distill
source /scratch/sj157/RACE_Attention/distill/env.sh
export HF_HUB_DOWNLOAD_TIMEOUT=60
: "${CKPT:?set CKPT to the checkpoint path}"
COMMON="--model ar --checkpoint $CKPT --max-examples 20 --decode-mode cache"
CUDA_VISIBLE_DEVICES=0 "$PYBIN" eval_ruler.py $COMMON --context-len 16384 > results/rmulti_16k.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 "$PYBIN" eval_ruler.py $COMMON --context-len 32768 > results/rmulti_32k.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 "$PYBIN" eval_ruler.py $COMMON --context-len 65536 > results/rmulti_64k.log 2>&1 &
wait
echo "########## RULER per-length ##########"
for L in 16k 32k 64k; do echo "=== $L ==="; grep -E "OVERALL RULER|decode_throughput|prefill_throughput|peak" results/rmulti_$L.log | tail -3; done
echo "RULER_MULTI_DONE"
