#!/bin/bash
#SBATCH --job-name=ruler_ckpt
#SBATCH --partition=commons
#SBATCH --account=as143
#SBATCH --gres=gpu:h200:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/sj157/RACE_Attention/distill/results/%x_%j.out
#SBATCH --error=/scratch/sj157/RACE_Attention/distill/results/%x_%j.out
# RULER 16K + 32K gate for a hybrid checkpoint (env CKPT). Teacher RULER (80.2) already
# exists in results/ for the summary comparison. Two context lengths in parallel.
set -uo pipefail
cd /scratch/sj157/RACE_Attention/distill
source /scratch/sj157/RACE_Attention/distill/env.sh
export HF_HUB_DOWNLOAD_TIMEOUT=60          # RULER dataset streams from hub (worked in job 144605)
: "${CKPT:?set CKPT to the checkpoint path}"
COMMON="--model ar --checkpoint $CKPT --max-examples 20 --decode-mode cache"

CUDA_VISIBLE_DEVICES=0 "$PYBIN" eval_ruler.py $COMMON --context-len 16384 \
    > results/ruler_ckpt_16k.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 "$PYBIN" eval_ruler.py $COMMON --context-len 32768 \
    > results/ruler_ckpt_32k.log 2>&1 &
wait
echo "########## RULER 16K tail ##########"; tail -6 results/ruler_ckpt_16k.log || true
echo "########## RULER 32K tail ##########"; tail -6 results/ruler_ckpt_32k.log || true
echo "########## summary ##########"
"$PYBIN" eval_ruler.py --plot || true
cat results/ruler_summary.md || true
echo "RULER_CKPT_DONE"
