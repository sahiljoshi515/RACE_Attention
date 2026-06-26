#!/bin/bash
#SBATCH --job-name=mqar_diag
#SBATCH --partition=commons
#SBATCH --account=as143
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/sj157/RACE_Attention/distill/results/%x_%j.out
#SBATCH --error=/scratch/sj157/RACE_Attention/distill/results/%x_%j.out
set -uo pipefail
cd /scratch/sj157/RACE_Attention
source distill/env.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "########## SMOKE (tiny, validates harness imports+runs) ##########"
"$PYBIN" distill/mqar_diag.py --steps 60 --log-every 60 --seeds 0 --kinds softmax mean delta \
    --batch 16 --eval-batch 64 --hidden 128 --layers 2 || { echo "SMOKE_FAILED"; exit 1; }
echo "########## FULL DIAGNOSTIC (3 kinds x 3 seeds, solvable regime) ##########"
"$PYBIN" distill/mqar_diag.py --steps 4000 --seeds 0 1 2 --kinds softmax mean delta \
    --hidden 256 --layers 2 --mlp-hidden 512 --lr 1e-3 \
    --kv-pairs 4 --queries 8 --num-keys 32 --num-values 16
echo "MQAR_DIAG_DONE"
