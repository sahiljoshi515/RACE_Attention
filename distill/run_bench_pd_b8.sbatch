#!/bin/bash
#SBATCH --job-name=bench_pd_b8
#SBATCH --partition=commons
#SBATCH --account=as143
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=160G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/sj157/RACE_Attention/distill/results/%x_%j.out
#SBATCH --error=/scratch/sj157/RACE_Attention/distill/results/%x_%j.out
set -o pipefail
export PYTHONPATH="${PYTHONPATH:-}"
cd /scratch/sj157/RACE_Attention
source distill/env_fa3.sh
export HF_HUB_DOWNLOAD_TIMEOUT=60
echo "########## B=8 prefill+decode sweep (FA3 full vs FA3+RACE hybrid), 4-64K, decode 64 ##########"
"$PYBIN" distill/bench_prefill_decode.py --batch 8 --decode-steps 64
echo "BENCH_PD_B8_DONE"
