#!/bin/bash
#SBATCH --job-name=dbg_cudagraph
#SBATCH --partition=commons
#SBATCH --account=as143
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/sj157/RACE_Attention/distill/results/%x_%j.out
#SBATCH --error=/scratch/sj157/RACE_Attention/distill/results/%x_%j.out
set -euo pipefail
cd /scratch/sj157/RACE_Attention/distill
source /scratch/sj157/RACE_Attention/distill/env.sh
"$PYBIN" /scratch/sj157/RACE_Attention/kernels/gpu/race_decode_build.py
"$PYBIN" debug_cudagraph_hybrid.py
