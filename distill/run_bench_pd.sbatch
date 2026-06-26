#!/bin/bash
#SBATCH --job-name=bench_pd
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
source distill/env_fa3.sh                     # swa_env (torch 2.8) + genuine FA3 + .torch_ext_fa3
export HF_HUB_DOWNLOAD_TIMEOUT=60
echo "########## STEP 1: SANITY (FA3 + RACE decode kernel build under torch 2.8) ##########"
# --quick: smallest T, 1 decode step; builds race_causal/race_decode kernels (first JIT, ~min),
# asserts FA3 engaged, prints race_kernel_used.
"$PYBIN" distill/bench_prefill_decode.py --quick --srrr 2>&1 | tail -25
echo "########## STEP 2: FULL SWEEP 4-64K + decode 64 ##########"
"$PYBIN" distill/bench_prefill_decode.py --srrr --decode-steps 64
echo "########## PLOT ##########"
"$PYBIN" distill/plot_prefill_decode.py
echo "BENCH_PD_DONE"
