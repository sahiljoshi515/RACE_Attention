#!/bin/bash
#SBATCH --job-name=race_smoke_core
#SBATCH --partition=commons
#SBATCH --account=as143
#SBATCH --gres=gpu:h200:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=00:40:00
#SBATCH --output=/scratch/sj157/RACE_Attention/distill/results/%x_%j.out
#SBATCH --error=/scratch/sj157/RACE_Attention/distill/results/%x_%j.out
# Core infra smoke that does NOT touch data_long/PG19 (fineweb loader only): validates the
# rewritten loop, cosine LR, grad-accum, full-state resume, and manual-allreduce DDP.
set -uo pipefail
cd /scratch/sj157/RACE_Attention
source distill/env.sh
rm -f distill/checkpoints/*smoke* distill/results/metrics_global_smoke*
P="$PYBIN distill/distill_global.py --grad-checkpoint --warmup-steps 1 --kl-chunk 1024 --data fineweb"

echo "===== A. fineweb 3-step (1 GPU, default path, ARRR S8) ====="
CUDA_VISIBLE_DEVICES=0 $P --pattern ARRR --race-l 2 --race-k 2 --seq-len 1024 \
    --batch-size 1 --max-steps 3 --eval-every 2 --tag smoke_fw

echo "===== B. grad-accum=2 + cosine, 4-step, save_resume_every 2 ====="
CUDA_VISIBLE_DEVICES=0 $P --pattern AR --race-l 4 --race-k 4 --train-hash-geometry --hash-lr 1e-3 \
    --seq-len 1024 --batch-size 1 --grad-accum 2 --lr-schedule cosine --total-steps 4 \
    --max-steps 4 --eval-every 10 --save-resume-every 2 --tag smoke_res

echo "===== C. resume auto -> continue 4 -> 6 (expect 'RESUMED ... at step 4') ====="
CUDA_VISIBLE_DEVICES=0 $P --pattern AR --race-l 4 --race-k 4 --train-hash-geometry --hash-lr 1e-3 \
    --seq-len 1024 --batch-size 1 --grad-accum 2 --lr-schedule cosine --total-steps 4 \
    --max-steps 6 --eval-every 10 --save-resume-every 2 --resume auto --tag smoke_res

echo "===== D. DDP 2-GPU 3-step (manual allreduce; bGrad must stay 0) ====="
"$PYBIN" -m torch.distributed.run --standalone --nproc_per_node=2 \
    distill/distill_global.py --grad-checkpoint --warmup-steps 1 --kl-chunk 1024 --data fineweb --ddp \
    --pattern edges:4 --race-l 4 --race-k 4 --seq-len 1024 --batch-size 1 --max-steps 3 \
    --eval-every 10 --tag smoke_ddp

echo "===== SMOKE_CORE_DONE ====="
