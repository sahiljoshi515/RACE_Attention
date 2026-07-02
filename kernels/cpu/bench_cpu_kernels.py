"""Optimality benchmark for the CPU RACE prefix-mean kernel.

Compares the OpenMP C++ kernel to the naive PyTorch cumsum reference at
model-relevant shapes, and reports OpenMP thread scaling. Time is median of
several reps after warmup.

Run:  python bench_cpu_kernels.py
"""
import os
import sys
import time
import statistics as stats
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from race_ext import load_race

race = load_race(verbose=False)


def ref_prefix_mean_flat(w, v, eps):
    A = w.cumsum(1)
    B = (w.unsqueeze(-1) * v).cumsum(1)
    return B / (A.unsqueeze(-1) + eps)


def median_ms(fn, warmup=2, reps=8):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return stats.median(ts) * 1e3


eps = 1e-6
# d24 RACE attention at T=2048: N=B*H*M=12, S=L*2^K=3*8=24 -> NS=288, D=head_dim=128
SHAPES = [(288, 512, 128), (288, 1024, 128), (288, 2048, 128), (288, 4096, 128)]

print(f"threads torch={torch.get_num_threads()} OMP={os.getenv('OMP_NUM_THREADS')}")
print(f"{'shape (NS,T,D)':>22} | {'kernel ms':>10} | {'torch ref ms':>12} | {'speedup':>8}")
for (NS, T, D) in SHAPES:
    w = torch.rand(NS, T, dtype=torch.float32).contiguous()
    v = torch.randn(NS, T, D, dtype=torch.float32).contiguous()
    tk = median_ms(lambda: race.race_prefix_mean_flat(w, v, eps))
    tr = median_ms(lambda: ref_prefix_mean_flat(w, v, eps))
    print(f"{str((NS,T,D)):>22} | {tk:10.2f} | {tr:12.2f} | {tr/tk:7.1f}x")

# --- backward timing (kernel only; torch autograd ref for comparison) ---
print("\nbackward (NS=288,T=2048,D=128):")
w = torch.rand(288, 2048, dtype=torch.float32).contiguous()
v = torch.randn(288, 2048, 128, dtype=torch.float32).contiguous()
g = torch.randn(288, 2048, 128, dtype=torch.float32).contiguous()
tk = median_ms(lambda: race.race_prefix_mean_flat_bw(w, v, g, eps))
print(f"  kernel backward: {tk:.2f} ms")
print("\nNOTE: kernel accumulates in fp64, O(NS*T*D), OpenMP over NS streams "
      "(bandwidth-bound). Set OMP_NUM_THREADS to scale.")
