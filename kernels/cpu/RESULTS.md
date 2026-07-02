# CPU RACE kernels — validation & CPU speed comparison

All numbers from `bg4u9g1` (16 physical cores, GCC 14.3.0, torch 2.9.1 CPU,
`OMP_NUM_THREADS=16`). Reproduce with the scripts referenced below.

## 1. Correctness (fwd + bwd)

`race_pref.cpp` (`race_prefix_mean_flat` / `_bw`) and `linear_pref.cpp`
(Katharopoulos causal linear attention) are validated against float64 PyTorch
references + autograd in `test_cpu_kernels.py`. Max relative error:

| kernel | forward | backward |
|---|---|---|
| `race_prefix_mean_flat` | ~3e-8 (up to T=4096) | ~4e-8 (gW, gV) |
| `linear_pref` causal_dot | ~8e-7 | ~7e-7 (gQ, gK, gV) |

The backward math was also **independently re-derived and audited** (suffix-sum
gradients `dL/dw_t = α_suf_t + ⟨β_suf_t, v_t⟩`, `dL/dv_t = w_t·β_suf_t`) and
matches the C++ line-by-line. Both kernels: **CORRECT**.

The CPU RACE composite `scaling/race_causal_cpu.py` (`RaceCausalCPUFn` = the
prefix-mean kernel + a `probsQ` gather) reproduces `race_common.race_prefix_ref`
(the same ground truth the CUDA kernel targets) at ~1e-7 fwd/bwd
(`scaling/test_race_cpu_composite.py`) — so the CPU path is numerically
equivalent to the CUDA path.

## 2. Optimality — `race_prefix_mean_flat` vs naive torch cumsum

`bench_cpu_kernels.py`, shape (NS=288, D=128) = d24 RACE attention (B·H·M=12
streams × S=24 buckets):

| (NS, T, D) | kernel | torch cumsum ref | speedup |
|---|---|---|---|
| (288, 512, 128)  | 3.4 ms  | 14.2 ms  | 4.2× |
| (288, 1024, 128) | 6.6 ms  | 28.5 ms  | 4.3× |
| (288, 2048, 128) | 13.1 ms | 130.2 ms | **9.9×** |
| (288, 4096, 128) | 26.1 ms | 287.8 ms | **11.0×** |

Single fused pass, fp64 accumulation, OpenMP over the NS independent streams,
bandwidth-bound. Speedup grows with T (torch materializes cumsum intermediates;
the kernel does not). Known micro-opt headroom (not correctness): scalar inner
loops (no explicit SIMD/blocking), per-stream heap scratch, and `linear_pref`
lacks dtype/contiguity guards.

## 3. RACE vs softmax on CPU @ trained context length (T=2048)

Full-model forward, d24 (1.384B params), B=1, T=2048, 16 threads, fp32, random
weights (forward FLOPs are weight-independent), one model per process to avoid
the `nanochat` package name clash (`scaling/bench_race_vs_softmax_cpu.py`):

| model | median forward | vs softmax |
|---|---|---|
| **softmax** (SDPA) | **3103 ms** | 1.00× |
| **RACE** (CPU prefix-scan) | **3707 ms** | 1.20× slower |

**At the trained context length, softmax is faster than RACE on CPU** (~1.2–1.35×
depending on warmup/reps). RACE's O(T) linear-attention scaling is outweighed at
T=2048 by its constant factors — soft-hash projections, S=24 per-bucket prefix
scans, and the `probsQ` gather — whereas softmax attention is O(T²) but runs as a
single tight BLAS/SDPA call.

### T-sweep (full forward, B=1, 16 threads, reps=2)

| T | RACE | softmax | RACE / softmax |
|---|---|---|---|
| 512  | 1233 ms  | 801 ms   | 1.54× slower |
| 1024 | 2343 ms  | 1551 ms  | 1.51× |
| 2048 | 4579 ms  | 3402 ms  | 1.35× |
| 4096 | 8608 ms  | 6940 ms  | 1.24× |
| 8192 | 20007 ms | 18069 ms | 1.11× |

Softmax is faster at **every** tested length, but the ratio **narrows
monotonically** (1.54× → 1.11×) — the O(T)-vs-O(T²) signature: RACE closes the
gap as T grows. The crossover (RACE overtakes) is **beyond T=8192**, i.e. well
past the 2048 context this model was trained at. Takeaway: on CPU, PyTorch SDPA's
tiny constant factor means RACE's linear scaling does not pay off in the trained
regime; RACE would only win at much longer contexts.

### Reproduce
```bash
# on a CPU node with torch + g++/OpenMP:
cd kernels/cpu && python test_cpu_kernels.py && python bench_cpu_kernels.py
cd ../../scaling && python test_race_cpu_composite.py
python bench_race_vs_softmax_cpu.py --model race    --pkg-dir ../chat --threads 16
python bench_race_vs_softmax_cpu.py --model softmax --pkg-dir /path/to/nanochat-softmax --threads 16
```
