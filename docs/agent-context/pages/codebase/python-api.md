# Python API - the nn.Module surface (`misc/race.py`)

The drop-in attention modules. Mirrors Algorithm 1/2 with `K` (hash bits, paper's P), `L` (tables),
`M` (ensembles). Total bucket summaries per position $S = L\cdot 2^K$ (paper's $S = L\cdot R$, $R = 2^P$).

## Classes (`misc/race.py`)

- `BatchedACE` (L26) - the core causal attention engine. Soft-hashes Q/K into bucket probabilities
  and computes the causal prefix mean (pure-PyTorch `cumsum` path here; the CUDA path lives in
  `scaling/`). Multi-ensemble (averages M independent sketches).
- `RACEAttention` (L144) - multi-head wrapper. Constructor
  `__init__(d, h, K, L, M, drop=0.1, qkv_bias=False, device='cpu')`; `forward(x)` takes `[B,T,d]`
  $\to$ `[B,T,d]`. Packs heads/ensembles `[B,T,H,d]` $\to$ `[M,B,H,T,d]` $\to$ `[N,T,d]` ($N = M\cdot B\cdot H$) before the ACE
  engine.
- `RACEBlock` (L176) - a full Transformer block (attention + feedforward + LayerNorm). Expects a
  `cfg` dict with `emb_dim`, `n_heads`, `K`, `L`, `M`, `drop_rate`, `qkv_bias`.

## Parameter mapping (code ↔ paper)

| Code | Paper | Meaning |
| --- | --- | --- |
| `K` | `P` | hyperplanes per table; $2^K$ corners/buckets |
| `L` | `L` | hash tables (variance $\downarrow$ as $L \uparrow$) |
| `M` | (ensembles) | independent sketches averaged |
| $S = L\cdot 2^K$ | $S = L\cdot R$ | bucket summaries each query mixes with |
| trainable temperature | $\beta$ | softmax sharpness in soft assignment |

## Two compute paths

1. **Pure PyTorch** (`misc/race.py` / `scaling/race_torch_cumsum.py`): `A_pref = probsK.cumsum`,
   `B_pref = (probsK[...,None]*V[...,None,:]).cumsum`, `E = B_pref/(A_pref+eps)`. Correct but
   materializes `B_pref[N,T,S,D]` $\to$ OOMs on long T. Used as reference.
2. **CUDA-backed** (`scaling/race_causal_cuda.py`): `RaceCausalFn`/`RaceCausalCuda` call the
   chunked-scan kernels (`codebase/gpu-kernels`). This is what scales to millions of tokens.

`gpt.py` provides a standard softmax `MultiHeadAttention` for reference comparisons.

---
Source: misc/race.py (BatchedACE L26, RACEAttention L144, RACEBlock L176). Verified against HEAD c620cdc.
