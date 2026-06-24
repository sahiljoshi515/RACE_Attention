# Crosswalk: open questions carried into the vLLM backend

The risks/unknowns that survive the paper + review and matter for `feat/vllm-race-attention-backend`
(`codebase/vllm-backend`). Each links the paper gap, the review evidence, and the code seam.

## 1. Causal correctness has no theoretical guarantee

- **Gap:** Theorem 1's bias-variance bound is **non-causal only**; extending it to causal masking is
  explicitly an open problem (cumulative-sum constraint vs random features). `paper/06-theory`.
- **Review:** if3U Q1 asked exactly this; authors confirmed Algorithm 2 is implemented but the
  analysis is out of scope. `crosswalk/concerns-tracker` #10.
- **Backend implication:** the decode path is inherently causal, so correctness rests on **empirical
  parity** (`scaling/test_kernels.py`), not theory. Extend that harness for the vLLM path.

## 2. "Context window" = single attention primitive, not a full model

- **Gap:** 12M (GPU) / 75M (CPU) are limits of one attention layer's forward-backward pass, fixed
  batch=1, 4 heads, d=128. `paper/08-scaling`.
- **Review:** if3U's central objection - memory compounds across layers; a full model adds
  embeddings, FFN, many layers, and KV-cache. Title was rescoped; the abstract still reads
  ambiguously (AC caution). `crosswalk/concerns-tracker` #7/#8.
- **Backend implication:** do not advertise model context = primitive scaling. Budget per-layer
  memory × depth + KV-cache when sizing a real deployment.

## 3. Decode is incremental-append; current kernels are full-sequence scans

- **Observation:** Algorithm 2's `A_cum`/`B_cum` are exactly a **per-table KV-cache** (size $R + R\cdot d$
  per table per head). Autoregressive decode appends one token's $\phi_K$/$\phi_K\cdot V$ to these and reads
  with the new $\phi_Q$. `paper/09-causal-algorithm`.
- **Code reality:** `race_fused_fwd` is a chunked scan over the whole sequence, not a single-token
  append; an incremental-decode kernel is **new work**. `codebase/gpu-kernels`.
- The paper's conclusion explicitly names "efficient KV caching during inference" as the follow-up.

## 4. FlashAttention version / parity target

- **Review:** eQBU asked which FlashAttention was benchmarked; authors added FA2/FA3 + Sigmoid.
  `crosswalk/concerns-tracker` #4.
- **Backend implication:** pin the parity target (which FA version, which precision) before claiming
  throughput wins for the vLLM backend.

## 5. Numerical precision

- CPU kernel accumulates in **double precision** (`kernels/cpu/race_pref.cpp`); CUDA path is fp32
  with exact A/B reconstruction in backward. Decide the decode-kernel precision deliberately and
  test against `race_prefix_ref` (fp64). `codebase/cpu-kernels`, `codebase/gpu-kernels`.

## 6. Novelty framing (for any write-up)

- if3U: the durable novelty is **differentiable soft bucketing** + a formal bound on top of
  LSH-attention (à la YOSO), and the **systems** work; frame it as "makes LSH-attention practical,"
  not as a wholly new mechanism. Relevant if the backend ships with a paper/PR narrative.

---
Source: synthesized from `paper/06-theory`, `paper/08-scaling`, `paper/09-causal-algorithm`, `reviews/*`, and code seams in `codebase/{gpu-kernels,cpu-kernels,scaling-module,vllm-backend}`. Verified against HEAD c620cdc.
