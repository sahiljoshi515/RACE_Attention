# RULER long-context evaluation — teacher vs ARRR hybrid (eval-only)

_Evaluation-only harness (`eval_ruler.py`); no training, no checkpoint writes. Compares
Llama-3.2-3B-Instruct (teacher) against the ARRR 75%-RACE hybrid on RULER at 32K & 64K._

## Pipeline

- **Harness:** `distill/eval_ruler.py`. CLI: `--model {teacher,ar,arrr}` / `--checkpoint`,
  `--context-len {32768,65536}`, `--max-examples {N,all}`, `--tasks`, `--max-new-tokens`,
  `--attn-impl`, `--plot`. Launchers: `run_ruler_full_{32768,65536}.sbatch`.
- **Data:** `tonychenxyz/ruler-full` (streaming), filtered on the integer
  `extra_info.context_length` field. Prompts split query-agnostically into
  (context, question, answer_prefix) and re-templated for the Llama-3.2 chat format.
- **Tasks (6):** niah_single_1, niah_single_2, niah_multikey_1, niah_multivalue,
  vt (variable_tracking), cwe (common_words_extraction). All 13 RULER tasks selectable.
- **Inference:** teacher uses native KV-cached greedy decode; the hybrid has no RACE KV
  cache, so decode forces `use_cache=False` and recomputes the full sequence per token.
  Both split prefill vs decode timing; `logits_to_keep=1` avoids the full-sequence lm_head
  tensor. A repetition early-stop bounds decode cost on collapsed generations.
- **Metrics:** RULER string-match (per task + overall), normalized exact-match, prefill /
  decode latency, prefill / decode throughput, avg tok/s, peak GPU memory.
- **AR (50%):** no checkpoint exists, so `--model ar` is auto-skipped.

## Harness-faithfulness check (`ppl_probe.py`)

Reproduced the distillation teacher-forced eval (FineWeb held-out, seed=0, seq 4096, B=1):

| build | ppl | distillation target |
|---|:--:|:--:|
| teacher | 22.0 | ~22 |
| ARRR — harness build (bf16 params) | 147.4 | ~146 |
| ARRR — fp32 params + autocast | 147.4 | ~146 |

The harness's hybrid build reproduces the distillation perplexity exactly; the bf16 param
cast is lossless (RACE core math is fp32 internally). The build loads and runs correctly.

## Results (20 examples/task, greedy, seed 0; hybrid decode_mode=cache)

| Model | Context | Avg Score | Exact Match | Prefill (tok/s) | Decode (tok/s) | Peak Memory (GB) |
| ----- | ------- | --------- | ----------- | --------------- | -------------- | ---------------- |
| teacher | 32K | 80.17 | 0.0 | 34910.0 | 66.34 | 12.64 |
| arrr | 32K | 0.0 | 0.0 | 48892.4 | 52.65 | 9.83 |
| teacher | 64K | 80.46 | 0.0 | 22324.1 | 62.43 | 18.83 |
| arrr | 64K | 0.0 | 0.0 | 41556.2 | 50.76 | 13.20 |

(Decode now uses the incremental KV-cache path — see `REPORT_decode_cache.md`. Quality is
token-identical to the earlier `recompute` runs; only speed/memory change. ARRR decode rose
from 1.6/0.68 tok/s recompute → 52.7/50.8 tok/s. Hybrid decode is flat in T and overtakes the
teacher beyond ~131K; at ≤64K it is ~0.73× the teacher because decode there is weight-bandwidth-
bound, not attention-bound.)

Per-task RULER string-match:

| task | teacher 32K | teacher 64K | arrr 32K | arrr 64K |
|---|:--:|:--:|:--:|:--:|
| niah_single_1 | 100 | 100 | 0 | 0 |
| niah_single_2 | 100 | 100 | 0 | 0 |
| niah_multikey_1 | 100 | 100 | 0 | 0 |
| niah_multivalue | 100 | 98.75 | 0 | 0 |
| vt | 81.0 | 84.0 | 0 | 0 |
| cwe | 0 | 0 | 0 | 0 |

(cwe is 0 for both: predictions are correctly-formatted 10-word lists whose words do not
match the planted reference words — a genuine task miss under the query-agnostic format,
not a pipeline error. exact-match is 0 throughout because predictions carry surrounding
punctuation/prefixes while RULER's metric is substring match.)

## Outputs (`results/`)

- `ruler_{teacher,arrr}_{32k,64k}.json` — full per-task quality, speed, memory, per-example rows.
- `ruler_accuracy.png`, `ruler_speed.png`, `ruler_memory.png`, `ruler_context_scaling.png`.
- `ruler_summary.md` — the summary table above.

## Reproduce

```bash
sbatch run_ruler_full_32768.sbatch    # teacher + arrr @ 32K, 20/task
sbatch run_ruler_full_65536.sbatch    # teacher + arrr @ 64K, 20/task
python eval_ruler.py --plot           # rebuild plots + summary from results/
# pilot / full: --max-examples 100 | all  (ARRR decode is full-recompute, so 64K-all is slow)
```
