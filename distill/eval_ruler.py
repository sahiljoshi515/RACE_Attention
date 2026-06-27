"""Long-context RULER evaluation harness — Llama-3.2-3B-Instruct (teacher) vs the
RACE-hybrid students (AR 50% / ARRR 75%). EVALUATION ONLY: loads checkpoints, runs
inference, computes RULER quality + speed + memory metrics, writes JSON/plots/tables.
It does NOT train, modify training code, or write checkpoints.

Data: tonychenxyz/ruler-full (streaming), filtered by the `_{context_length}` category
marker so --context-len {32768,65536} selects the slice. Prompts are split into
(context, question, answer_prefix) the same way as Prism-Test's ruler64k benchmark
(query-agnostic), then re-templated for the Llama-3.2-3B-Instruct chat format and
scored with RULER string-match (the standard metric) plus normalized exact-match.

Model inference:
  * teacher  — native softmax, KV-cached greedy decode (its fast path).
  * hybrid   — RACE layers have no KV cache, so decode forces use_cache=False and
               re-runs the full sequence each step (correct, intentionally slower;
               this is what the speed comparison is meant to expose).

Examples
--------
  # smoke (20/task), ARRR @32K
  python eval_ruler.py --model arrr \
      --checkpoint checkpoints/arrr_L2_K2_arrr_1k_ce_step1000.pt \
      --context-len 32768 --max-examples 20
  # teacher @64K, pilot
  python eval_ruler.py --model teacher --context-len 65536 --max-examples 100
  # render plots + summary table from whatever results/ruler_*.json exist
  python eval_ruler.py --plot
"""
from __future__ import annotations

import os
import sys
import re
import ast
import json
import time
import argparse
from typing import Dict, List, Optional, Tuple

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
RES = os.path.join(HERE, "results")

TEACHER_MODEL = "meta-llama/Llama-3.2-3B-Instruct"

# Default checkpoints for the named students (overridable with --checkpoint).
DEFAULT_CKPTS = {
    "arrr": "checkpoints/arrr_L2_K2_arrr_1k_ce_step1000.pt",
    "ar": "checkpoints/ar_L2_K2_ar_step1000.pt",  # optional; skipped if absent
}

# ---------------------------------------------------------------------------
# RULER task selection. The spec's friendly names map onto ruler-full task ids.
# ---------------------------------------------------------------------------
TASK_ALIASES = {
    "niah_single_1": "niah_single_1",
    "niah_single_2": "niah_single_2",
    "niah_single_3": "niah_single_3",
    "niah_multikey_1": "niah_multikey_1",
    "niah_multikey_2": "niah_multikey_2",
    "niah_multikey_3": "niah_multikey_3",
    "niah_multiquery": "niah_multiquery",
    "niah_multivalue": "niah_multivalue",
    "variable_tracking": "vt", "vt": "vt",
    "common_words_extraction": "cwe", "cwe": "cwe",
    "frequent_words_extraction": "fwe", "fwe": "fwe",
    "qa_1": "qa_1", "qa_2": "qa_2",
}
DEFAULT_TASKS = [
    "niah_single_1", "niah_single_2", "niah_multikey_1",
    "niah_multivalue", "vt", "cwe",
]
ALL_TASKS = [
    "niah_single_1", "niah_single_2", "niah_single_3",
    "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
    "niah_multiquery", "niah_multivalue", "vt", "cwe", "fwe", "qa_1", "qa_2",
]

# Per-task generation budgets (match att-hub ruler defaults; --max-new-tokens overrides).
MAX_NEW_TOKENS = {"qa_1": 32, "qa_2": 32, "vt": 30, "cwe": 120, "fwe": 50}
DEFAULT_MAX_NEW = 128

# ---------------------------------------------------------------------------
# Prompt decomposition (vendored from Prism-Test ruler64k — query-agnostic split).
# ---------------------------------------------------------------------------
_USER_HEAD_RE = re.compile(r"<\|im_start\|>user\n")
_ASSISTANT_TAIL = "<|im_end|>\n<|im_start|>assistant\n"
QUESTION_ANCHORS = {
    "niah_single_1": "What is the special magic number for",
    "niah_single_2": "What is the special magic number for",
    "niah_multikey_1": "What is the special magic number for",
    "niah_multikey_2": "What is the special magic number for",
    "niah_single_3": "What is the special magic uuid for",
    "niah_multikey_3": "What is the special magic uuid for",
    "niah_multiquery": "What are all the special magic numbers for",
    "niah_multivalue": "What are all the special magic numbers for",
    "qa_1": "Answer the question based on the given documents",
    "qa_2": "Answer the question based on the given documents",
    "vt": "Question: Find all variables that are assigned the value",
    "cwe": "Question: What are the 10 most common words",
    "fwe": "Question: Do not provide any explanation",
}


def _split_prompt(prompt: str, task: str) -> Optional[Tuple[str, str, str]]:
    """Split a ruler-full chat-templated prompt into (context, question, answer_prefix).
    Returns None if it cannot be split (caller falls back to whole-prompt context)."""
    body = prompt
    heads = list(_USER_HEAD_RE.finditer(body))
    if heads:
        body = body[heads[-1].end():]
    tail = body.rfind(_ASSISTANT_TAIL)
    if tail != -1:
        body = body[:tail]
    else:
        body = body.rstrip()
        if body.endswith("<|im_end|>"):
            body = body[: -len("<|im_end|>")]
    anchor = QUESTION_ANCHORS.get(task)
    if not anchor or anchor not in body:
        return None
    qi = body.rfind(anchor)
    context, q_block = body[:qi], body[qi:]
    sp = q_block.find("? ")
    if sp != -1:
        return context, q_block[: sp + 2], q_block[sp + 2:]
    ap = q_block.find("Answer:")
    if ap == -1:
        return None
    return context, q_block[:ap], q_block[ap:]


# ---------------------------------------------------------------------------
# Answer parsing + scoring (vendored from Prism-Test common.py / ruler scorers).
# ---------------------------------------------------------------------------
def parse_answers(value) -> List[str]:
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        try:
            listed = value.tolist()
            if isinstance(listed, (list, tuple)):
                return [str(v) for v in listed]
        except Exception:
            pass
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    if value is None:
        return []
    if isinstance(value, str):
        s = value.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, (list, tuple)):
                    return [str(v) for v in parsed]
            except Exception:
                quoted = re.findall(r"'([^']*)'|\"([^\"]*)\"", s)
                recovered = [a or b for a, b in quoted if (a or b)]
                if recovered:
                    return recovered
        return [s]
    return [str(value)]


_CTRL = re.compile(r"[\x00-\x1f]")


def _clean(text) -> str:
    return _CTRL.sub("", str(text).strip()).strip().lower()


def _match_part(pred: str, refs: List[str]) -> float:
    """RULER substring match — ANY ref present (used for qa)."""
    return max((1.0 if r and r in pred else 0.0) for r in refs) if refs else 0.0


def _match_all(pred: str, refs: List[str]) -> float:
    """RULER substring match — fraction of refs present (niah/vt/cwe/fwe)."""
    return sum(1.0 if r and r in pred else 0.0 for r in refs) / len(refs) if refs else 0.0


def _exact(pred: str, refs: List[str]) -> float:
    return 1.0 if any(pred == r for r in refs) else 0.0


def score_rows(rows: List[dict]) -> Dict[str, object]:
    """RULER string-match per task + overall, plus normalized exact-match per task."""
    by_task: Dict[str, List[dict]] = {}
    for r in rows:
        by_task.setdefault(r["task"], []).append(r)
    task_scores, task_exact, task_n = {}, {}, {}
    for task, trows in by_task.items():
        match_fn = _match_part if str(task).split("_")[0] == "qa" else _match_all
        sm, em = [], []
        for r in trows:
            pred = _clean(r["prediction"])
            refs = [_clean(x) for x in parse_answers(r["answer"])]
            sm.append(match_fn(pred, refs))
            em.append(_exact(pred, refs))
        task_scores[task] = round(100 * sum(sm) / len(sm), 2) if sm else 0.0
        task_exact[task] = round(100 * sum(em) / len(em), 2) if em else 0.0
        task_n[task] = len(trows)
    overall = round(sum(task_scores.values()) / len(task_scores), 2) if task_scores else 0.0
    overall_em = round(sum(task_exact.values()) / len(task_exact), 2) if task_exact else 0.0
    return {
        "overall_score": overall,
        "overall_exact_match": overall_em,
        "task_scores": task_scores,
        "task_exact_match": task_exact,
        "task_n": task_n,
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_examples(context_len: int, tasks: List[str], max_examples: Optional[int]) -> List[dict]:
    """Stream tonychenxyz/ruler-full, keep rows whose category carries the
    `_{context_len}` marker and whose task is requested; cap to max_examples per task."""
    from datasets import load_dataset

    wanted = set(tasks)
    per_task: Dict[str, int] = {t: 0 for t in wanted}
    marker = f"_{context_len}"
    rows: List[dict] = []
    for variant in ["plain", "memwrap"]:
        ds = load_dataset("tonychenxyz/ruler-full", variant, split="validation", streaming=True)
        for sample in ds:
            extra = sample.get("extra_info") or {}
            category = str(sample.get("category", ""))
            # Prefer the explicit integer field; fall back to the category marker.
            cl = extra.get("context_length")
            if cl is not None:
                if int(cl) != context_len:
                    continue
            elif marker not in category:
                continue
            task = str(extra.get("ruler_task", ""))
            if task not in wanted:
                suffix = category.split("/")[-1]
                if suffix.endswith(marker):
                    task = suffix[: -len(marker)]
            if task not in wanted:
                continue
            if max_examples is not None and per_task[task] >= max_examples:
                continue
            gt = extra.get("ground_truth") or {}
            answer = gt.get("answers", "") if isinstance(gt, dict) else gt
            prompt = str(sample.get("prompt", ""))
            split = _split_prompt(prompt, task)
            if split is not None:
                context, question, answer_prefix = split
            else:
                context, question, answer_prefix = prompt, "", ""
            rows.append({
                "task": task, "context": context, "question": question,
                "answer_prefix": answer_prefix, "answer": answer,
                "context_length": context_len, "variant": variant,
            })
            per_task[task] += 1
        if max_examples is not None and all(per_task[t] >= max_examples for t in wanted):
            break
    return rows


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------
def resolve_ckpt(path: str) -> str:
    cands = [path, os.path.join(HERE, path),
             os.path.join(HERE, "checkpoints", os.path.basename(path))]
    for c in cands:
        if os.path.isfile(c):
            return c
    raise FileNotFoundError(f"checkpoint not found; tried {cands}")


def build_model(kind: str, checkpoint: Optional[str], attn_impl: str, device: str,
                decode_mode: str = "cache"):
    """Returns (model, is_hybrid, meta). Teacher = native Llama. Hybrid = RACE layers
    swapped in and a trained checkpoint loaded (pattern/L/K read from the checkpoint)."""
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        TEACHER_MODEL, dtype=torch.bfloat16, attn_implementation=attn_impl).to(device).eval()
    model.requires_grad_(False)

    if kind == "teacher":
        return model, False, {"model": "teacher", "base": TEACHER_MODEL, "attn_impl": attn_impl}

    # hybrid student
    from hybrid import build_race_modules, convert_to_hybrid, make_replace_pred
    from race_llama_attention import RaceLlamaAttention  # noqa: F401 (ensures import ok)

    ckpath = resolve_ckpt(checkpoint or DEFAULT_CKPTS.get(kind, ""))
    ck = torch.load(ckpath, map_location=device, weights_only=False)
    cfg = ck.get("config", {})
    pattern = cfg.get("pattern", "ARRR" if kind == "arrr" else "AR")
    L, K, M = cfg.get("L", 2), cfg.get("K", 2), cfg.get("M", 1)
    # make_replace_pred reconstructs the SAME replaced-layer set the trainer used, including
    # the custom 'edges:P' pattern (not just repeating A/R strings).
    pred = make_replace_pred(pattern, model.config.num_hidden_layers)
    # Cache decode reads sequence length + causal-mask size from KV-cache slot 0 only.
    # RACE layers don't write the cache, so layer 0 MUST be a softmax (cache-writing)
    # layer or decode would silently use position 0 / a 1-length mask. ARRR/AR keep
    # layer 0 softmax (pattern[0]=='A'); guard against an 'R'-leading pattern.
    if decode_mode == "cache" and pred(0):
        raise ValueError(
            f"decode_mode='cache' requires layer 0 to be softmax (pattern[0]=='A'); "
            f"pattern={pattern!r} makes layer 0 RACE. Use --decode-mode recompute.")
    race = build_race_modules(model, L=L, Kbits=K, M=M, device=device, replace_pred=pred)
    # planes_T/protos_T are buffers here; a trainhash checkpoint stored them as
    # Parameters, but state-dict keys are identical so strict load copies the values
    # into the buffers cleanly either way.
    race.load_state_dict(ck["race_state"], strict=True)
    for m in race.values():
        m.to(torch.bfloat16)  # inference regime (matches bench_forward)
        if decode_mode == "cache":
            m.enable_decode_cache(True)  # incremental RACE state -> KV cache on softmax layers
    convert_to_hybrid(model, race)
    # Partial-unfreeze checkpoints (--unfreeze mlp) also carry the trained base weights; load them
    # or the eval silently reverts the MLPs to stock Llama (would erase the MMLU recovery).
    n_base = 0
    if ck.get("base_state"):
        model.load_state_dict({k: v.to(device) for k, v in ck["base_state"].items()}, strict=False)
        n_base = len(ck["base_state"])
        print(f"loaded base_state ({n_base} tensors) from checkpoint")
    model.eval()
    replaced = sorted(int(k) for k in race.keys())
    meta = {"model": kind, "base": TEACHER_MODEL, "attn_impl": attn_impl,
            "checkpoint": ckpath, "ckpt_step": ck.get("step"),
            "pattern": pattern, "L": L, "K": K, "M": M, "S": L * (1 << K),
            "race_layers": replaced, "n_race": len(replaced),
            "n_softmax": model.config.num_hidden_layers - len(replaced),
            "unfreeze": cfg.get("unfreeze", "none"), "n_base_loaded": n_base,
            "decode_mode": decode_mode}
    return model, True, meta


def build_prompt_ids(tok, context: str, question: str, answer_prefix: str, device: str):
    """Re-template the query-agnostic (context, question, answer_prefix) for Llama chat."""
    msg = [{"role": "user", "content": context + question}]
    s = tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    s = s + answer_prefix
    ids = tok(s, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    return ids


def _is_looping(gen: List[int], max_period: int = 6) -> bool:
    """True if the tail of `gen` is a repeated cycle of period <= max_period (degenerate
    output, e.g. ', and, and, ...'). Bounds decode cost on collapsed generations without
    affecting coherent ones (which keep producing fresh tokens until EOS)."""
    n = len(gen)
    for p in range(1, max_period + 1):
        if n >= 4 * p and gen[-2 * p:] == gen[-4 * p:-2 * p]:
            return True
    return False


@torch.no_grad()
def generate(model, ids, use_cache: bool, max_new: int, eos_ids, device):
    """Greedy decode with separated prefill / decode timing. use_cache=True drives the
    incremental path (teacher KV cache; hybrid = softmax KV cache + RACE running state);
    use_cache=False re-runs the full sequence each step (hybrid recompute baseline)."""
    # Explicitly clear any RACE running state so it never leaks across sequences
    # (don't rely on the prefill overwrite alone — guards a 1-token-prompt edge case).
    for m in model.modules():
        if hasattr(m, "reset_decode_state"):
            m.reset_decode_state()
    n_prompt = ids.shape[1]

    # logits_to_keep=1: only the last position's logits are needed for greedy decode.
    # Without it the lm_head materializes a [B, T, vocab] tensor (~8GB at 32K) that would
    # dominate the peak-memory metric and risk OOM.
    torch.cuda.synchronize(); t0 = time.perf_counter()
    out = model(input_ids=ids, use_cache=use_cache, logits_to_keep=1)
    torch.cuda.synchronize()
    prefill_ms = (time.perf_counter() - t0) * 1e3
    past = out.past_key_values if use_cache else None
    next_id = out.logits[:, -1, :].argmax(-1, keepdim=True)

    gen = [int(next_id)]
    cur = torch.cat([ids, next_id], dim=1)
    decode_ms = 0.0
    n_decode_steps = 0
    stop_reason = "max_new_tokens"
    for _ in range(max_new - 1):
        if int(next_id) in eos_ids:
            stop_reason = "eos"; break
        if _is_looping(gen):
            stop_reason = "repetition"; break
        torch.cuda.synchronize(); td = time.perf_counter()
        if use_cache:
            out = model(input_ids=next_id, past_key_values=past, use_cache=True, logits_to_keep=1)
            past = out.past_key_values
        else:
            out = model(input_ids=cur, use_cache=False, logits_to_keep=1)
        torch.cuda.synchronize()
        decode_ms += (time.perf_counter() - td) * 1e3
        n_decode_steps += 1
        next_id = out.logits[:, -1, :].argmax(-1, keepdim=True)
        gen.append(int(next_id))
        cur = torch.cat([cur, next_id], dim=1)
    # strip trailing eos for cleaner decode
    while gen and gen[-1] in eos_ids:
        gen.pop()
    return gen, n_prompt, prefill_ms, decode_ms, n_decode_steps, stop_reason


# ---------------------------------------------------------------------------
# Run one (model, context) evaluation
# ---------------------------------------------------------------------------
def run_eval(args) -> Optional[dict]:
    from transformers import AutoTokenizer

    device = "cuda"
    torch.manual_seed(args.seed)
    tasks = [TASK_ALIASES[t] for t in args.tasks]
    tag = args.model
    cl_tag = f"{args.context_len // 1024}k"

    # graceful skip for an optional student with no checkpoint
    if args.model in ("ar", "arrr"):
        ckpath = args.checkpoint or DEFAULT_CKPTS.get(args.model, "")
        try:
            resolve_ckpt(ckpath)
        except FileNotFoundError:
            print(f"[skip] no checkpoint for model={args.model} ({ckpath}); skipping.")
            return None

    print(f"=== RULER eval | model={args.model} | context={args.context_len} "
          f"| tasks={tasks} | max_examples={args.max_examples} ===")
    print("GPU:", torch.cuda.get_device_name(0))

    tok = AutoTokenizer.from_pretrained(TEACHER_MODEL)
    eos_ids = {tok.eos_token_id}
    for t in ("<|eot_id|>", "<|end_of_text|>"):
        tid = tok.convert_tokens_to_ids(t)
        if tid is not None and tid >= 0:
            eos_ids.add(tid)

    examples = load_examples(args.context_len, tasks, args.max_examples)
    print(f"loaded {len(examples)} examples across {len(set(e['task'] for e in examples))} tasks")
    if not examples:
        raise SystemExit(f"no examples for context_len={args.context_len}, tasks={tasks}")

    model, is_hybrid, meta = build_model(args.model, args.checkpoint, args.attn_impl,
                                         device, decode_mode=args.decode_mode)
    # use_cache: teacher always; hybrid only in cache mode (RACE incremental state).
    use_cache = (not is_hybrid) or (args.decode_mode == "cache")
    if is_hybrid:
        print(f"hybrid: pattern={meta['pattern']} S={meta['S']} "
              f"{meta['n_race']}/{meta['n_race']+meta['n_softmax']} RACE layers; "
              f"ckpt step {meta['ckpt_step']}; decode={args.decode_mode} (use_cache={use_cache})")

    torch.cuda.reset_peak_memory_stats()
    rows: List[dict] = []
    tot_prefill_ms = tot_decode_ms = 0.0
    tot_prompt_tok = tot_gen_tok = 0
    t_start = time.perf_counter()

    for i, ex in enumerate(examples):
        task = ex["task"]
        max_new = args.max_new_tokens or MAX_NEW_TOKENS.get(task, DEFAULT_MAX_NEW)
        ids = build_prompt_ids(tok, ex["context"], ex["question"], ex["answer_prefix"], device)
        gen_ids, n_prompt, prefill_ms, decode_ms, n_steps, stop_reason = generate(
            model, ids, use_cache, max_new, eos_ids, device)
        pred = tok.decode(gen_ids, skip_special_tokens=True)
        tot_prefill_ms += prefill_ms
        tot_decode_ms += decode_ms
        tot_prompt_tok += n_prompt
        tot_gen_tok += len(gen_ids)
        rows.append({
            "task": task, "answer": ex["answer"], "prediction": pred,
            "n_prompt_tok": n_prompt, "n_gen_tok": len(gen_ids),
            "prefill_ms": round(prefill_ms, 2), "decode_ms": round(decode_ms, 2),
            "stop_reason": stop_reason, "variant": ex["variant"],
        })
        if (i + 1) % 10 == 0 or i == len(examples) - 1:
            print(f"  [{i+1}/{len(examples)}] {task:18s} prompt={n_prompt:>6} "
                  f"gen={len(gen_ids):>3} prefill={prefill_ms:7.1f}ms "
                  f"decode={decode_ms:8.1f}ms pred={pred[:40]!r}")

    total_runtime_s = time.perf_counter() - t_start
    peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9
    quality = score_rows(rows)

    decode_s = tot_decode_ms / 1e3
    prefill_s = tot_prefill_ms / 1e3
    speed = {
        "total_runtime_s": round(total_runtime_s, 2),
        "prefill_latency_ms_mean": round(tot_prefill_ms / len(rows), 2),
        "decode_latency_ms_mean": round(tot_decode_ms / len(rows), 2),
        "decode_latency_ms_per_token": round(tot_decode_ms / max(tot_gen_tok - len(rows), 1), 3),
        "prefill_throughput_tok_s": round(tot_prompt_tok / prefill_s, 1) if prefill_s else None,
        "decode_throughput_tok_s": round((tot_gen_tok - len(rows)) / decode_s, 2) if decode_s else None,
        "avg_tokens_per_sec": round(tot_gen_tok / total_runtime_s, 2) if total_runtime_s else None,
        "total_prompt_tokens": tot_prompt_tok,
        "total_generated_tokens": tot_gen_tok,
    }

    result = {
        "model": args.model, "context_len": args.context_len,
        "n_examples": len(rows), "tasks": tasks,
        "meta": meta, "quality": quality, "speed": speed,
        "peak_memory_gb": round(peak_mem_gb, 3),
        "rows": rows if args.save_predictions else [],
    }

    os.makedirs(RES, exist_ok=True)
    out_path = os.path.join(RES, f"ruler_{tag}_{cl_tag}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nOVERALL RULER score = {quality['overall_score']}  "
          f"(exact-match {quality['overall_exact_match']})  "
          f"peak_mem={peak_mem_gb:.1f}GB  runtime={total_runtime_s:.0f}s")
    print(f"wrote {out_path}")
    return result


# ---------------------------------------------------------------------------
# Plots + summary table (read whatever results/ruler_*.json exist)
# ---------------------------------------------------------------------------
def _load_results() -> List[dict]:
    out = []
    if not os.path.isdir(RES):
        return out
    for fn in sorted(os.listdir(RES)):
        m = re.match(r"ruler_(\w+?)_(\d+)k\.json$", fn)
        if m:
            with open(os.path.join(RES, fn)) as f:
                out.append(json.load(f))
    return out


def make_plots(results: List[dict]):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not results:
        print("no results to plot")
        return
    os.makedirs(RES, exist_ok=True)
    models = sorted({r["model"] for r in results})
    ctxs = sorted({r["context_len"] for r in results})
    colors = {m: c for m, c in zip(models, plt.cm.tab10.colors)}

    def get(model, ctx):
        for r in results:
            if r["model"] == model and r["context_len"] == ctx:
                return r
        return None

    # 1. accuracy: grouped bars (model × context) of overall RULER score
    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.8 / max(len(models), 1)
    for j, m in enumerate(models):
        ys = [(get(m, c) or {}).get("quality", {}).get("overall_score", 0) for c in ctxs]
        xs = [i + j * width for i in range(len(ctxs))]
        ax.bar(xs, ys, width, label=m, color=colors[m])
    ax.set_xticks([i + width * (len(models) - 1) / 2 for i in range(len(ctxs))])
    ax.set_xticklabels([f"{c//1024}K" for c in ctxs])
    ax.set_ylabel("Overall RULER score"); ax.set_title("RULER accuracy"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(RES, "ruler_accuracy.png"), dpi=120); plt.close(fig)

    # 2. speed: prefill + decode throughput
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, key, title in [(axes[0], "prefill_throughput_tok_s", "Prefill throughput (tok/s)"),
                           (axes[1], "decode_throughput_tok_s", "Decode throughput (tok/s)")]:
        for j, m in enumerate(models):
            ys = [(get(m, c) or {}).get("speed", {}).get(key) or 0 for c in ctxs]
            xs = [i + j * width for i in range(len(ctxs))]
            ax.bar(xs, ys, width, label=m, color=colors[m])
        ax.set_xticks([i + width * (len(models) - 1) / 2 for i in range(len(ctxs))])
        ax.set_xticklabels([f"{c//1024}K" for c in ctxs])
        ax.set_title(title); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(RES, "ruler_speed.png"), dpi=120); plt.close(fig)

    # 3. memory
    fig, ax = plt.subplots(figsize=(8, 5))
    for j, m in enumerate(models):
        ys = [(get(m, c) or {}).get("peak_memory_gb", 0) for c in ctxs]
        xs = [i + j * width for i in range(len(ctxs))]
        ax.bar(xs, ys, width, label=m, color=colors[m])
    ax.set_xticks([i + width * (len(models) - 1) / 2 for i in range(len(ctxs))])
    ax.set_xticklabels([f"{c//1024}K" for c in ctxs])
    ax.set_ylabel("Peak GPU memory (GB)"); ax.set_title("RULER peak memory"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(RES, "ruler_memory.png"), dpi=120); plt.close(fig)

    # 4. context scaling: overall score vs context length (lines per model)
    fig, ax = plt.subplots(figsize=(8, 5))
    for m in models:
        pts = [(c, get(m, c)) for c in ctxs]
        pts = [(c, r["quality"]["overall_score"]) for c, r in pts if r]
        if pts:
            ax.plot([p[0] // 1024 for p in pts], [p[1] for p in pts],
                    "o-", label=m, color=colors[m])
    ax.set_xlabel("Context length (K tokens)"); ax.set_ylabel("Overall RULER score")
    ax.set_title("Context-length scaling"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(RES, "ruler_context_scaling.png"), dpi=120); plt.close(fig)
    print(f"wrote 4 plots to {RES}")


def make_summary(results: List[dict]):
    lines = ["| Model | Context | Avg Score | Exact Match | Prefill (tok/s) | Decode (tok/s) | Peak Memory (GB) |",
             "| ----- | ------- | --------- | ----------- | --------------- | -------------- | ---------------- |"]
    order = {"teacher": 0, "ar": 1, "arrr": 2}
    for r in sorted(results, key=lambda r: (r["context_len"], order.get(r["model"], 9))):
        q, s = r["quality"], r["speed"]
        lines.append(
            f"| {r['model']} | {r['context_len']//1024}K | {q['overall_score']} | "
            f"{q['overall_exact_match']} | {s.get('prefill_throughput_tok_s')} | "
            f"{s.get('decode_throughput_tok_s')} | {r['peak_memory_gb']} |")
    table = "\n".join(lines)
    with open(os.path.join(RES, "ruler_summary.md"), "w") as f:
        f.write("# RULER evaluation summary\n\n" + table + "\n")
    print("\n" + table)
    print(f"\nwrote {os.path.join(RES, 'ruler_summary.md')}")


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="RULER long-context evaluation (teacher / AR / ARRR).")
    p.add_argument("--model", choices=["teacher", "ar", "arrr"], help="which model to evaluate")
    p.add_argument("--checkpoint", default=None, help="RACE checkpoint .pt (overrides default for ar/arrr)")
    p.add_argument("--context-len", type=int, default=32768, help="32768 or 65536")
    p.add_argument("--max-examples", default="20",
                   help="per-task example cap: integer or 'all'")
    p.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS,
                   help=f"RULER tasks (friendly or id). Choices: {sorted(TASK_ALIASES)}")
    p.add_argument("--max-new-tokens", type=int, default=0, help="0 = per-task default")
    p.add_argument("--attn-impl", default="sdpa", help="attn impl for softmax layers (sdpa/flash_attention_2)")
    p.add_argument("--decode-mode", choices=["cache", "recompute"], default="cache",
                   help="hybrid decode: 'cache' = softmax KV cache + RACE incremental state "
                        "(fast); 'recompute' = full-sequence re-run per token (baseline)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-predictions", action="store_true", default=True)
    p.add_argument("--no-save-predictions", dest="save_predictions", action="store_false")
    p.add_argument("--plot", action="store_true", help="render plots + summary from existing results/ and exit")
    args = p.parse_args()

    if args.plot:
        results = _load_results()
        make_plots(results)
        make_summary(results)
        return

    if not args.model:
        p.error("--model is required (or use --plot)")
    args.max_examples = None if str(args.max_examples).lower() == "all" else int(args.max_examples)
    bad = [t for t in args.tasks if t not in TASK_ALIASES]
    if bad:
        p.error(f"unknown tasks {bad}; choices: {sorted(TASK_ALIASES)}")

    run_eval(args)
    # best-effort refresh of plots/summary across whatever results now exist
    try:
        results = _load_results()
        make_plots(results)
        make_summary(results)
    except Exception as e:
        print(f"[warn] plot/summary refresh skipped: {e}")


if __name__ == "__main__":
    main()
