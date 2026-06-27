"""Multiple-choice loglikelihood eval — MMLU / HellaSwag / Winogrande — for the
RACE-hybrid students vs the Llama-3.2-3B-Instruct teacher.

WHY loglikelihood (not generation): these tasks are scored by comparing the
loglikelihood the model assigns to each answer choice, NOT by free generation.
That deliberately BYPASSES the free-generation collapse the distilled hybrids
suffer (RULER=0, repetition) — so it both (a) provides the requested benchmark
deliverable and (b) is a fast diagnostic separating "ranking/calibration quality"
from "free-generation quality".

Mirrors eval_ruler.py: reuses eval_ruler.build_model() (loads pattern/L/K from the
checkpoint config; returns a standard HF AutoModelForCausalLM with RACE layers
swapped in), the tokenizer, and the results/*.json + summary-table convention. It
does NOT train or modify the model.

Scoring (standard MC conventions, 0-shot):
  * MMLU       — letter scoring: logprob of " A"/" B"/" C"/" D" after "Answer:".  metric=acc.
  * HellaSwag  — full-continuation loglik of each ending; metric=acc_norm (per-char
                 length-normalized, the standard) plus raw acc.
  * Winogrande — fill the blank with each option, score loglik of the suffix that
                 follows the blank (partial scoring); metric=acc.

Examples
--------
  # teacher ceiling (run FIRST), all three tasks, 200 examples each
  python eval_choice.py --model teacher --tasks mmlu hellaswag winogrande --max-examples 200
  # the AR/S24 hybrid
  python eval_choice.py --model ar --checkpoint checkpoints/ar_L3_K3_abl2_ar_s24h_step2000.pt \
      --tasks mmlu hellaswag winogrande --max-examples 200
  # render the cross-model summary table from existing results/
  python eval_choice.py --summary
"""
from __future__ import annotations

import os
import re
import sys
import json
import argparse
from typing import Dict, List, Optional, Tuple

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
RES = os.path.join(HERE, "results")

import eval_ruler  # noqa: E402  (build_model + result conventions)

TEACHER_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
TASKS = ("mmlu", "hellaswag", "winogrande")
MMLU_LETTERS = [" A", " B", " C", " D"]


# --------------------------------------------------------------------------- data
def _hs_preprocess(text: str) -> str:
    """HellaSwag text normalization (matches lm-eval-harness)."""
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    text = text.replace("  ", " ")
    return text


_MMLU_FEWSHOT_CACHE = {}


def _mmlu_fewshot_by_subject(n_shot, seed=0):
    """Per-subject MMLU few-shot prefixes from the `dev` split (the canonical 5-exemplars/subject
    set used by the standard MMLU protocol). Returns {subject_raw: prefix_str}. Cached."""
    if n_shot <= 0:
        return {}
    key = (n_shot, seed)
    if key in _MMLU_FEWSHOT_CACHE:
        return _MMLU_FEWSHOT_CACHE[key]
    from datasets import load_dataset
    dev = load_dataset("cais/mmlu", "all", split="dev")
    by_sub = {}
    for r in dev:
        by_sub.setdefault(str(r["subject"]), []).append(r)
    out = {}
    for subj, rs in by_sub.items():
        block = ""
        for r in rs[:n_shot]:
            q = str(r["question"]).strip()
            ch = list(r["choices"])
            letter = "ABCD"[int(r["answer"])]
            block += q + "\n" + "".join(f"{L}. {c}\n" for L, c in zip("ABCD", ch))
            block += f"Answer: {letter}\n\n"
        out[subj] = block
    _MMLU_FEWSHOT_CACHE[key] = out
    return out


def load_task(task: str, max_examples: Optional[int], seed: int, num_fewshot: int = 0) -> List[dict]:
    """Return a list of normalized examples for `task`. Each example is a dict with a
    `context` and a list of `choices` (each {text}) plus `gold` (int index)."""
    from datasets import load_dataset

    rows: List[dict] = []
    if task == "mmlu":
        ds = load_dataset("cais/mmlu", "all", split="test")
        ds = ds.shuffle(seed=seed)
        if max_examples:
            ds = ds.select(range(min(max_examples, len(ds))))
        fewshot = _mmlu_fewshot_by_subject(num_fewshot, seed)   # standard 5-shot prefix per subject
        for r in ds:
            subj_raw = str(r["subject"])
            subj = subj_raw.replace("_", " ")
            q = str(r["question"]).strip()
            choices = list(r["choices"])
            header = (f"The following are multiple choice questions (with answers) about "
                      f"{subj}.\n\n")
            body = q + "\n" + "".join(f"{L}. {c}\n" for L, c in zip("ABCD", choices)) + "Answer:"
            ctx = header + fewshot.get(subj_raw, "") + body
            rows.append({"task": task, "context": ctx, "subject": subj_raw,
                         "choices": [{"text": L} for L in MMLU_LETTERS],
                         "gold": int(r["answer"]), "score": "letter"})
    elif task == "hellaswag":
        ds = load_dataset("Rowan/hellaswag", split="validation")
        ds = ds.shuffle(seed=seed)
        if max_examples:
            ds = ds.select(range(min(max_examples, len(ds))))
        for r in ds:
            ctx_full = r["ctx_a"] + " " + r["ctx_b"].capitalize() if r.get("ctx_b") else r["ctx"]
            query = _hs_preprocess(r["activity_label"] + ": " + ctx_full)
            endings = [_hs_preprocess(e) for e in r["endings"]]
            rows.append({"task": task, "context": query,
                         "choices": [{"text": " " + e} for e in endings],
                         "gold": int(r["label"]), "score": "norm"})
    elif task == "winogrande":
        # datasets 4.x is parquet-backed (no dataset scripts); allenai/winogrande is the
        # canonical parquet repo. Fall back to the short id if the org-qualified one moves.
        try:
            ds = load_dataset("allenai/winogrande", "winogrande_xl", split="validation")
        except Exception:
            ds = load_dataset("winogrande", "winogrande_xl", split="validation")
        ds = ds.shuffle(seed=seed)
        if max_examples:
            ds = ds.select(range(min(max_examples, len(ds))))
        for r in ds:
            sent = r["sentence"]
            prefix, suffix = sent.split("_", 1)
            # Partial scoring: context = prefix+option, continuation = the suffix that
            # follows the blank. The discriminating signal is P(suffix | prefix+option).
            rows.append({"task": task,
                         "choices": [{"context": prefix + r["option1"], "text": suffix},
                                     {"context": prefix + r["option2"], "text": suffix}],
                         "gold": int(r["answer"]) - 1, "score": "suffix"})
    else:
        raise ValueError(f"unknown task {task}")
    return rows


# --------------------------------------------------------------------------- scoring
@torch.no_grad()
def _reset_race(model):
    for m in model.modules():
        if hasattr(m, "reset_decode_state"):
            m.reset_decode_state()


@torch.no_grad()
def loglik(model, tok, context: str, continuation: str, device) -> Tuple[float, int, int]:
    """Sum log P(continuation | context). Returns (logprob_sum, n_cont_tok, n_cont_chars).
    Continuation tokens are the suffix of tok(context+continuation) past tok(context)."""
    ctx_ids = tok(context, add_special_tokens=True).input_ids
    full_ids = tok(context + continuation, add_special_tokens=True).input_ids
    n_ctx = len(ctx_ids)
    if len(full_ids) <= n_ctx:
        return -1e9, 0, max(1, len(continuation))
    _reset_race(model)
    inp = torch.tensor([full_ids], device=device)
    logits = model(input_ids=inp, use_cache=False).logits[0].float()  # [T, V]
    logp = torch.log_softmax(logits, dim=-1)
    idx = torch.tensor(full_ids[n_ctx:], device=device)              # continuation token ids
    pos = torch.arange(n_ctx - 1, len(full_ids) - 1, device=device)   # predicting positions
    total = logp[pos, idx].sum().item()
    return total, len(full_ids) - n_ctx, max(1, len(continuation))


@torch.no_grad()
def score_letter(model, tok, context: str, choices: List[dict], device) -> List[float]:
    """MMLU fast path: ONE forward on the context; read the first-token logprob of each
    candidate letter at the final position."""
    _reset_race(model)
    ctx_ids = tok(context, add_special_tokens=True).input_ids
    inp = torch.tensor([ctx_ids], device=device)
    logits = model(input_ids=inp, use_cache=False).logits[0, -1].float()  # [V]
    logp = torch.log_softmax(logits, dim=-1)
    out = []
    for ch in choices:
        cand_ids = tok(ch["text"], add_special_tokens=False).input_ids
        out.append(logp[cand_ids[0]].item() if cand_ids else -1e9)
    return out


@torch.no_grad()
def eval_example(model, tok, ex: dict, device) -> Tuple[int, int]:
    """Return (pred_acc, pred_acc_norm) choice indices for one example."""
    mode = ex["score"]
    if mode == "letter":
        scores = score_letter(model, tok, ex["context"], ex["choices"], device)
        pred = int(max(range(len(scores)), key=lambda i: scores[i]))
        return pred, pred
    # full-continuation loglik (hellaswag norm / winogrande suffix)
    raw, norm = [], []
    for ch in ex["choices"]:
        ctx = ch.get("context", ex.get("context", ""))
        lp, _ntok, nchar = loglik(model, tok, ctx, ch["text"], device)
        raw.append(lp)
        norm.append(lp / nchar)
    pred_raw = int(max(range(len(raw)), key=lambda i: raw[i]))
    pred_norm = int(max(range(len(norm)), key=lambda i: norm[i]))
    return pred_raw, pred_norm


# --------------------------------------------------------------------------- run
def run_eval(args):
    device = "cuda"
    torch.manual_seed(args.seed)
    tag = getattr(args, "out_tag", None) or args.model   # distinct output name (avoids ar/ar collision)

    if args.model in ("ar", "arrr") and not args.checkpoint:
        raise SystemExit(f"--checkpoint required for --model {args.model}")

    print(f"=== eval_choice | model={args.model} | tasks={args.tasks} "
          f"| max_examples={args.max_examples} ===")
    print("GPU:", torch.cuda.get_device_name(0))

    model, is_hybrid, meta = eval_ruler.build_model(
        args.model, args.checkpoint, args.attn_impl, device, decode_mode="recompute")
    if is_hybrid:
        print(f"hybrid: pattern={meta['pattern']} S={meta['S']} "
              f"{meta['n_race']}/{meta['n_race']+meta['n_softmax']} RACE; ckpt step {meta['ckpt_step']}")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TEACHER_MODEL)

    torch.cuda.reset_peak_memory_stats()
    results = {}
    for task in args.tasks:
        examples = load_task(task, args.max_examples, args.seed, num_fewshot=args.num_fewshot)
        n = len(examples)
        print(f"[{task}] loaded {n} examples"
              + (f" ({args.num_fewshot}-shot)" if task == "mmlu" and args.num_fewshot else ""))
        acc_hits = norm_hits = 0
        sub = {}  # per-subject accumulator (mmlu)
        for i, ex in enumerate(examples):
            pred_raw, pred_norm = eval_example(model, tok, ex, device)
            ok = int(pred_raw == ex["gold"])
            acc_hits += ok
            norm_hits += int(pred_norm == ex["gold"])
            if "subject" in ex:
                s = sub.setdefault(ex["subject"], [0, 0])
                s[0] += ok; s[1] += 1
            if (i + 1) % 100 == 0 or i == n - 1:
                print(f"  [{i+1}/{n}] acc={100*acc_hits/(i+1):.1f} "
                      f"acc_norm={100*norm_hits/(i+1):.1f}")
        acc = round(100 * acc_hits / n, 2) if n else 0.0
        acc_norm = round(100 * norm_hits / n, 2) if n else 0.0
        primary = acc_norm if task == "hellaswag" else acc
        results[task] = {"n": n, "acc": acc, "acc_norm": acc_norm, "primary": primary}
        if sub:
            results[task]["per_subject"] = {k: round(100 * v[0] / v[1], 1)
                                            for k, v in sorted(sub.items())}
        print(f"[{task}] acc={acc} acc_norm={acc_norm} primary={primary}")

    peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9
    out = {"model": args.model, "meta": meta, "results": results,
           "peak_memory_gb": round(peak_mem_gb, 3), "max_examples": args.max_examples,
           "num_fewshot": args.num_fewshot}
    os.makedirs(RES, exist_ok=True)
    out_path = os.path.join(RES, f"choice_{tag}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {out_path}")
    return out


# --------------------------------------------------------------------------- summary
def make_summary():
    rows = []
    if os.path.isdir(RES):
        for fn in sorted(os.listdir(RES)):
            m = re.match(r"choice_(\w+)\.json$", fn)
            if m:
                with open(os.path.join(RES, fn)) as f:
                    rows.append(json.load(f))
    if not rows:
        print("no choice_*.json results found")
        return
    order = {"teacher": 0, "ar": 1, "arrr": 2}
    rows.sort(key=lambda r: order.get(r["model"], 9))
    header = "| Model | MMLU (acc) | HellaSwag (acc_norm) | Winogrande (acc) | Peak GB |"
    sep = "| --- | --- | --- | --- | --- |"
    lines = [header, sep]
    for r in rows:
        res = r["results"]
        def g(t, k="primary"):
            return res[t][k] if t in res else "-"
        lines.append(f"| {r['model']} | {g('mmlu')} | {g('hellaswag')} | "
                     f"{g('winogrande')} | {r.get('peak_memory_gb','-')} |")
    table = "\n".join(lines)
    with open(os.path.join(RES, "choice_summary.md"), "w") as f:
        f.write("# MMLU / HellaSwag / Winogrande summary\n\n" + table + "\n")
    print("\n" + table)
    print(f"\nwrote {os.path.join(RES, 'choice_summary.md')}")


def main():
    p = argparse.ArgumentParser(description="MMLU/HellaSwag/Winogrande loglikelihood eval.")
    p.add_argument("--model", choices=["teacher", "ar", "arrr"], help="which model to evaluate")
    p.add_argument("--checkpoint", default=None, help="RACE checkpoint .pt (required for ar/arrr)")
    p.add_argument("--tasks", nargs="+", default=list(TASKS),
                   help=f"subset of {list(TASKS)}")
    p.add_argument("--max-examples", type=int, default=200,
                   help="per-task example cap (0 = all)")
    p.add_argument("--num-fewshot", type=int, default=0,
                   help="MMLU few-shot exemplars from the dev split (standard=5; 0=0-shot). "
                        "HellaSwag/Winogrande stay 0-shot (their standard).")
    p.add_argument("--attn-impl", default="sdpa")
    p.add_argument("--out-tag", default=None,
                   help="output filename suffix choice_<tag>.json (default = model name); set to "
                        "distinguish multiple checkpoints of the same --model")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--summary", action="store_true",
                   help="render the cross-model summary table from existing results/ and exit")
    args = p.parse_args()

    if args.summary:
        make_summary()
        return
    if not args.model:
        p.error("--model is required (or use --summary)")
    if args.max_examples == 0:
        args.max_examples = None
    bad = [t for t in args.tasks if t not in TASKS]
    if bad:
        p.error(f"unknown tasks {bad}; choices: {list(TASKS)}")

    run_eval(args)
    try:
        make_summary()
    except Exception as e:
        print(f"[warn] summary refresh skipped: {e}")


if __name__ == "__main__":
    main()
