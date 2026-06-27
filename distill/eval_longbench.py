"""LongBench long-context evaluation for the RACE-hybrid vs Llama-3.2-3B-Instruct.

Faithful to official THUDM/LongBench (config/dataset2prompt.json, dataset2maxlen.json,
metrics.py): per-task prompt templates + generation budgets + metrics, middle-truncation to
a token budget, the no-chat-template set for the few-shot/code tasks, first-line truncation
for classification/few-shot tasks, and max-over-references scoring. The official prompt/maxlen
JSONs are vendored under distill/refs/ (lb_dataset2prompt.json, lb_dataset2maxlen.json).

Reuses eval_ruler.build_model (loads teacher OR a RACE-hybrid incl. partial-unfreeze base_state)
and eval_ruler.generate (greedy decode with prefill/decode timing) so hybrid-vs-teacher
throughput + peak memory drop out for free. Writes results/longbench_<model_or_tag>.json.

Examples
--------
  # teacher smoke
  python eval_longbench.py --model teacher --tasks qasper multifieldqa_en triviaqa --max-examples 5
  # a RACE-hybrid checkpoint
  python eval_longbench.py --model ar --checkpoint checkpoints/best/<ckpt>.pt --max-examples 50
  # cross-model summary table
  python eval_longbench.py --summary
"""
from __future__ import annotations

import os
import re
import sys
import json
import string
import zipfile
import argparse
from collections import Counter
from typing import List, Optional

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
RES = os.path.join(HERE, "results")
CFG = os.path.join(HERE, "refs")

import eval_ruler  # noqa: E402  (build_model + generate + result conventions)

TEACHER_MODEL = "meta-llama/Llama-3.2-3B-Instruct"

# Official vendored config (THUDM/LongBench).
PROMPT = json.load(open(os.path.join(CFG, "lb_dataset2prompt.json")))
MAXLEN = json.load(open(os.path.join(CFG, "lb_dataset2maxlen.json")))

# English long-context subset (default). All present in the official config.
DEFAULT_TASKS = [
    "narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa",
    "gov_report", "qmsum", "multi_news", "trec", "triviaqa", "samsum",
    "passage_count", "passage_retrieval_en", "lcc", "repobench-p",
]
# Official: these tasks are NOT wrapped in the chat template (few-shot / code completion).
NO_CHAT = {"trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"}
# Official: prediction truncated to its first line for these.
FIRST_LINE = {"trec", "triviaqa", "samsum", "lsht"}


# --------------------------------------------------------------------------- metrics (verbatim)
def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def _f1(pred_tokens, gt_tokens):
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return (2 * precision * recall) / (precision + recall)


def qa_f1_score(prediction, ground_truth, **kw):
    return _f1(normalize_answer(prediction).split(), normalize_answer(ground_truth).split())


def rouge_score(prediction, ground_truth, **kw):
    from rouge import Rouge
    try:
        scores = Rouge().get_scores([prediction], [ground_truth], avg=True)
    except Exception:
        return 0.0
    return scores["rouge-l"]["f"]


def classification_score(prediction, ground_truth, **kw):
    all_classes = kw.get("all_classes") or []
    em = [c for c in all_classes if c in prediction]
    for m in list(em):
        if m in ground_truth and m != ground_truth:
            em.remove(m)
    return (1.0 / len(em)) if ground_truth in em else 0.0


def retrieval_score(prediction, ground_truth, **kw):
    matches = re.findall(r"Paragraph (\d+)", ground_truth)
    if not matches:
        return 0.0
    gt_id = matches[0]
    nums = re.findall(r"\d+", prediction)
    right = sum(1 for n in nums if str(n) == str(gt_id))
    return 0.0 if not nums else right / len(nums)


def count_score(prediction, ground_truth, **kw):
    nums = re.findall(r"\d+", prediction)
    right = sum(1 for n in nums if str(n) == str(ground_truth))
    return 0.0 if not nums else right / len(nums)


def code_sim_score(prediction, ground_truth, **kw):
    from fuzzywuzzy import fuzz
    line = ""
    for ln in prediction.lstrip("\n").split("\n"):
        if ("`" not in ln) and ("#" not in ln) and ("//" not in ln):
            line = ln
            break
    return fuzz.ratio(line, ground_truth) / 100.0


DATASET2METRIC = {
    "narrativeqa": qa_f1_score, "qasper": qa_f1_score, "multifieldqa_en": qa_f1_score,
    "hotpotqa": qa_f1_score, "2wikimqa": qa_f1_score, "musique": qa_f1_score,
    "gov_report": rouge_score, "qmsum": rouge_score, "multi_news": rouge_score,
    "trec": classification_score, "triviaqa": qa_f1_score, "samsum": rouge_score,
    "passage_count": count_score, "passage_retrieval_en": retrieval_score,
    "lcc": code_sim_score, "repobench-p": code_sim_score,
}


# --------------------------------------------------------------------------- data + prompt
_LB_ZIP = {"path": None}


def _longbench_zip():
    """Canonical LongBench data.zip (per-task JSONL). datasets 4.x can't run the script-based
    THUDM/LongBench loader, so we read the official zip directly (cached after first fetch)."""
    if _LB_ZIP["path"] is None:
        from huggingface_hub import hf_hub_download
        _LB_ZIP["path"] = hf_hub_download("THUDM/LongBench", "data.zip", repo_type="dataset")
    return _LB_ZIP["path"]


def load_examples(task, max_examples):
    rows = []
    with zipfile.ZipFile(_longbench_zip()) as z:
        with z.open(f"data/{task}.jsonl") as f:
            for i, line in enumerate(f):
                if max_examples and i >= max_examples:
                    break
                rows.append(json.loads(line))
    return rows


def _middle_truncate(tok, text, max_tokens):
    """Official LongBench middle-truncation: keep the head and tail when over budget."""
    ids = tok(text, add_special_tokens=False).input_ids
    if len(ids) <= max_tokens:
        return text, len(ids)
    half = max_tokens // 2
    trimmed = tok.decode(ids[:half], skip_special_tokens=True) + \
        tok.decode(ids[-half:], skip_special_tokens=True)
    return trimmed, max_tokens


def make_input_ids(tok, task, ex, device, max_prompt_tokens):
    prompt = PROMPT[task].format(input=ex.get("input", ""), context=ex.get("context", ""))
    prompt, _ = _middle_truncate(tok, prompt, max_prompt_tokens)
    if task in NO_CHAT:
        s, add_special = prompt, True                 # raw completion (+BOS)
    else:
        s = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                    tokenize=False, add_generation_prompt=True)
        add_special = False                            # chat template already has BOS
    return tok(s, return_tensors="pt", add_special_tokens=add_special).input_ids.to(device)


def score_one(task, pred, answers, all_classes):
    metric = DATASET2METRIC[task]
    if task in FIRST_LINE:
        pred = pred.lstrip("\n").split("\n")[0]
    return max((metric(pred, gt, all_classes=all_classes) for gt in answers), default=0.0)


# --------------------------------------------------------------------------- run
def run_eval(args):
    from transformers import AutoTokenizer
    device = "cuda"
    tag = args.out_tag or args.model
    tasks = [t for t in args.tasks if t in DATASET2METRIC]
    print(f"=== LongBench | model={args.model} | tasks={tasks} | max_examples={args.max_examples} ===")
    print("GPU:", torch.cuda.get_device_name(0))

    model, is_hybrid, meta = eval_ruler.build_model(
        args.model, args.checkpoint, args.attn_impl, device, decode_mode=args.decode_mode)
    use_cache = (not is_hybrid) or (args.decode_mode == "cache")
    if is_hybrid:
        print(f"hybrid: pattern={meta['pattern']} S={meta['S']} {meta['n_race']} RACE; "
              f"unfreeze={meta.get('unfreeze')} base_loaded={meta.get('n_base_loaded')}; "
              f"ckpt step {meta['ckpt_step']}; decode={args.decode_mode}")

    tok = AutoTokenizer.from_pretrained(TEACHER_MODEL)
    eos_ids = {tok.eos_token_id}
    for t in ("<|eot_id|>", "<|end_of_text|>"):
        tid = tok.convert_tokens_to_ids(t)
        if tid is not None and tid >= 0:
            eos_ids.add(tid)

    torch.cuda.reset_peak_memory_stats()
    task_scores = {}
    tot_prefill_ms = tot_decode_ms = 0.0
    tot_prompt_tok = tot_gen_tok = n_seen = 0
    for task in tasks:
        examples = load_examples(task, args.max_examples)
        max_new = MAXLEN[task]
        scores = []
        for ex in examples:
            ids = make_input_ids(tok, task, ex, device, args.max_prompt_tokens)
            gen_ids, n_prompt, prefill_ms, decode_ms, _ns, _stop = eval_ruler.generate(
                model, ids, use_cache, max_new, eos_ids, device)
            pred = tok.decode(gen_ids, skip_special_tokens=True)
            scores.append(score_one(task, pred, list(ex["answers"]), ex.get("all_classes")))
            tot_prefill_ms += prefill_ms; tot_decode_ms += decode_ms
            tot_prompt_tok += n_prompt; tot_gen_tok += len(gen_ids); n_seen += 1
        task_scores[task] = round(100 * sum(scores) / len(scores), 2) if scores else 0.0
        print(f"[{task:20s}] n={len(scores):3d} score={task_scores[task]} (max_new={max_new})")

    overall = round(sum(task_scores.values()) / len(task_scores), 2) if task_scores else 0.0
    peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9
    decode_s = tot_decode_ms / 1e3
    prefill_s = tot_prefill_ms / 1e3
    speed = {
        "prefill_throughput_tok_s": round(tot_prompt_tok / prefill_s, 1) if prefill_s else None,
        "decode_throughput_tok_s": round((tot_gen_tok - n_seen) / decode_s, 2) if decode_s else None,
        "mean_prompt_tokens": round(tot_prompt_tok / max(1, n_seen), 1),
        "total_generated_tokens": tot_gen_tok,
    }
    result = {"model": args.model, "meta": meta, "overall": overall,
              "task_scores": task_scores, "speed": speed,
              "peak_memory_gb": round(peak_mem_gb, 3), "max_examples": args.max_examples}
    os.makedirs(RES, exist_ok=True)
    out_path = os.path.join(RES, f"longbench_{tag}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nOVERALL LongBench = {overall}  peak_mem={peak_mem_gb:.1f}GB  "
          f"prefill={speed['prefill_throughput_tok_s']} tok/s")
    print(f"wrote {out_path}")
    return result


# --------------------------------------------------------------------------- summary
def make_summary():
    rows = []
    if os.path.isdir(RES):
        for fn in sorted(os.listdir(RES)):
            m = re.match(r"longbench_(\w+)\.json$", fn)
            if m:
                with open(os.path.join(RES, fn)) as f:
                    rows.append(json.load(f))
    if not rows:
        print("no longbench_*.json results found")
        return
    all_tasks = sorted({t for r in rows for t in r["task_scores"]})
    head = "| Model | Overall | " + " | ".join(all_tasks) + " | prefill tok/s | peak GB |"
    sep = "| " + " | ".join(["---"] * (len(all_tasks) + 4)) + " |"
    lines = [head, sep]
    for r in sorted(rows, key=lambda r: {"teacher": 0}.get(r["model"], 9)):
        cells = [r["model"], str(r["overall"])]
        cells += [str(r["task_scores"].get(t, "-")) for t in all_tasks]
        cells += [str(r["speed"].get("prefill_throughput_tok_s")), str(r.get("peak_memory_gb"))]
        lines.append("| " + " | ".join(cells) + " |")
    table = "\n".join(lines)
    with open(os.path.join(RES, "longbench_summary.md"), "w") as f:
        f.write("# LongBench summary\n\n" + table + "\n")
    print("\n" + table)
    print(f"\nwrote {os.path.join(RES, 'longbench_summary.md')}")


def main():
    p = argparse.ArgumentParser(description="LongBench eval (teacher / AR / ARRR).")
    p.add_argument("--model", choices=["teacher", "ar", "arrr"])
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    p.add_argument("--max-examples", type=int, default=50, help="per-task cap (0 = all)")
    p.add_argument("--max-prompt-tokens", type=int, default=31500,
                   help="middle-truncate the prompt to this many tokens (official LongBench style)")
    p.add_argument("--attn-impl", default="sdpa")
    p.add_argument("--decode-mode", choices=["cache", "recompute"], default="cache")
    p.add_argument("--out-tag", default=None, help="output filename suffix longbench_<tag>.json")
    p.add_argument("--summary", action="store_true")
    args = p.parse_args()

    if args.summary:
        make_summary(); return
    if not args.model:
        p.error("--model is required (or use --summary)")
    if args.max_examples == 0:
        args.max_examples = None

    run_eval(args)
    try:
        make_summary()
    except Exception as e:
        print(f"[warn] summary refresh skipped: {e}")


if __name__ == "__main__":
    main()
