"""Phase-0 ablation scorer for RACE-hybrid checkpoints — REAL signals, not 400-step ppl.

For each checkpoint (argv, default glob checkpoints/*abl*step2000.pt) this:
  1. Builds the hybrid student from the ckpt (eval_ruler.build_model, decode cache ON;
     pattern/L/K read from the ckpt config).
  2. PPL: teacher-forced next-token cross-entropy on a small FIXED FineWeb held-out
     batch (data_fineweb.make_eval_and_train seed=0, seq_len 4096, B=1) — same batch
     ppl_probe.py uses. Lower is better.
  3. COLLAPSE: ppl_probe.free_gen_probe greedy free-gen on a few fixed prompts; reports
     the fraction that collapse to repetition.
  4. RETRIEVAL (the gate): a self-contained synthetic NIAH-4K probe. We generate N
     single-needle samples with data_long.synthetic_ruler_batch (task="niah_single",
     query_agnostic) using an EXPLICIT filler_pool string (offline-safe; no RULER HF
     dataset, no PG19 hang). For each sample we decode the prompt and regex the embedded
     needle value (the 7-digit magic number) as the KNOWN ground truth, greedily decode
     ~16 tokens, and check whether that digit string appears in the answer -> niah_acc.
     The TEACHER is scored on the SAME samples as the retrieval ceiling.
  5. Prints a per-checkpoint table + a final RANKING (prefer: not collapsed, low ppl,
     high niah_acc) and writes results/ablation_score.json.

Robust: each checkpoint is wrapped in try/except (one bad ckpt won't kill the run) and
GPU memory is freed between checkpoints (del model; torch.cuda.empty_cache()).
"""
from __future__ import annotations

import os
import sys
import re
import gc
import glob
import json
import argparse
import traceback

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from transformers import AutoModelForCausalLM           # noqa: E402
from data_fineweb import get_tokenizer, make_eval_and_train  # noqa: E402
from data_long import synthetic_ruler_batch             # noqa: E402
import eval_ruler                                        # noqa: E402
import ppl_probe                                         # noqa: E402

MODEL = "meta-llama/Llama-3.2-3B-Instruct"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
RES = os.path.join(HERE, "results")
DEFAULT_GLOB = os.path.join(HERE, "checkpoints", "*abl*step2000.pt")

# Fixed free-gen collapse prompts (Phase-1 de-collapse gate).
COLLAPSE_PROMPTS = [
    "Tell me about the history of the printing press.",
    "Explain in a few sentences how photosynthesis works.",
    "What are the main causes of the French Revolution?",
]

# The niah_single needle is: "The special magic number for {key} is {val}." with
# val = str(randint(1_000_000, 9_999_999)) (7 digits) -> regex the (key, value). The
# needle (with value) precedes the value-less answer_prefix tail, so .search finds it
# first; the answer_prefix has no digit so \d{6,8} cannot match it anyway.
_NEEDLE_RE = re.compile(r"special magic number for\s+([a-z]+)\s+is\s+(\d{6,8})\b", re.I)

N_NIAH = 20            # number of single-needle samples
NIAH_NEW_TOKENS = 16   # tokens to decode for the answer
SEQ_LEN = 4096


# --------------------------------------------------------------------------- helpers
def _free():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def ce_ppl(logits, ids):
    """Teacher-forced next-token CE + ppl over a [B,T] sample (matches ppl_probe.ce_ppl)."""
    sl = logits[:, :-1].float().reshape(-1, logits.size(-1))
    lbl = ids[:, 1:].reshape(-1)
    ce = F.cross_entropy(sl, lbl).item()
    return ce, float(torch.exp(torch.tensor(ce)))


def build_filler_pool(tok, seed=0, min_chars=400_000):
    """Build a filler TEXT string ONCE, offline-safe. Try a tiny FineWeb pull via the
    packed loader; on any failure (504 / offline) fall back to a deterministic
    pseudo-corpus. Passed explicitly to synthetic_ruler_batch so PG19 never streams."""
    try:
        # A couple of small packed batches give us ~tens of K tokens of real text cheaply.
        eval_b, _ = make_eval_and_train(tok, seq_length=SEQ_LEN, batch_size=2,
                                        num_eval_batches=2, max_train_batches=1, seed=seed)
        parts = [tok.decode(b.reshape(-1).tolist(), skip_special_tokens=True) for b in eval_b]
        text = "\n\n".join(parts)
        # Repeat to reach a comfortable size for slicing distinct windows per sample.
        if 0 < len(text) < min_chars:
            text = (text * (min_chars // len(text) + 1))
        if len(text) >= 2000:
            return text
    except Exception as e:
        print(f"[ablation_score] filler FineWeb pull failed ({type(e).__name__}: {e}); "
              f"using deterministic pseudo-corpus")
    # Deterministic offline fallback (no network, no hang).
    import random
    rng = random.Random(seed or 1)
    words = ["the", "of", "and", "to", "in", "a", "that", "was", "his", "he", "with",
             "it", "as", "for", "her", "had", "is", "at", "but", "on", "not", "they",
             "river", "mountain", "evening", "letter", "garden", "window", "memory",
             "harbor", "lantern", "meadow", "ribbon", "cottage", "thunder", "willow"]
    buf = []
    while sum(len(w) + 1 for w in buf) < min_chars:
        buf.append(rng.choice(words))
        if rng.random() < 0.05:
            buf.append(".\n")
    return " ".join(buf)


def make_niah_samples(tok, filler_pool, n=N_NIAH, seed=0):
    """Build n single-needle samples as [1,SEQ_LEN] id tensors with a KNOWN ground-truth
    value. We draw from synthetic_ruler_batch(task='niah_single', query_agnostic) one row
    at a time, decode the prompt, and regex the embedded magic number as ground truth.
    Samples whose needle cannot be recovered (rare length-trim clobber) are skipped."""
    samples = []  # list of (ids[1,T] cpu long, gt_value str, key str)
    gen = synthetic_ruler_batch(tok, seq_length=SEQ_LEN, batch_size=1, seed=seed,
                                task="niah_single", fmt="query_agnostic",
                                filler_pool=filler_pool, max_batches=n * 3)
    for batch in gen:
        row = batch[0]                          # [T] long
        text = tok.decode(row.tolist(), skip_special_tokens=True)
        m = _NEEDLE_RE.search(text)
        if not m:
            continue
        key, val = m.group(1), m.group(2)
        # The question must ask for THIS key (query-agnostic prompt embeds it at the end).
        if f"magic number for {key}".lower() not in text.lower():
            continue
        samples.append((row.unsqueeze(0).long(), val, key))
        if len(samples) >= n:
            break
    return samples


@torch.no_grad()
def greedy_decode(model, ids, max_new, eos_ids, use_cache):
    """Greedy decode mirroring eval_ruler.generate's logic: reset RACE state, prefill with
    logits_to_keep=1, then step. use_cache drives the incremental path (teacher KV cache or
    hybrid softmax-KV + RACE running state); use_cache=False recomputes full seq each step."""
    for mod in model.modules():
        if hasattr(mod, "reset_decode_state"):
            mod.reset_decode_state()
    out = model(input_ids=ids, use_cache=use_cache, logits_to_keep=1)
    past = out.past_key_values if use_cache else None
    nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
    gen = [int(nxt)]
    cur = torch.cat([ids, nxt], dim=1)
    for _ in range(max_new - 1):
        if int(nxt) in eos_ids:
            break
        if eval_ruler._is_looping(gen):
            break
        if use_cache:
            out = model(input_ids=nxt, past_key_values=past, use_cache=True, logits_to_keep=1)
            past = out.past_key_values
        else:
            out = model(input_ids=cur, use_cache=False, logits_to_keep=1)
        nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
        gen.append(int(nxt))
        cur = torch.cat([cur, nxt], dim=1)
    return gen


def niah_accuracy(model, tok, samples, use_cache):
    """Fraction of samples whose ground-truth magic number appears in the decoded answer."""
    if not samples:
        return 0.0, 0
    eos_ids = {tok.eos_token_id}
    for t in getattr(tok, "additional_special_tokens_ids", []) or []:
        eos_ids.add(t)
    hits = 0
    for ids, gt, _key in samples:
        try:
            gen = greedy_decode(model, ids.to(DEV), NIAH_NEW_TOKENS, eos_ids, use_cache)
            ans = tok.decode(gen, skip_special_tokens=True)
            if gt in ans:
                hits += 1
        except Exception:
            pass
    return hits / len(samples), len(samples)


def collapse_fraction(model, tok, prompts):
    """Run the free-gen collapse probe on each prompt; return fraction COLLAPSED.
    ppl_probe.free_gen_probe enables/resets RACE decode state, greedy-gens, returns
    True when NOT collapsed."""
    collapsed = 0
    for p in prompts:
        try:
            ok = ppl_probe.free_gen_probe(model, tok, p, max_new=64)
            if not ok:
                collapsed += 1
        except Exception as e:
            print(f"  collapse probe error on prompt: {type(e).__name__}: {e}")
            collapsed += 1          # treat a crash during free-gen as a failure
    return collapsed / max(1, len(prompts))


# --------------------------------------------------------------------------- per-ckpt
def score_checkpoint(path, tok, eval_ids, niah_samples, args):
    """Build the hybrid from `path`, return a result dict. Raises on build failure."""
    model, is_hybrid, meta = eval_ruler.build_model(
        kind="arrr", checkpoint=path, attn_impl="sdpa", device=DEV, decode_mode="cache")
    name = os.path.basename(path)
    pattern, S = meta.get("pattern", "?"), meta.get("S", "?")
    try:
        # 2. PPL (teacher-forced, no autocast — matches eval_ruler harness build / ppl_probe).
        with torch.no_grad():
            ce, ppl = ce_ppl(model(input_ids=eval_ids, use_cache=False).logits, eval_ids)

        # 3. COLLAPSE free-gen probe.
        coll = collapse_fraction(model, tok, COLLAPSE_PROMPTS)

        # 4. RETRIEVAL (hybrid uses cache decode = the eval path).
        acc, n_used = niah_accuracy(model, tok, niah_samples, use_cache=True)

        return {
            "name": name, "checkpoint": path, "pattern": pattern, "S": S,
            "L": meta.get("L"), "K": meta.get("K"), "n_race": meta.get("n_race"),
            "ce": round(ce, 4), "ppl": round(ppl, 2),
            "collapsed_frac": round(coll, 3),
            "niah_acc": round(acc, 3), "niah_n": n_used,
            "ok": True, "error": None,
        }
    finally:
        del model
        _free()


def teacher_ceiling(tok, eval_ids, niah_samples):
    """Score the frozen teacher: ppl + niah retrieval ceiling on the SAME samples."""
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()
    model.requires_grad_(False)
    try:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            ce, ppl = ce_ppl(model(input_ids=eval_ids, use_cache=False).logits, eval_ids)
        acc, n_used = niah_accuracy(model, tok, niah_samples, use_cache=True)
        return {"name": "TEACHER", "checkpoint": MODEL, "pattern": "softmax", "S": "-",
                "ce": round(ce, 4), "ppl": round(ppl, 2),
                "collapsed_frac": None, "niah_acc": round(acc, 3), "niah_n": n_used,
                "ok": True, "error": None}
    finally:
        del model
        _free()


# --------------------------------------------------------------------------- main
def main():
    global N_NIAH
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoints", nargs="*", help="ckpt paths (default: glob *abl*step2000.pt)")
    ap.add_argument("--n-niah", type=int, default=N_NIAH)
    ap.add_argument("--no-teacher", action="store_true", help="skip the teacher ceiling")
    ap.add_argument("--out", default=os.path.join(RES, "ablation_score.json"))
    args = ap.parse_args()
    N_NIAH = args.n_niah

    paths = args.checkpoints or sorted(glob.glob(DEFAULT_GLOB))
    if not paths:
        # Fallback: any ablation step checkpoint (step2000 may not exist yet).
        paths = sorted(glob.glob(os.path.join(HERE, "checkpoints", "*abl*step*.pt")))
        if paths:
            print(f"[ablation_score] no *abl*step2000.pt; falling back to {len(paths)} "
                  f"*abl*step*.pt checkpoints")
    if not paths:
        print(f"[ablation_score] no checkpoints found (glob={DEFAULT_GLOB})")
        return
    print(f"[ablation_score] scoring {len(paths)} checkpoint(s) on device={DEV}")

    tok = get_tokenizer()

    # Fixed FineWeb held-out batch (seed=0, seq_len 4096, B=1) — same as ppl_probe.
    eval_batches, _ = make_eval_and_train(tok, seq_length=SEQ_LEN, batch_size=2,
                                          num_eval_batches=1, max_train_batches=1, seed=0)
    eval_ids = eval_batches[0][:1].to(DEV)
    print(f"[ablation_score] eval batch {tuple(eval_ids.shape)} (FineWeb held-out, seed=0)")

    # Offline-safe filler + the fixed NIAH sample set (shared across teacher + students).
    filler = build_filler_pool(tok, seed=0)
    niah_samples = make_niah_samples(tok, filler, n=N_NIAH, seed=0)
    print(f"[ablation_score] built {len(niah_samples)} niah_single samples "
          f"(requested {N_NIAH})")

    results = []

    if not args.no_teacher:
        try:
            tr = teacher_ceiling(tok, eval_ids, niah_samples)
            print(f"[ablation_score] TEACHER ppl={tr['ppl']} niah_acc={tr['niah_acc']}")
            results.append(tr)
        except Exception as e:
            print(f"[ablation_score] teacher ceiling failed: {type(e).__name__}: {e}")
            traceback.print_exc()
            _free()

    for p in paths:
        print(f"\n[ablation_score] === {os.path.basename(p)} ===")
        try:
            r = score_checkpoint(p, tok, eval_ids, niah_samples, args)
            print(f"  pattern={r['pattern']}/S={r['S']} ppl={r['ppl']} "
                  f"collapsed={r['collapsed_frac']} niah_acc={r['niah_acc']}")
            results.append(r)
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
            traceback.print_exc()
            results.append({"name": os.path.basename(p), "checkpoint": p, "ok": False,
                            "error": f"{type(e).__name__}: {e}", "ppl": None,
                            "collapsed_frac": None, "niah_acc": None})
            _free()

    # ----------------------------------------------------------------- table + ranking
    def fmt(v, w, prec=None):
        if v is None:
            return "-".ljust(w) if w else "-"
        if prec is not None:
            return f"{v:.{prec}f}".ljust(w)
        return str(v).ljust(w)

    print("\n" + "=" * 92)
    print(f"{'name':40s} {'pattern/S':12s} {'ppl':>8s} {'collapsed':>10s} {'niah_acc':>9s}")
    print("-" * 92)
    for r in results:
        ps = f"{r.get('pattern','?')}/{r.get('S','?')}"
        print(f"{r['name'][:40]:40s} {ps:12s} "
              f"{fmt(r.get('ppl'), 0, 2):>8s} "
              f"{fmt(r.get('collapsed_frac'), 0, 3):>10s} "
              f"{fmt(r.get('niah_acc'), 0, 3):>9s}")
    print("=" * 92)

    # Ranking among successful STUDENTS only (teacher is the ceiling, not ranked).
    students = [r for r in results if r.get("ok") and r["name"] != "TEACHER"]
    # Prefer: not collapsed (low collapsed_frac), high niah_acc, then low ppl.
    def rank_key(r):
        return (round(r.get("collapsed_frac") or 1.0, 3),     # asc: fewer collapses better
                -(r.get("niah_acc") or 0.0),                  # desc: higher acc better
                r.get("ppl") if r.get("ppl") is not None else float("inf"))  # asc: lower ppl
    students.sort(key=rank_key)

    print("\nRANKING (best first: not-collapsed > high niah_acc > low ppl)")
    for i, r in enumerate(students, 1):
        print(f"  {i}. {r['name'][:50]:50s} "
              f"S={r.get('S')} collapsed={r.get('collapsed_frac')} "
              f"niah_acc={r.get('niah_acc')} ppl={r.get('ppl')}")
    if not students:
        print("  (no successful student checkpoints)")

    os.makedirs(RES, exist_ok=True)
    payload = {
        "model": MODEL, "device": DEV, "seq_len": SEQ_LEN,
        "n_niah_requested": N_NIAH, "n_niah_built": len(niah_samples),
        "collapse_prompts": COLLAPSE_PROMPTS,
        "results": results,
        "ranking": [r["name"] for r in students],
    }
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[ablation_score] wrote {args.out}")


if __name__ == "__main__":
    main()
