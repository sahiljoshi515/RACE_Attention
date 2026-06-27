"""Steady-state decode-latency probe: teacher vs ARRR hybrid (cache mode).

For each context length, prefill a real RULER prompt then time a FIXED number of
greedy decode steps (early-stop disabled) and report steady-state per-token decode
latency / throughput. Answers: is the hybrid faster than the teacher at decode?
"""
import time
import argparse
import torch
from transformers import AutoTokenizer
import eval_ruler as E

CKPT = "checkpoints/arrr_L2_K2_arrr_1k_ce_step1000.pt"
DEV = "cuda"


def median(xs):
    xs = sorted(xs); return xs[len(xs) // 2] if xs else float("nan")


@torch.no_grad()
def timed_decode(model, ids, n_steps, warmup):
    """Prefill, then n_steps cached decode steps; return (prefill_ms, median_step_ms)."""
    torch.cuda.synchronize(); t0 = time.perf_counter()
    out = model(input_ids=ids, use_cache=True, logits_to_keep=1)
    torch.cuda.synchronize(); prefill_ms = (time.perf_counter() - t0) * 1e3
    past = out.past_key_values
    nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
    step_ms = []
    for i in range(n_steps):
        torch.cuda.synchronize(); t = time.perf_counter()
        out = model(input_ids=nxt, past_key_values=past, use_cache=True, logits_to_keep=1)
        torch.cuda.synchronize(); dt = (time.perf_counter() - t) * 1e3
        past = out.past_key_values
        nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
        if i >= warmup:
            step_ms.append(dt)
    return prefill_ms, median(step_ms)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--context-lens", type=int, nargs="+", default=[32768, 65536])
    ap.add_argument("--n-steps", type=int, default=40)
    ap.add_argument("--warmup", type=int, default=8)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(E.TEACHER_MODEL)
    print("GPU:", torch.cuda.get_device_name(0))
    for cl in args.context_lens:
        ex = E.load_examples(cl, ["niah_single_1"], max_examples=1)[0]
        ids = E.build_prompt_ids(tok, ex["context"], ex["question"], ex["answer_prefix"], DEV)
        print(f"\n===== context {cl} (prompt {ids.shape[1]} tok), "
              f"{args.n_steps} steps ({args.warmup} warmup) =====")

        tm, _, _ = E.build_model("teacher", None, "sdpa", DEV)
        pf_t, st_t = timed_decode(tm, ids, args.n_steps, args.warmup)
        del tm; torch.cuda.empty_cache()

        hm, _, _ = E.build_model("arrr", CKPT, "sdpa", DEV, decode_mode="cache")
        pf_h, st_h = timed_decode(hm, ids, args.n_steps, args.warmup)
        del hm; torch.cuda.empty_cache()

        print(f"  teacher : prefill {pf_t:8.1f}ms | decode {st_t:6.2f} ms/tok | {1e3/st_t:6.1f} tok/s")
        print(f"  hybrid  : prefill {pf_h:8.1f}ms | decode {st_h:6.2f} ms/tok | {1e3/st_h:6.1f} tok/s")
        print(f"  --> decode speedup hybrid/teacher = {st_t/st_h:.2f}x   "
              f"prefill speedup = {pf_t/pf_h:.2f}x")


if __name__ == "__main__":
    main()
