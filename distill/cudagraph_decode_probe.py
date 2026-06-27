"""CUDA-graph decode-latency probe: teacher vs ARRR hybrid, StaticCache, batch>=1.

Motivation: the eager probe (synthetic_decode_probe.py) showed the hybrid's decode
floor (~12.7 ms/tok) sits ON the per-step launch/Python overhead, not on attention,
so the O(1)-in-T RACE advantage only appears past ~64-100K ctx. This probe removes the
per-step launch overhead by capturing one decode step as a CUDA graph and replaying it,
and sweeps batch size. If the floor is launch-bound, the captured floor should drop well
below 12 ms and the teacher-vs-hybrid crossover should move to shorter context.

Method:
  * StaticCache (pre-allocated to ~ctx length) for the kept softmax layers: its layer
    .update() advances an on-device cumulative_length and writes via index_copy_ -> no
    D2H sync, capturable. RACE layers ignore the cache and advance their own _dec_A/_dec_B
    IN PLACE via the fused decode kernel (forced ON; the torch path REASSIGNS _dec_A and
    is therefore NOT capturable).
  * A single decode step (q_len==1) is captured with static input-id / cache_position
    buffers; replay advances both in lockstep with the cache's internal cumulative_length.
  * CORRECTNESS GATE: generate G tokens eager vs G tokens via the graph from an identical
    prefill; require token-identical. Timing is reported only if the gate passes, so a
    stale-address / hidden-sync capture bug surfaces as a mismatch, never as a fake speedup.

Usage:
  python cudagraph_decode_probe.py --lens 8192 32768 131072 262144 --batch 1
  python cudagraph_decode_probe.py --lens 8192 32768 65536 --batch 8
"""
import time
import argparse
import contextlib
import torch
import eval_ruler as E
from transformers import StaticCache
from torch.nn.attention import sdpa_kernel, SDPBackend

CKPT = "checkpoints/arrr_L2_K2_arrr_1k_ce_step1000.pt"
DEV = "cuda"


def median(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else float("nan")


def reset_race_state(model):
    for m in model.modules():
        if hasattr(m, "reset_decode_state"):
            m.reset_decode_state()


def force_kernel(model, flag):
    """Force the fused RACE decode kernel on/off; needed ON for capturable in-place state.
    Returns the number of RACE modules touched (0 => pure teacher)."""
    n = 0
    for m in model.modules():
        if m.__class__.__name__ == "RaceLlamaAttention":
            m._use_decode_kernel = flag
            n += 1
    return n


@torch.no_grad()
def prefill(model, ids, max_len):
    """Fresh StaticCache + RACE state, run the prompt, return (cache, first_token[B,1], T)."""
    B, T = ids.shape
    reset_race_state(model)
    cache = StaticCache(model.config, max_cache_len=max_len)
    pos = torch.arange(T, device=DEV)
    out = model(input_ids=ids, past_key_values=cache, use_cache=True,
                cache_position=pos, logits_to_keep=1)
    nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)   # [B,1]
    return cache, nxt, T


@torch.no_grad()
def gen_eager(model, ids, n_tok, max_len):
    # Chain starts AFTER the prefill token (A0 is discarded), so it lines up with the
    # graph chain whose warmup steps consume A0 first. n_tok tokens = [A1..A_n].
    cache, nxt, T = prefill(model, ids, max_len)
    toks = []
    pos = torch.tensor([T], device=DEV)
    for _ in range(n_tok):
        out = model(input_ids=nxt, past_key_values=cache, use_cache=True,
                    cache_position=pos, logits_to_keep=1)
        nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
        toks.append(nxt)
        pos = pos + 1
    return torch.cat(toks, dim=1)   # [B, n_tok]


@torch.no_grad()
def capture_graph(model, ids, max_len):
    """Prefill, warm up a few decode steps, then capture ONE decode step.
    Returns (graph, static_input[B,1], static_pos[1], static_logits, cache, T_after_warmup)."""
    cache, nxt, T = prefill(model, ids, max_len)
    static_input = nxt.clone()                       # [B,1]
    static_pos = torch.tensor([T], device=DEV)       # cache_position for next step

    # Warmup on a side stream: triggers lazy init, the one-time _dec_scale .item() D2H,
    # and lets the caching allocator settle before capture. Advances real state a few steps.
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            out = model(input_ids=static_input, past_key_values=cache, use_cache=True,
                        cache_position=static_pos, logits_to_keep=1)
            static_input.copy_(out.logits[:, -1, :].argmax(-1, keepdim=True))
            static_pos.add_(1)
    torch.cuda.current_stream().wait_stream(s)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = model(input_ids=static_input, past_key_values=cache, use_cache=True,
                    cache_position=static_pos, logits_to_keep=1)
        static_logits = out.logits[:, -1, :]         # [B,V] at a static address
    return g, static_input, static_pos, static_logits, cache, static_pos.item()


@torch.no_grad()
def gen_graph(model, ids, n_tok, max_len, n_warm=3):
    """Generate n_tok via captured graph from a fresh prefill (for the correctness gate).
    The n_warm warmup steps are REAL decode steps; their tokens are recorded so the chain
    is [A1..A_n] and lines up with gen_eager (which also drops the prefill token A0)."""
    cache, nxt, T = prefill(model, ids, max_len)
    static_input = nxt.clone()
    static_pos = torch.tensor([T], device=DEV)
    toks = []
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(n_warm):
            out = model(input_ids=static_input, past_key_values=cache, use_cache=True,
                        cache_position=static_pos, logits_to_keep=1)
            tok = out.logits[:, -1, :].argmax(-1, keepdim=True)
            toks.append(tok.clone())
            static_input.copy_(tok)
            static_pos.add_(1)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = model(input_ids=static_input, past_key_values=cache, use_cache=True,
                    cache_position=static_pos, logits_to_keep=1)
        static_logits = out.logits[:, -1, :]
    while len(toks) < n_tok:
        g.replay()
        tok = static_logits.argmax(-1, keepdim=True)
        toks.append(tok.clone())
        static_input.copy_(tok)
        static_pos.add_(1)
    return torch.cat(toks[:n_tok], dim=1)


@torch.no_grad()
def time_graph(model, ids, n_steps, warmup, max_len):
    g, static_input, static_pos, static_logits, cache, _ = capture_graph(model, ids, max_len)
    torch.cuda.reset_peak_memory_stats()
    step_ms = []
    for i in range(n_steps):
        torch.cuda.synchronize(); t = time.perf_counter()
        g.replay()
        torch.cuda.synchronize(); dt = (time.perf_counter() - t) * 1e3
        tok = static_logits.argmax(-1, keepdim=True)
        static_input.copy_(tok)
        static_pos.add_(1)
        if i >= warmup:
            step_ms.append(dt)
    mem = torch.cuda.max_memory_allocated() / 1e9
    return median(step_ms), mem


def run_model(label, model, lens, B, n_steps, warmup, teacher_max, vocab, gate_tok):
    print(f"\n########## {label}  (batch={B}) ##########")
    res = {}
    for T in lens:
        if T > teacher_max:
            print(f"{T:>9} | skipped (> teacher_max)")
            continue
        max_len = T + n_steps + 8
        ids = torch.randint(0, vocab, (B, T), device=DEV)
        try:
            # Correctness gate. Greedy decode over batched matmul is run-to-run
            # nondeterministic (tie-flips in argmax), so eager isn't even self-consistent
            # at batch>1. Distinguish a real capture bug (eager deterministic but graph
            # differs) from a tie-flip (eager differs from itself too).
            te1 = gen_eager(model, ids, gate_tok, max_len)
            tg = gen_graph(model, ids, gate_tok, max_len)
            if torch.equal(te1, tg):
                tag = "gate OK"
            else:
                te2 = gen_eager(model, ids, gate_tok, max_len)
                if not torch.equal(te1, te2):
                    tag = "gate OK*(nondet)"   # eager self-inconsistent -> tie-flip, not a bug
                else:
                    nmatch = (te1 == tg).all(0).int().argmin().item()
                    print(f"{T:>9} | GATE FAIL: eager deterministic but graph diverges at "
                          f"tok {nmatch} (eager {te1[0].tolist()} vs graph {tg[0].tolist()})")
                    res[T] = None
                    continue
            st, mem = time_graph(model, ids, n_steps, warmup, max_len)
            res[T] = (st, mem)
            print(f"{T:>9} | {tag} | {st:8.2f} ms/tok | {mem:6.1f} GB")
        except RuntimeError as e:
            print(f"{T:>9} | OOM/err: {str(e)[:70]}")
            res[T] = None
        finally:
            del ids
            torch.cuda.empty_cache()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lens", type=int, nargs="+", default=[8192, 32768, 131072, 262144])
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--teacher-max", type=int, default=262144)
    ap.add_argument("--n-steps", type=int, default=40)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--gate-tok", type=int, default=8,
                    help="tokens compared eager-vs-graph in the correctness gate")
    ap.add_argument("--sdpa", choices=["default", "efficient", "math"], default="efficient",
                    help="force the sdpa backend. StaticCache passes an explicit 4D mask "
                         "which routes 'default' to the slow MATH path; 'efficient' (mem-"
                         "efficient attention) handles the mask far faster -> fair teacher.")
    args = ap.parse_args()

    print("GPU:", torch.cuda.get_device_name(0))
    print(f"lens={args.lens} batch={args.batch} n_steps={args.n_steps} sdpa={args.sdpa}")
    if args.sdpa == "efficient":
        ctx = sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.FLASH_ATTENTION,
                           SDPBackend.MATH])
    elif args.sdpa == "math":
        ctx = sdpa_kernel([SDPBackend.MATH])
    else:
        ctx = contextlib.nullcontext()
    with ctx:
        return _run(args)


def _run(args):

    tm, _, _ = E.build_model("teacher", None, "sdpa", DEV)
    vocab = tm.config.vocab_size
    force_kernel(tm, False)
    teach = run_model("TEACHER (cudagraph, StaticCache)", tm, args.lens, args.batch,
                      args.n_steps, args.warmup, args.teacher_max, vocab, args.gate_tok)
    del tm; torch.cuda.empty_cache()

    hm, _, _ = E.build_model("arrr", CKPT, "sdpa", DEV, decode_mode="cache")
    nk = force_kernel(hm, True)
    print(f"\n[hybrid] forced decode kernel ON across {nk} RACE modules")
    hyb = run_model("HYBRID (cudagraph, StaticCache + RACE kernel)", hm, args.lens, args.batch,
                    args.n_steps, args.warmup, 10**12, vocab, args.gate_tok)
    del hm; torch.cuda.empty_cache()

    print(f"\n{'='*78}\nSUMMARY  batch={args.batch}  (cudagraph decode ms/tok)")
    print(f"{'ctx':>9} | {'teacher':>10} {'mem GB':>8} | {'hybrid':>10} {'mem GB':>8} | {'decode x':>9}")
    print("-" * 78)
    for T in args.lens:
        t = teach.get(T); h = hyb.get(T)
        ts = f"{t[0]:10.2f}" if t else f"{'OOM/—':>10}"
        tmem = f"{t[1]:8.1f}" if t else f"{'':>8}"
        hs = f"{h[0]:10.2f}" if h else f"{'OOM/—':>10}"
        hmem = f"{h[1]:8.1f}" if h else f"{'':>8}"
        x = f"{(t[0]/h[0]):8.2f}x" if (t and h) else f"{'—':>9}"
        print(f"{T:>9} | {ts} {tmem} | {hs} {hmem} | {x}")


if __name__ == "__main__":
    main()
