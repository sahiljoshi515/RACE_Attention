"""Decode-latency scaling: teacher vs ARRR hybrid (cache), synthetic random prompts.

Decode speed is data-independent, so we use random input_ids (no dataset streaming) and
sweep context length to locate the crossover where the hybrid's flat-in-T incremental
decode overtakes the teacher (whose KV read grows with T). Reports decode ms/tok and peak
memory per model per length. Hybrid alone can be pushed to lengths where the teacher OOMs.
"""
import time
import argparse
import torch
import eval_ruler as E

CKPT = "checkpoints/arrr_L2_K2_arrr_1k_ce_step1000.pt"
DEV = "cuda"


def median(xs):
    xs = sorted(xs); return xs[len(xs) // 2] if xs else float("nan")


@torch.no_grad()
def timed(model, T, n_steps, warmup, vocab):
    torch.cuda.reset_peak_memory_stats()
    ids = torch.randint(0, vocab, (1, T), device=DEV)
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
    mem = torch.cuda.max_memory_allocated() / 1e9
    del ids, past, out
    torch.cuda.empty_cache()
    return prefill_ms, median(step_ms), mem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lens", type=int, nargs="+",
                    default=[8192, 32768, 65536, 131072, 262144])
    ap.add_argument("--teacher-max", type=int, default=262144,
                    help="skip teacher above this length (KV cache OOM)")
    ap.add_argument("--n-steps", type=int, default=40)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--decode-kernel", choices=["auto", "on", "off"], default="auto",
                    help="force the RACE fused decode kernel on/off on the hybrid "
                         "(auto = leave each module's default)")
    args = ap.parse_args()

    print("GPU:", torch.cuda.get_device_name(0))
    print(f"decode-kernel: {args.decode_kernel}")
    tm, _, _ = E.build_model("teacher", None, "sdpa", DEV)
    vocab = tm.config.vocab_size
    print(f"\n{'ctx':>9} | {'teacher ms/tok':>15} {'mem GB':>8} | {'hybrid ms/tok':>14} {'mem GB':>8} | {'decode x':>8}")
    print("-" * 78)
    # teacher pass
    teach = {}
    for T in args.lens:
        if T > args.teacher_max:
            continue
        try:
            pf, st, mem = timed(tm, T, args.n_steps, args.warmup, vocab)
            teach[T] = (st, mem, pf)
        except RuntimeError as e:
            teach[T] = None
            print(f"{T:>9} | teacher OOM/err: {str(e)[:40]}")
    del tm; torch.cuda.empty_cache()

    hm, _, _ = E.build_model("arrr", CKPT, "sdpa", DEV, decode_mode="cache")
    if args.decode_kernel != "auto":
        flag = (args.decode_kernel == "on")
        for m in hm.modules():
            if m.__class__.__name__ == "RaceLlamaAttention":
                m._use_decode_kernel = flag
    for T in args.lens:
        try:
            pf_h, st_h, mem_h = timed(hm, T, args.n_steps, args.warmup, vocab)
        except RuntimeError as e:
            print(f"{T:>9} | hybrid OOM/err: {str(e)[:40]}")
            continue
        t = teach.get(T)
        if t:
            st_t, mem_t, _ = t
            ratio = st_t / st_h
            print(f"{T:>9} | {st_t:>15.2f} {mem_t:>8.1f} | {st_h:>14.2f} {mem_h:>8.1f} | {ratio:>7.2f}x")
        else:
            print(f"{T:>9} | {'(teacher n/a)':>15} {'':>8} | {st_h:>14.2f} {mem_h:>8.1f} | {'—':>8}")
    del hm


if __name__ == "__main__":
    main()
