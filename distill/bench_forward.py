"""Forward-pass latency: full Llama-3.2-3B vs a RACE-hybrid. --pattern selects which
attention layers become causal RACE: 'alt' = every odd layer (14/28); 'srrr' =
(Softmax,RACE,RACE,RACE) repeating (21/28). Times the decoder stack `model.model(input_ids)` (the
lm_head is identical for both and infeasible at 1M, so it is excluded). B=1, bf16,
attn backend for the softmax layers (full model + hybrid even layers) is set by
--attn-impl: sdpa (flash/cuDNN) or flash_attention_3 (genuine FA3 on H200).

Sweep T = 4K..1M; for each T time the full model and each hybrid config (S=8, S=24)
by swapping the RACE modules in/out of the SAME loaded model. OOM-safe; adaptive
iters at long T. Writes a CSV; plot with plot_forward.py.
"""
import os
import sys
import csv
import time
import argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transformers import AutoModelForCausalLM
from hybrid import build_race_modules, convert_to_hybrid, odd_layers, srrr_layers

MODEL = "meta-llama/Llama-3.2-3B-Instruct"
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# Which layers become RACE. alt = 14/28 (every odd); srrr = 21/28 ((S,R,R,R) repeating).
PATTERNS = {"alt": odd_layers, "srrr": srrr_layers}


def median(xs):
    xs = sorted(xs); return xs[len(xs) // 2] if xs else float("nan")


def time_forward(model, ids, warmup, iters):
    with torch.no_grad():
        for _ in range(warmup):
            model.model(input_ids=ids, use_cache=False)
            torch.cuda.synchronize()
        ts = []
        for _ in range(iters):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            model.model(input_ids=ids, use_cache=False)
            torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1e3)
    return median(ts)


def safe_time(model, ids, warmup, iters):
    torch.cuda.reset_peak_memory_stats()
    try:
        ms = time_forward(model, ids, warmup, iters)
    except RuntimeError as e:
        torch.cuda.empty_cache()
        if "out of memory" in str(e).lower():
            return float("nan"), float("nan")
        raise
    return ms, torch.cuda.max_memory_allocated() / 1e9


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exps", type=int, nargs="+", default=[12, 13, 14, 15, 16, 17, 18, 19, 20])
    p.add_argument("--configs", nargs="+", default=["2,2", "3,3"])
    p.add_argument("--attn-impl", default="sdpa",
                   help="attention impl for the softmax (full + hybrid even) layers, "
                        "e.g. flash_attention_3, sdpa")
    p.add_argument("--pattern", default="alt", choices=list(PATTERNS),
                   help="alt = RACE every odd layer (14/28); "
                        "srrr = (Softmax,RACE,RACE,RACE) repeating (21/28)")
    p.add_argument("--out", default=os.path.join(RES, "fwd_latency.csv"))
    args = p.parse_args()
    configs = [tuple(int(x) for x in c.split(",")) for c in args.configs]
    os.makedirs(RES, exist_ok=True)
    dev = "cuda"
    print("GPU:", torch.cuda.get_device_name(0))

    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16,
                                                 attn_implementation=args.attn_impl).to(dev).eval()
    model.requires_grad_(False)
    print(f"attn_implementation={args.attn_impl}  (config._attn_implementation="
          f"{getattr(model.config, '_attn_implementation', '?')})")
    vocab = model.config.vocab_size
    pred = PATTERNS[args.pattern]
    repl = [i for i in range(model.config.num_hidden_layers) if pred(i)]
    orig = {i: model.model.layers[i].self_attn for i in repl}

    race_by_cfg = {}
    for (L, K) in configs:
        rm = build_race_modules(model, L=L, Kbits=K, M=1, device=dev, replace_pred=pred)
        for m in rm.values():
            m.to(torch.bfloat16)          # inference: match the bf16 regime
        race_by_cfg[(L, K)] = rm
    print(f"pattern={args.pattern}: replaced {len(repl)}/{model.config.num_hidden_layers} "
          f"layers (RACE at {repl}); softmax kept at "
          f"{[i for i in range(model.config.num_hidden_layers) if not pred(i)]}")
    print(f"hybrid configs: {[ (L,K, L*(1<<K)) for (L,K) in configs ]}")

    f = open(args.out, "w", newline=""); w = csv.writer(f)
    w.writerow(["method", "S", "exponent", "T", "ms", "mem_gb"]); f.flush()

    def restore_full():
        for i in repl:
            model.model.layers[i].self_attn = orig[i]

    for e in args.exps:
        T = 1 << e
        ids = torch.randint(0, vocab, (1, T), device=dev)
        if e >= 20:
            warm, it = 1, 1
        elif e >= 18:
            warm, it = 1, 2
        else:
            warm, it = 2, 3
        # full
        restore_full()
        ms, mem = safe_time(model, ids, warm, it)
        w.writerow(["full", -1, e, T, ms, mem]); f.flush()
        print(f"T=2^{e:<2d} ({T:>8}) full        {ms:10.2f} ms  {mem:6.1f} GB")
        # hybrids
        for (L, K), rm in race_by_cfg.items():
            S = L * (1 << K)
            convert_to_hybrid(model, rm)
            ms, mem = safe_time(model, ids, warm, it)
            w.writerow([f"hybrid_S{S}", S, e, T, ms, mem]); f.flush()
            print(f"            {' '*8}    hybrid_S{S:<3d}{ms:10.2f} ms  {mem:6.1f} GB")
        restore_full()
        del ids; torch.cuda.empty_cache()

    f.close()
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
