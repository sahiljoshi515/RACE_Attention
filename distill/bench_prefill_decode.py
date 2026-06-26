"""Prefill + decode latency: FULL Llama-3.2-3B (all softmax = genuine FA3) vs the RACE-HYBRID
(FA3 for kept-softmax layers + the fused RACE decode kernel for RACE layers).

Context lengths T in {4k,8k,16k,32k,64k}; decode DECODE_STEPS new tokens (fixed, no early stop --
this is a latency bench, weights/tokens are irrelevant). B=1, bf16. RUN UNDER distill/env_fa3.sh
(swa_env, torch 2.8, genuine FA3; RACE CUDA kernels JIT-build against torch 2.8 in .torch_ext_fa3).

Each method:
  * full   = teacher (all 28 softmax -> FA3 KV-cache decode)
  * hybrid = AR/S24 (14/28 RACE; FA3 KV cache on softmax layers + fused RACE decode kernel on RACE
             layers via decode_mode="cache"); optionally srrr (21/28 RACE) built fresh.
Reuses eval_ruler.build_model. Writes results/prefill_decode_latency.csv; plot with
plot_prefill_decode.py.
"""
import os, sys, csv, time, argparse, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_ruler
from hybrid import build_race_modules, convert_to_hybrid, srrr_layers, make_replace_pred

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
MODEL = eval_ruler.TEACHER_MODEL


def median(xs):
    xs = sorted(xs); return xs[len(xs) // 2] if xs else float("nan")


def _reset_race(model):
    for m in model.modules():
        if hasattr(m, "reset_decode_state"):
            m.reset_decode_state()


def _kernel_used(model):
    """True if every RACE module is using the fused decode kernel (no silent torch fallback)."""
    flags = [bool(getattr(m, "_use_decode_kernel", False))
             for m in model.modules() if hasattr(m, "_use_decode_kernel")]
    return (all(flags), len(flags))


@torch.no_grad()
def time_one(model, T, vocab, steps, warmup, iters, device, batch=1):
    pre_list, dec_list = [], []
    peak = 0.0
    for it in range(warmup + iters):
        try:
            ids = torch.randint(0, vocab, (batch, T), device=device)
            _reset_race(model)
            torch.cuda.reset_peak_memory_stats()
            # ---- prefill ----
            torch.cuda.synchronize(); t0 = time.perf_counter()
            out = model(input_ids=ids, use_cache=True, logits_to_keep=1)
            torch.cuda.synchronize(); pre = (time.perf_counter() - t0) * 1e3
            past = out.past_key_values
            nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
            # ---- decode (fixed #steps, no early stop) ----
            torch.cuda.synchronize(); t1 = time.perf_counter()
            for _ in range(steps):
                out = model(input_ids=nxt, past_key_values=past, use_cache=True, logits_to_keep=1)
                past = out.past_key_values
                nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
            torch.cuda.synchronize(); dec = (time.perf_counter() - t1) * 1e3
            peak = max(peak, torch.cuda.max_memory_allocated() / 1e9)
            if it >= warmup:
                pre_list.append(pre); dec_list.append(dec)
            del ids, out, past, nxt
            torch.cuda.empty_cache()
        except RuntimeError as e:        # OOM-safe (B=8 full model can exceed 141GB at long T)
            torch.cuda.empty_cache()
            if "out of memory" in str(e).lower():
                return float("nan"), float("nan"), float("nan")
            raise
    return median(pre_list), median(dec_list), peak


def build_full(attn_impl, device):
    m, _, meta = eval_ruler.build_model("teacher", None, attn_impl, device, decode_mode="cache")
    return m, meta


def build_hybrid_ar(ckpt, attn_impl, device):
    m, _, meta = eval_ruler.build_model("ar", ckpt, attn_impl, device, decode_mode="cache")
    return m, meta


def build_hybrid_fresh(pattern, L, K, attn_impl, device):
    """Fresh hybrid (random RACE weights -- fine for latency) for patterns we have no ckpt for."""
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16,
                                                 attn_implementation=attn_impl).to(device).eval()
    model.requires_grad_(False)
    pred = make_replace_pred(pattern, model.config.num_hidden_layers)
    race = build_race_modules(model, L=L, Kbits=K, M=1, device=device, replace_pred=pred)
    for r in race.values():
        r.to(torch.bfloat16)
        r.enable_decode_cache(True)
    convert_to_hybrid(model, race)
    model.eval()
    replaced = sorted(int(k) for k in race.keys())
    return model, {"pattern": pattern, "S": L * (1 << K), "n_race": len(replaced),
                   "n_softmax": model.config.num_hidden_layers - len(replaced)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--Ts", type=int, nargs="+", default=[4096, 8192, 16384, 32768, 65536])
    p.add_argument("--decode-steps", type=int, default=64)
    p.add_argument("--batch", type=int, default=1, help="batch size B (prefill+decode); B=8 full model may OOM at long T -> recorded as nan")
    p.add_argument("--attn-impl", default="flash_attention_3")
    p.add_argument("--ckpt", default="checkpoints/best/race_hybrid_AR_S24_p1c_ext_step450.pt")
    p.add_argument("--srrr", action="store_true", help="also bench a fresh srrr (21/28 RACE) hybrid")
    p.add_argument("--quick", action="store_true", help="sanity: only the smallest T, 1 iter")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    dev = "cuda"
    if args.out is None:
        args.out = os.path.join(RES, "prefill_decode_latency.csv" if args.batch == 1
                                else f"prefill_decode_latency_b{args.batch}.csv")
    Ts = args.Ts[:1] if args.quick else args.Ts
    os.makedirs(RES, exist_ok=True)
    print("GPU:", torch.cuda.get_device_name(0), "| decode_steps:", args.decode_steps, "| Ts:", Ts)

    methods = [("full", lambda: build_full(args.attn_impl, dev)),
               ("hybrid_S24_AR", lambda: build_hybrid_ar(args.ckpt, args.attn_impl, dev))]
    if args.srrr:
        methods.append(("hybrid_S24_srrr", lambda: build_hybrid_fresh("srrr", 3, 3, args.attn_impl, dev)))

    f = open(args.out, "w", newline=""); w = csv.writer(f)
    w.writerow(["method", "S", "B", "T", "prefill_ms", "decode_ms_total", "decode_ms_per_tok", "mem_gb"]); f.flush()

    for name, builder in methods:
        model, meta = builder()
        ai = getattr(model.config, "_attn_implementation", "?")
        print(f"\n=== {name} | attn_impl={ai} | S={meta.get('S','-')} "
              f"n_race={meta.get('n_race','-')}/{meta.get('n_race',0)+meta.get('n_softmax',28)} ===")
        assert ai == args.attn_impl, f"FA3 not engaged for {name}: config._attn_implementation={ai}"
        vocab = model.config.vocab_size
        for T in Ts:
            if args.quick:
                warm, it = 0, 1
            elif T >= 65536:
                warm, it = 1, 1
            elif T >= 32768:
                warm, it = 1, 2
            else:
                warm, it = 1, 3
            pre, dec, mem = time_one(model, T, vocab, 1 if args.quick else args.decode_steps,
                                     warm, it, dev, batch=args.batch)
            steps = 1 if args.quick else args.decode_steps
            per = dec / max(1, steps)
            kused, nk = _kernel_used(model)
            w.writerow([name, meta.get("S", -1), args.batch, T, round(pre, 2), round(dec, 2),
                        round(per, 3), round(mem, 2)]); f.flush()
            print(f"  B={args.batch} T={T:>6}  prefill={pre:9.2f} ms  decode={dec:8.2f} ms "
                  f"({per:6.2f} ms/tok)  mem={mem:5.1f} GB  | race_kernel_used={kused}({nk} modules)")
        del model; torch.cuda.empty_cache()
    f.close()
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
