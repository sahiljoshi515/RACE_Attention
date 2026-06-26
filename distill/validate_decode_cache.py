"""Validate the hybrid KV-cache decode path against the verified full-recompute path.

Builds the ARRR hybrid twice (decode_mode='recompute' and 'cache'), generates greedily
from the SAME RULER prompt, and checks the produced token ids are identical. Also reports
decode tokens/sec for each so the speedup is visible. If tokens match, the softmax-KV-cache
+ RACE-incremental-state path is correct.
"""
import time
import torch

from transformers import AutoTokenizer
import eval_ruler as E

CKPT = "checkpoints/arrr_L2_K2_arrr_1k_ce_step1000.pt"
CTX = 8192          # modest length so recompute is affordable for the comparison
N_TOK = 24
DEV = "cuda"


def run(decode_mode, ids, eos_ids, use_kernel=False):
    model, is_hybrid, meta = E.build_model("arrr", CKPT, "sdpa", DEV, decode_mode=decode_mode)
    # For the cache path, explicitly pin the RACE decode kernel on/off so both the torch
    # reference (use_kernel=False) and the fused kernel (use_kernel=True) are validated
    # against recompute (the default flag may be True when the ext built cleanly).
    if decode_mode == "cache":
        for m in model.modules():
            if m.__class__.__name__ == "RaceLlamaAttention":
                m._use_decode_kernel = use_kernel
    use_cache = (decode_mode == "cache")
    torch.cuda.synchronize(); t0 = time.perf_counter()
    gen, n_prompt, prefill_ms, decode_ms, n_steps, stop = E.generate(
        model, ids, use_cache, N_TOK, eos_ids, DEV)
    torch.cuda.synchronize()
    tok_s = (len(gen) - 1) / (decode_ms / 1e3) if decode_ms > 0 else float("nan")
    del model; torch.cuda.empty_cache()
    return gen, prefill_ms, decode_ms, tok_s, stop


def main():
    tok = AutoTokenizer.from_pretrained(E.TEACHER_MODEL)
    eos_ids = {tok.eos_token_id}
    for t in ("<|eot_id|>", "<|end_of_text|>"):
        tid = tok.convert_tokens_to_ids(t)
        if tid is not None and tid >= 0:
            eos_ids.add(tid)
    ex = E.load_examples(CTX, ["niah_single_1"], max_examples=1)[0]
    ids = E.build_prompt_ids(tok, ex["context"], ex["question"], ex["answer_prefix"], DEV)
    print(f"prompt tokens={ids.shape[1]}  generating {N_TOK} greedy tokens both ways\n")

    g_rec, pf_r, dec_r, ts_r, st_r = run("recompute", ids, eos_ids)
    g_cac, pf_c, dec_c, ts_c, st_c = run("cache", ids, eos_ids, use_kernel=False)
    g_ker, pf_k, dec_k, ts_k, st_k = run("cache", ids, eos_ids, use_kernel=True)

    print(f"recompute     : {len(g_rec)} tok  prefill={pf_r:.0f}ms decode={dec_r:.0f}ms "
          f"({ts_r:.2f} tok/s) stop={st_r}")
    print(f"cache (torch) : {len(g_cac)} tok  prefill={pf_c:.0f}ms decode={dec_c:.0f}ms "
          f"({ts_c:.2f} tok/s) stop={st_c}")
    print(f"cache (kernel): {len(g_ker)} tok  prefill={pf_k:.0f}ms decode={dec_k:.0f}ms "
          f"({ts_k:.2f} tok/s) stop={st_k}")
    print(f"\nrecompute     ids: {g_rec}")
    print(f"cache (torch) ids: {g_cac}")
    print(f"cache (kernel)ids: {g_ker}")

    def report(name, g_ref, g_cmp):
        n = min(len(g_ref), len(g_cmp))
        first_div = next((i for i in range(n) if g_ref[i] != g_cmp[i]), None)
        if g_ref == g_cmp:
            print(f"✅ {name}: TOKEN-IDENTICAL ({len(g_ref)} tokens).")
            return True
        if first_div is not None:
            print(f"⚠️  {name}: diverge at step {first_div}: ref={g_ref[first_div]} "
                  f"cmp={g_cmp[first_div]} (matched first {first_div} tokens)")
        else:
            print(f"⚠️  {name}: one is a prefix of the other (len {len(g_ref)} vs {len(g_cmp)})")
        return False

    print()
    ok_torch = report("recompute vs cache (torch)", g_rec, g_cac)
    ok_kernel = report("recompute vs cache (kernel)", g_rec, g_ker)
    # Strongest check: the fused decode kernel must be token-identical to recompute.
    assert g_ker == g_rec, "cache+kernel decode diverged from recompute (token ids differ)"
    if ts_r > 0:
        print(f"\ndecode speedup (cache torch /recompute): {ts_c / ts_r:.1f}x")
        print(f"decode speedup (cache kernel/recompute): {ts_k / ts_r:.1f}x")


if __name__ == "__main__":
    main()
