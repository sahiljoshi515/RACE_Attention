"""Focused diagnosis of the cudagraph capture for teacher vs hybrid at small ctx.

Findings to resolve:
  (a) teacher gate 'failed' but graph output was the eager output SHIFTED by the warmup
      steps -> bookkeeping bug in the gate, graph is faithful. Verified here by recording
      warmup tokens so eager and graph chains line up.
  (b) hybrid graph emitted all-zero token ids -> real capture bug. Probe whether
      static_logits is zero/NaN after capture, and whether an eager step from the SAME
      post-warmup state is healthy (isolates kernel-under-capture vs model).
"""
import torch
import eval_ruler as E
from transformers import StaticCache

DEV = "cuda"
CKPT = "checkpoints/arrr_L2_K2_arrr_1k_ce_step1000.pt"


def reset_race(m):
    for mod in m.modules():
        if hasattr(mod, "reset_decode_state"):
            mod.reset_decode_state()


def force_kernel(m, flag):
    n = 0
    for mod in m.modules():
        if mod.__class__.__name__ == "RaceLlamaAttention":
            mod._use_decode_kernel = flag; n += 1
    return n


def stat(t):
    return (f"min={t.min().item():.3e} max={t.max().item():.3e} "
            f"nan={torch.isnan(t).any().item()} argmax={t.argmax(-1).flatten()[:4].tolist()}")


@torch.no_grad()
def diagnose(model, label, T=4096, n=6):
    print(f"\n========== {label}  T={T} ==========")
    vocab = model.config.vocab_size
    ids = torch.randint(0, vocab, (1, T), device=DEV)

    # ---- reference: fully eager chain (record EVERY token incl. warmup window) ----
    reset_race(model)
    cache = StaticCache(model.config, max_cache_len=T + 32)
    out = model(input_ids=ids, past_key_values=cache, use_cache=True,
                cache_position=torch.arange(T, device=DEV), logits_to_keep=1)
    nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
    eager_chain = []
    pos = torch.tensor([T], device=DEV)
    eager_first_logits = None
    for i in range(n):
        out = model(input_ids=nxt, past_key_values=cache, use_cache=True,
                    cache_position=pos, logits_to_keep=1)
        lg = out.logits[:, -1, :]
        if i == 0:
            eager_first_logits = lg.clone()
        nxt = lg.argmax(-1, keepdim=True)
        eager_chain.append(nxt.item())
        pos = pos + 1
    print("eager chain :", eager_chain)

    # ---- graph path: warmup (record tokens), then capture, then replay ----
    reset_race(model)
    cache = StaticCache(model.config, max_cache_len=T + 32)
    out = model(input_ids=ids, past_key_values=cache, use_cache=True,
                cache_position=torch.arange(T, device=DEV), logits_to_keep=1)
    static_input = out.logits[:, -1, :].argmax(-1, keepdim=True)
    static_pos = torch.tensor([T], device=DEV)
    graph_chain = []

    n_warm = 3
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(n_warm):
            out = model(input_ids=static_input, past_key_values=cache, use_cache=True,
                        cache_position=static_pos, logits_to_keep=1)
            tok = out.logits[:, -1, :].argmax(-1, keepdim=True)
            graph_chain.append(tok.item())
            static_input.copy_(tok); static_pos.add_(1)
    torch.cuda.current_stream().wait_stream(s)

    # sanity: an EAGER step from the post-warmup state (not captured)
    probe = model(input_ids=static_input, past_key_values=cache, use_cache=True,
                  cache_position=static_pos, logits_to_keep=1)
    print("post-warmup EAGER-step logits:", stat(probe.logits[:, -1, :]))
    # NOTE: that eager probe advanced the cache once; reset position bookkeeping is fine
    # for diagnosis since we only inspect logits health, not the chain past here.

    g = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(g):
            out = model(input_ids=static_input, past_key_values=cache, use_cache=True,
                        cache_position=static_pos, logits_to_keep=1)
            static_logits = out.logits[:, -1, :]
    except Exception as e:
        print("CAPTURE RAISED:", repr(e)[:200]); return
    print("post-capture static_logits :", stat(static_logits))
    for _ in range(n - n_warm):
        g.replay()
        tok = static_logits.argmax(-1, keepdim=True)
        graph_chain.append(tok.item())
        static_input.copy_(tok); static_pos.add_(1)
    print("graph chain :", graph_chain, "(first", n_warm, "are eager warmup)")
    print("MATCH eager==graph:", eager_chain == graph_chain)


def main():
    print("GPU:", torch.cuda.get_device_name(0))
    tm, _, _ = E.build_model("teacher", None, "sdpa", DEV); force_kernel(tm, False)
    diagnose(tm, "TEACHER")
    del tm; torch.cuda.empty_cache()
    hm, _, _ = E.build_model("arrr", CKPT, "sdpa", DEV, decode_mode="cache")
    nk = force_kernel(hm, True); print(f"[hybrid] kernel ON x{nk}")
    diagnose(hm, "HYBRID kernel=ON")
    # also try hybrid with kernel OFF (torch path) to confirm it's NOT capturable
    force_kernel(hm, False)
    diagnose(hm, "HYBRID kernel=OFF (torch path, expected to break)")


if __name__ == "__main__":
    main()
