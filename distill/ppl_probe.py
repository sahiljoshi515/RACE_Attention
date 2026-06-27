"""Faithfulness probe for the RULER harness's hybrid build.

Reproduces the distillation eval (FineWeb held-out, seed=0, seq_len 4096, B=1,
teacher-forced next-token CE -> ppl) and reports it for:
  * teacher                         (expect ppl ~22)
  * student, harness build (bf16 params, no autocast)  <- what eval_ruler.py uses
  * student, distillation-faithful (fp32 params + autocast bf16)

If the two student ppls both land near the distillation-reported ~146, the harness
build is faithful and ARRR's degenerate FREE GENERATION is a genuine model property
(teacher-forced ppl 146 != coherent autoregressive generation), not a harness bug.
"""
import argparse
import torch
import torch.nn.functional as F

from transformers import AutoModelForCausalLM
from data_fineweb import get_tokenizer, make_eval_and_train
from hybrid import build_race_modules, convert_to_hybrid, make_replace_pred

MODEL = "meta-llama/Llama-3.2-3B-Instruct"
CKPT = "checkpoints/arrr_L2_K2_arrr_1k_ce_step1000.pt"
DEV = "cuda"


def ce_ppl(logits, ids):
    sl = logits[:, :-1].float().reshape(-1, logits.size(-1))
    lbl = ids[:, 1:].reshape(-1)
    ce = F.cross_entropy(sl, lbl).item()
    return ce, float(torch.exp(torch.tensor(ce)))


def _is_looping(gen, max_period=6):
    """True if the tail of `gen` is a repeated cycle of period <= max_period (degenerate
    repetition collapse). Ported from eval_ruler.py so the de-collapse gate matches eval."""
    n = len(gen)
    for p in range(1, max_period + 1):
        if n >= 4 * p and gen[-2 * p:] == gen[-4 * p:-2 * p]:
            return True
    return False


@torch.no_grad()
def greedy_generate(model, ids, max_new, eos_ids):
    """Greedy free generation with KV cache; stops on eos or repetition collapse.
    Returns (generated_token_ids, stop_reason, collapsed_bool)."""
    out = model(input_ids=ids, use_cache=True, logits_to_keep=1)
    past = out.past_key_values
    nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
    gen = [int(nxt)]
    stop = "max_new_tokens"
    for _ in range(max_new - 1):
        if int(nxt) in eos_ids:
            stop = "eos"; break
        if _is_looping(gen):
            stop = "repetition"; break
        out = model(input_ids=nxt, past_key_values=past, use_cache=True, logits_to_keep=1)
        past = out.past_key_values
        nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
        gen.append(int(nxt))
    return gen, stop, (stop == "repetition")


def free_gen_probe(model, tok, prompt, max_new=64):
    """Phase-1 de-collapse gate: greedily continue a prompt and report whether the
    hybrid collapses to repetition (the failure mode behind RULER=0)."""
    # RACE layers need incremental-decode state for cached generation (else a 1-token
    # decode step sees no context). Enable + reset, mirroring eval_ruler.generate.
    for mod in model.modules():
        if hasattr(mod, "enable_decode_cache"):
            mod.enable_decode_cache(True)
        if hasattr(mod, "reset_decode_state"):
            mod.reset_decode_state()
    msg = [{"role": "user", "content": prompt}]
    s = tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    ids = tok(s, return_tensors="pt", add_special_tokens=False).input_ids.to(DEV)
    eos_ids = {tok.eos_token_id}
    for t in getattr(tok, "additional_special_tokens_ids", []) or []:
        eos_ids.add(t)
    gen, stop, collapsed = greedy_generate(model, ids, max_new, eos_ids)
    text = tok.decode(gen, skip_special_tokens=True)
    verdict = "COLLAPSED (repetition)" if collapsed else f"OK (stop={stop})"
    print(f"free-gen [{verdict}] {len(gen)} tok: {text!r}")
    return not collapsed


def build_hybrid(bf16_params: bool):
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()
    model.requires_grad_(False)
    ck = torch.load(CKPT, map_location=DEV, weights_only=False)
    cfg = ck.get("config", {})
    pred = make_replace_pred(cfg.get("pattern", "ARRR"), model.config.num_hidden_layers)
    race = build_race_modules(model, L=cfg.get("L", 2), Kbits=cfg.get("K", 2),
                              M=cfg.get("M", 1), device=DEV, replace_pred=pred)
    race.load_state_dict(ck["race_state"], strict=True)
    if bf16_params:
        for m in race.values():
            m.to(torch.bfloat16)
    convert_to_hybrid(model, race)
    return model.eval()


def main():
    global CKPT
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=CKPT, help="hybrid RACE checkpoint to probe")
    ap.add_argument("--gen-tokens", type=int, default=64)
    ap.add_argument("--prompt", default="Tell me about the history of the printing press.")
    ap.add_argument("--no-gen", action="store_true", help="skip the free-generation collapse probe")
    args = ap.parse_args()
    CKPT = args.checkpoint
    tok = get_tokenizer()
    eval_batches, _ = make_eval_and_train(tok, seq_length=4096, batch_size=2,
                                          num_eval_batches=1, max_train_batches=1, seed=0)
    ids = eval_batches[0][:1].to(DEV)
    print(f"eval batch {tuple(ids.shape)} (FineWeb held-out, seed=0)")

    # teacher
    teacher = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()
    teacher.requires_grad_(False)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        ce, ppl = ce_ppl(teacher(input_ids=ids, use_cache=False).logits, ids)
    print(f"teacher                         ce={ce:.4f}  ppl={ppl:.1f}   (distill ~22)")
    del teacher; torch.cuda.empty_cache()

    # student — harness build (bf16 params), no autocast (matches eval_ruler.py)
    m = build_hybrid(bf16_params=True)
    with torch.no_grad():
        ce, ppl = ce_ppl(m(input_ids=ids, use_cache=False).logits, ids)
    print(f"student harness (bf16 params)   ce={ce:.4f}  ppl={ppl:.1f}   (distill ~146)")
    del m; torch.cuda.empty_cache()

    # student — distillation-faithful (fp32 params + autocast)
    m = build_hybrid(bf16_params=False)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        ce, ppl = ce_ppl(m(input_ids=ids, use_cache=False).logits, ids)
    print(f"student fp32+autocast (distill) ce={ce:.4f}  ppl={ppl:.1f}   (distill ~146)")

    if not args.no_gen:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            free_gen_probe(m, tok, args.prompt, max_new=args.gen_tokens)


if __name__ == "__main__":
    main()
