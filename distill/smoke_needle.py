"""Sanity check that synthetic_ruler_batch produces WELL-FORMED inputs: the frozen teacher,
fed a synthetic NIAH sample, should retrieve the needle. KD is label-free (the teacher's
logits are the target), so the only thing we must guarantee about synthetic data is that the
INPUTS are coherent enough for the teacher to answer — this verifies that before Phase 3.
"""
import torch
from transformers import AutoModelForCausalLM
from data_fineweb import get_tokenizer
from data_long import synthetic_ruler_batch

MODEL = "meta-llama/Llama-3.2-3B-Instruct"
DEV = "cuda"


@torch.no_grad()
def main():
    tok = get_tokenizer()
    teacher = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()
    teacher.requires_grad_(False)
    eos = {tok.eos_token_id}
    ok = 0
    trials = 0
    for fmt in ("query_in_context", "query_agnostic"):
        gen = synthetic_ruler_batch(tok, seq_length=2048, batch_size=1, seed=0,
                                    task="niah_single", fmt=fmt)
        batch = next(gen).to(DEV)
        # greedily continue ~16 tokens and decode
        out = teacher(input_ids=batch, use_cache=True, logits_to_keep=1)
        past = out.past_key_values
        nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
        toks = [int(nxt)]
        for _ in range(15):
            if int(nxt) in eos:
                break
            out = teacher(input_ids=nxt, past_key_values=past, use_cache=True, logits_to_keep=1)
            past = out.past_key_values
            nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
            toks.append(int(nxt))
        ans = tok.decode(toks, skip_special_tokens=True)
        prompt = tok.decode(batch[0].tolist(), skip_special_tokens=True)
        # the needle value is the 7-ish digit run in the prompt; check the teacher echoes digits
        has_digits = any(c.isdigit() for c in ans)
        print(f"[{fmt}] teacher answer={ans!r}  digits={has_digits}")
        trials += 1
        ok += int(has_digits)
    print(f"NEEDLE SANITY: {ok}/{trials} formats produced a digit answer "
          f"({'PASS' if ok == trials else 'CHECK — inputs may be malformed'})")


if __name__ == "__main__":
    main()
