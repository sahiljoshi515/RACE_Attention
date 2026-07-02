#!/usr/bin/env python3
"""Interactive chat / completion with the RACE-nanochat d24 base model.

This is a BASE (not chat-tuned) model: it *continues* text rather than following
instructions. Completion-style prompts work best ("The capital of France is",
"def fibonacci(n):", "Once upon a time").

Usage:
    python chat.py                         # interactive REPL
    python chat.py --prompt "..."          # one-shot completion
    python chat.py --prompt "..." --temperature 0.8 --max-tokens 128

Weights auto-download from Hugging Face (sahilj2701/race-nanochat-d24, public) on
first run, into ./weights (override with NANOCHAT_BASE_DIR). Requires an NVIDIA GPU
+ CUDA toolkit (the RACE attention kernel is JIT-compiled on first token).
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)  # repo root; chat/ lives directly under it

# 1) Make the vendored `nanochat` package importable, plus the repo's RACE scaling
#    code. gpt.py does `from race_causal_cuda import RaceCausalFn` expecting the
#    scaling dir on sys.path; race_causal_cuda.py then self-locates kernels/gpu.
sys.path.insert(0, HERE)  # -> `import nanochat` resolves to the vendored package
_SCALING = os.path.join(REPO, "scaling")
sys.path.insert(0, _SCALING)
os.environ.setdefault("RACE_SCALING_DIR", _SCALING)

# 2) Weights location, laid out the way nanochat expects (NANOCHAT_BASE_DIR).
_DEFAULT_BASE = os.path.join(HERE, "weights")
BASE_DIR = os.environ.get("NANOCHAT_BASE_DIR", _DEFAULT_BASE)
os.environ["NANOCHAT_BASE_DIR"] = BASE_DIR

HF_REPO = "sahilj2701/race-nanochat-d24"


def ensure_weights():
    ckpt = os.path.join(BASE_DIR, "base_checkpoints", "d24", "model_005568.pt")
    tok = os.path.join(BASE_DIR, "tokenizer", "tokenizer.pkl")
    if os.path.exists(ckpt) and os.path.exists(tok):
        return
    print(f"[chat] downloading weights from HF '{HF_REPO}' -> {BASE_DIR}", flush=True)
    print("[chat] (~4.2 GB, first run only) ...", flush=True)
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id=HF_REPO, repo_type="model", local_dir=BASE_DIR,
        allow_patterns=["base_checkpoints/*", "tokenizer/*"],
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompt", default=None,
                    help="one-shot prompt; omit to enter the interactive REPL")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="0.0 = greedy; >0 enables sampling")
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model-tag", default="d24")
    ap.add_argument("--step", type=int, default=5568)
    args = ap.parse_args()

    ensure_weights()

    import torch  # noqa: F401  (import after env is set)
    from nanochat.common import compute_init, autodetect_device_type
    from nanochat.checkpoint_manager import load_model

    device_type = autodetect_device_type()
    ddp, rank, local_rank, world, device = compute_init(device_type)
    print(f"[chat] loading base model '{args.model_tag}' (step {args.step}) on {device} ...",
          flush=True)
    model, tokenizer, meta = load_model("base", device, phase="eval",
                                        model_tag=args.model_tag, step=args.step)
    model.eval()
    bos = tokenizer.get_bos_token_id()
    print(f"[chat] ready (step={meta.get('step')}).", flush=True)

    def generate(prompt):
        ids = tokenizer.encode(prompt, prepend=bos)
        if len(ids) < 2:  # the KV-cache-free forward asserts T > 1
            ids = tokenizer.encode(prompt + " ", prepend=bos)
        gen = list(model.generate(ids, max_tokens=args.max_tokens,
                                  temperature=args.temperature,
                                  top_k=args.top_k, seed=args.seed))
        return tokenizer.decode(gen)

    if args.prompt is not None:
        sys.stdout.write(args.prompt + generate(args.prompt) + "\n")
        return

    bar = "=" * 72
    print(bar)
    print(" RACE-nanochat d24 (base model) — interactive completion")
    print(" This is a BASE model: it CONTINUES text, it does not follow instructions.")
    print(f" settings: max_tokens={args.max_tokens}  temperature={args.temperature}"
          f"  top_k={args.top_k}")
    print(" Type a prompt + Enter.  'exit'/'quit' or Ctrl-D to leave.")
    print(bar)
    while True:
        try:
            prompt = input("\nprompt> ")
        except EOFError:
            print()
            break
        if prompt.strip().lower() in ("exit", "quit"):
            break
        if not prompt.strip():
            continue
        out = generate(prompt)
        print(f"\n{prompt}\033[1m{out}\033[0m")


if __name__ == "__main__":
    main()
