"""Forward-latency comparison: RACE vs softmax nanochat (d24) on CPU at the
trained context length (default T=2048).

The two nanochat packages (RACE vendored / softmax) collide on the name
`nanochat`, so run ONE model per process:

  python bench_race_vs_softmax_cpu.py --model race    --pkg-dir <RACE chat dir>
  python bench_race_vs_softmax_cpu.py --model softmax --pkg-dir <softmax repo>

Random weights (forward FLOPs are weight-independent), same d24 config, so the
only difference is the attention mechanism (RACE linear prefix-scan vs softmax
SDPA). Reports median wall-clock per forward.
"""
import argparse
import os
import sys
import time
import statistics as stats

ap = argparse.ArgumentParser()
ap.add_argument("--model", choices=["race", "softmax"], required=True)
ap.add_argument("--pkg-dir", required=True, help="dir containing the `nanochat` package")
ap.add_argument("--seq-len", type=int, default=2048)
ap.add_argument("--batch", type=int, default=1)
ap.add_argument("--threads", type=int, default=None)
ap.add_argument("--warmup", type=int, default=1)
ap.add_argument("--reps", type=int, default=3)
args = ap.parse_args()

sys.path.insert(0, args.pkg_dir)
import torch  # noqa: E402

if args.threads:
    torch.set_num_threads(args.threads)
torch.manual_seed(0)

if args.model == "race":
    _REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
    _SCALING = os.path.join(_REPO, "scaling")
    os.environ["RACE_SCALING_DIR"] = _SCALING
    sys.path.insert(0, _SCALING)
    import nanochat.gpt as gpt
    from race_causal_cpu import RaceCausalCPUFn
    gpt.RaceCausalFn = RaceCausalCPUFn  # swap CUDA op -> CPU kernel composite
    cfg = gpt.GPTConfig(sequence_len=args.seq_len, vocab_size=32768, n_layer=24,
                        n_head=12, n_kv_head=12, n_embd=1536, window_pattern="L",
                        race_l=3, race_k=3, race_m=1, race_eps=1e-6)
else:
    import nanochat.gpt as gpt
    cfg = gpt.GPTConfig(sequence_len=args.seq_len, vocab_size=32768, n_layer=24,
                        n_head=12, n_kv_head=12, n_embd=1536, window_pattern="L")

device = torch.device("cpu")
model = gpt.GPT(cfg)
model.init_weights()
model = model.to(device).eval()

B, T = args.batch, args.seq_len
idx = torch.randint(0, 32768, (B, T), device=device)


def once():
    with torch.no_grad():
        model(idx)  # kv_cache=None -> full attention path


for _ in range(args.warmup):
    once()
ts = []
for _ in range(args.reps):
    t0 = time.perf_counter()
    once()
    ts.append(time.perf_counter() - t0)

print(f"RESULT model={args.model} T={T} B={B} threads={torch.get_num_threads()} "
      f"median={stats.median(ts) * 1e3:.1f}ms min={min(ts) * 1e3:.1f}ms "
      f"reps={args.reps} params={sum(p.numel() for p in model.parameters())/1e6:.0f}M")
