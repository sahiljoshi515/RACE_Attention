# Chat with the RACE-nanochat d24 base model

Run a fully **RACE-attention** GPT (depth-24, ~1.384B params, trained with the
[nanochat](https://github.com/karpathy/nanochat) base pipeline) with **one command**.

```bash
bash run.sh
```

That's it. On a SLURM cluster it grabs a GPU via `srun`; then it builds a Python
env, JIT-compiles the RACE CUDA kernel, downloads the weights from Hugging Face
([`sahilj2701/race-nanochat-d24`](https://huggingface.co/sahilj2701/race-nanochat-d24),
public, ~4.2 GB, first run only), and drops you into an interactive prompt.

```
prompt> The capital of France is
The capital of France is a country that is known for its rich history, culture, ...
```

## One-shot / options

```bash
bash run.sh --prompt "Once upon a time"                 # single completion
bash run.sh --prompt "def fibonacci(n):" --max-tokens 128
bash run.sh --temperature 0.8                           # sampling instead of greedy
```

`--max-tokens` (default 64), `--temperature` (0 = greedy), `--top-k`, `--seed`.

## Requirements

- **NVIDIA GPU + CUDA toolkit** (`nvcc`). The RACE attention op is a custom CUDA
  kernel compiled on first use. Set `TORCH_CUDA_ARCH_LIST` to your GPU's arch
  (default `9.0` for H200/Hopper; e.g. `8.0` for A100, `9.0`/`10.0` etc.).
- Python 3.10+ (the launcher creates its own venv). Deps: see `requirements.txt`
  (torch 2.9.1 cu128, tiktoken, huggingface_hub, filelock, ninja).

## This is a BASE model

It **continues text**; it is not instruction-tuned, so it won't answer questions
in a chat style. Give it completion-style prompts. It is also a **research
artifact**: at this scale the full-RACE model underperforms an identical softmax
baseline (CORE 0.098 vs 0.263). See the model card for details.

## Manual run (already on a GPU)

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
python chat.py                       # or: python chat.py --prompt "..."
```

Weights download to `./weights` by default; override with `NANOCHAT_BASE_DIR`
(point it at an existing nanochat base dir to skip the download).

## Layout

- `chat.py` — loads the checkpoint and generates (KV-cache-free `GPT.generate`,
  the only RACE-compatible inference path).
- `run.sh` — the one-command launcher (GPU acquisition + env + download + launch).
- `nanochat/` — vendored inference subset of the nanochat package (model,
  tokenizer, checkpoint loader). The RACE attention lives in the repo's
  `../scaling` + `../kernels/gpu`.
