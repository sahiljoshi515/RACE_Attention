"""THE KEY TEST: does delta-RACE LEARN and BEAT mean-RACE on associative recall?

Synthetic MQAR (multi-query associative recall): a sequence is a list of
(key, value) token pairs followed by query tokens. At each query position the
model must emit the value previously paired with that key. This is the standard
linear-attention recall probe -- RACE's per-bucket MEAN readout averages
colliding keys and should struggle; delta-RACE's per-bucket gated-delta memory
overwrites and should win.

We build a TINY from-scratch causal LM (embedding + N decoder layers w/ RoPE +
LM head), instantiate the attention core once as DeltaRaceLlamaAttention and once
as the baseline RaceLlamaAttention (SAME soft-hash geometry, S=24), train both on
MQAR, and report recall accuracy (measured ONLY at query positions) for each.

PASS = delta-RACE recall clearly > mean-RACE recall AND delta trains stably.

Run on GPU via the SLURM wrapper (distill/run_mqar.sbatch).
"""
import os
import sys
import math
import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from delta_race import DeltaRaceLlamaAttention      # noqa: E402
from race_llama_attention import RaceLlamaAttention  # noqa: E402  (mean-RACE baseline)


# ---------------------------------------------------------------------------
# MQAR data
# ---------------------------------------------------------------------------
# Vocabulary layout (token ids):
#   0                     : PAD / ignore
#   1                     : QUERY-marker is NOT used; queries reuse key tokens
#   keys   : [K_LO, K_HI)
#   values : [V_LO, V_HI)
# A sequence: D distinct (key, value) pairs laid out as k0 v0 k1 v1 ... then Q
# query positions each holding one of the seen keys; the TARGET at a query
# position is the value that was paired with that key. Loss/accuracy are computed
# ONLY at query positions (all other targets are ignore_index).

class MQARConfig:
    def __init__(self, num_kv_pairs=8, num_queries=8, num_keys=64, num_values=64,
                 seq_len=None):
        self.num_kv_pairs = num_kv_pairs
        self.num_queries = num_queries
        self.num_keys = num_keys
        self.num_values = num_values
        # token id layout
        self.PAD = 0
        self.K_LO = 1
        self.K_HI = self.K_LO + num_keys
        self.V_LO = self.K_HI
        self.V_HI = self.V_LO + num_values
        self.vocab_size = self.V_HI
        # 2 tokens per kv pair (key, value) + 1 token per query
        self.seq_len = 2 * num_kv_pairs + num_queries if seq_len is None else seq_len


def make_mqar_batch(cfg: MQARConfig, batch_size, device, generator):
    """Return (input_ids[B,T], targets[B,T]) where targets==-100 except at query
    positions (the value to recall). Keys within a sequence are distinct."""
    B = batch_size
    D = cfg.num_kv_pairs
    Q = cfg.num_queries
    T = 2 * D + Q
    input_ids = torch.full((B, T), cfg.PAD, dtype=torch.long)
    targets = torch.full((B, T), -100, dtype=torch.long)

    num_keys = cfg.num_keys
    num_values = cfg.num_values
    for b in range(B):
        # distinct keys for this sequence
        perm = torch.randperm(num_keys, generator=generator)[:D]
        keys = perm + cfg.K_LO
        vals = torch.randint(0, num_values, (D,), generator=generator) + cfg.V_LO
        # interleave k v k v ...
        kv = torch.empty(2 * D, dtype=torch.long)
        kv[0::2] = keys
        kv[1::2] = vals
        input_ids[b, :2 * D] = kv
        # queries: sample D-pairs' keys (with replacement), put key token at query
        # slot, target = its value.
        qidx = torch.randint(0, D, (Q,), generator=generator)
        qpos = torch.arange(2 * D, 2 * D + Q)
        input_ids[b, qpos] = keys[qidx]
        targets[b, qpos] = vals[qidx]
    return input_ids.to(device), targets.to(device)


# ---------------------------------------------------------------------------
# Tiny from-scratch causal LM
# ---------------------------------------------------------------------------
class TinyCfg:
    """Minimal config exposing the attrs RaceLlamaAttention/DeltaRaceLlamaAttention read."""
    def __init__(self, hidden_size, num_heads, num_kv_heads, vocab_size):
        self.hidden_size = hidden_size
        self.num_attention_heads = num_heads
        self.num_key_value_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.vocab_size = vocab_size
        self.attention_bias = False


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        dt = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.to(dt)) * self.w


class MLP(nn.Module):
    def __init__(self, d, hidden):
        super().__init__()
        self.gate = nn.Linear(d, hidden, bias=False)
        self.up = nn.Linear(d, hidden, bias=False)
        self.down = nn.Linear(hidden, d, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, attn, d, mlp_hidden):
        super().__init__()
        self.attn = attn
        self.ln1 = RMSNorm(d)
        self.ln2 = RMSNorm(d)
        self.mlp = MLP(d, mlp_hidden)

    def forward(self, x, pos_emb):
        h, _ = self.attn(self.ln1(x), position_embeddings=pos_emb)
        x = x + h
        x = x + self.mlp(self.ln2(x))
        return x


class TinyLM(nn.Module):
    def __init__(self, cfg: TinyCfg, attn_kind, num_layers, mlp_hidden,
                 race_L, race_K, seed, device, rope_theta=10000.0):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        blocks = []
        for li in range(num_layers):
            if attn_kind == "delta":
                attn = DeltaRaceLlamaAttention(cfg, layer_idx=li, L=race_L, Kbits=race_K,
                                               device=device, seed=seed)
            elif attn_kind == "mean":
                attn = RaceLlamaAttention(cfg, layer_idx=li, L=race_L, Kbits=race_K,
                                          device=device, seed=seed)
            else:
                raise ValueError(attn_kind)
            blocks.append(Block(attn, cfg.hidden_size, mlp_hidden))
        self.blocks = nn.ModuleList(blocks)
        self.ln_f = RMSNorm(cfg.hidden_size)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        # tie
        self.lm_head.weight = self.embed.weight
        self.rope_theta = rope_theta
        self.head_dim = cfg.head_dim

    def _rope(self, T, device, dtype):
        hd = self.head_dim
        inv_freq = 1.0 / (self.rope_theta ** (torch.arange(0, hd, 2, device=device).float() / hd))
        pos = torch.arange(T, device=device).float()
        ang = torch.outer(pos, inv_freq)          # [T, hd/2]
        emb = torch.cat([ang, ang], dim=-1)        # [T, hd]
        return emb.cos().to(dtype)[None], emb.sin().to(dtype)[None]  # [1,T,hd]

    def forward(self, input_ids):
        x = self.embed(input_ids)
        B, T, _ = x.shape
        cos, sin = self._rope(T, x.device, x.dtype)
        pos_emb = (cos.expand(B, -1, -1), sin.expand(B, -1, -1))
        for blk in self.blocks:
            x = blk(x, pos_emb)
        x = self.ln_f(x)
        return self.lm_head(x)


# ---------------------------------------------------------------------------
# Train / eval one model
# ---------------------------------------------------------------------------
def train_model(attn_kind, args, device):
    torch.manual_seed(args.seed)
    cfg = TinyCfg(args.hidden, args.heads, args.kv_heads, args.mqar_vocab)
    mqar = MQARConfig(num_kv_pairs=args.kv_pairs, num_queries=args.queries,
                      num_keys=args.num_keys, num_values=args.num_values)
    cfg.vocab_size = mqar.vocab_size
    model = TinyLM(cfg, attn_kind, args.layers, args.mlp_hidden,
                   args.race_l, args.race_k, args.seed, device).to(device)
    if args.trainable_hash:
        for blk in model.blocks:
            blk.attn.make_hash_trainable()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{attn_kind}] params={n_params/1e6:.3f}M  S={model.blocks[0].attn.S}  "
          f"vocab={mqar.vocab_size}  T={mqar.seq_len}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0,
                            betas=(0.9, 0.95))
    gen = torch.Generator().manual_seed(args.seed + (0 if attn_kind == "delta" else 12345))
    # separate fixed eval set (same for both models)
    eval_gen = torch.Generator().manual_seed(999)
    eval_ids, eval_tgt = make_mqar_batch(mqar, args.eval_batch, device, eval_gen)

    model.train()
    loss_hist = []
    t0 = time.time()
    for step in range(1, args.steps + 1):
        ids, tgt = make_mqar_batch(mqar, args.batch, device, gen)
        logits = model(ids)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), tgt.view(-1),
                               ignore_index=-100)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        loss_hist.append(loss.item())
        if step % args.log_every == 0 or step == 1:
            acc = eval_acc(model, eval_ids, eval_tgt)
            print(f"[{attn_kind}] step {step:5d}  loss {loss.item():.4f}  "
                  f"eval_recall {acc:.4f}  ({time.time()-t0:.1f}s)", flush=True)
    final_acc = eval_acc(model, eval_ids, eval_tgt)
    return {
        "kind": attn_kind,
        "final_recall": final_acc,
        "loss_first": loss_hist[0],
        "loss_last": loss_hist[-1],
        "loss_min": min(loss_hist),
        "params_M": n_params / 1e6,
    }


@torch.no_grad()
def eval_acc(model, ids, tgt):
    model.eval()
    logits = model(ids)
    pred = logits.argmax(-1)
    mask = tgt != -100
    correct = ((pred == tgt) & mask).sum().item()
    total = mask.sum().item()
    model.train()
    return correct / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--eval-batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--log-every", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    # model
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--kv-heads", type=int, default=4)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--mlp-hidden", type=int, default=256)
    # RACE soft-hash: S = L * 2^K. Spec wants S=24 -> L=6, K=2.
    ap.add_argument("--race-l", type=int, default=6)
    ap.add_argument("--race-k", type=int, default=2)
    # MQAR
    ap.add_argument("--kv-pairs", type=int, default=8)
    ap.add_argument("--queries", type=int, default=8)
    ap.add_argument("--num-keys", type=int, default=64)
    ap.add_argument("--num-values", type=int, default=64)
    ap.add_argument("--mqar-vocab", type=int, default=0)  # set from MQARConfig
    ap.add_argument("--trainable-hash", action="store_true",
                    help="unfreeze soft-hash planes/protos (identical for both models)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  torch={torch.__version__}", flush=True)
    print(f"args={vars(args)}", flush=True)

    results = {}
    for kind in ("mean", "delta"):
        print(f"\n========== TRAIN {kind}-RACE ==========", flush=True)
        results[kind] = train_model(kind, args, device)

    print("\n========== MQAR RESULTS ==========", flush=True)
    for kind in ("mean", "delta"):
        r = results[kind]
        print(f"{kind:6s}: final_recall={r['final_recall']:.4f}  "
              f"loss {r['loss_first']:.3f}->{r['loss_last']:.3f} (min {r['loss_min']:.3f})  "
              f"params={r['params_M']:.3f}M", flush=True)
    gap = results["delta"]["final_recall"] - results["mean"]["final_recall"]
    delta_stable = results["delta"]["loss_last"] < results["delta"]["loss_first"]
    beats = results["delta"]["final_recall"] > results["mean"]["final_recall"]
    print(f"\nGAP (delta - mean) = {gap:+.4f}", flush=True)
    print(f"delta trains stably (loss decreased) = {delta_stable}", flush=True)
    print(f"PASS = {beats and delta_stable}", flush=True)
    print(f"MQAR_RESULT_JSON {{'mean_recall': {results['mean']['final_recall']:.4f}, "
          f"'delta_recall': {results['delta']['final_recall']:.4f}, 'gap': {gap:.4f}, "
          f"'delta_loss_first': {results['delta']['loss_first']:.4f}, "
          f"'delta_loss_last': {results['delta']['loss_last']:.4f}, "
          f"'mean_loss_last': {results['mean']['loss_last']:.4f}, "
          f"'pass': {beats and delta_stable}}}", flush=True)


if __name__ == "__main__":
    main()
