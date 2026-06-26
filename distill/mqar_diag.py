"""DEFINITIVE MQAR diagnostic: does per-bucket gated-delta (delta-RACE) beat the bucket-mean
RACE on associative recall — measured against a SOFTMAX baseline that proves the task is solvable?

Adds to the prior mqar_test.py probe (which was inconclusive: neither RACE variant solved the task):
  1. a full-softmax causal attention baseline (must reach high recall -> validates harness),
  2. multi-seed runs with mean +/- std,
  3. logging of the learned delta gates (beta=lr, alpha=forget) to confirm the mechanism is active.

Reuses the TinyLM harness pieces from mqar_test.py; does NOT modify delta_race.py.
Run on GPU via distill/run_mqar_diag.sbatch.
"""
import os
import sys
import math
import argparse
import statistics

import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from delta_race import DeltaRaceLlamaAttention                                   # noqa: E402
from race_llama_attention import RaceLlamaAttention, apply_rotary_pos_emb, repeat_kv  # noqa: E402
from mqar_test import (TinyCfg, RMSNorm, MLP, Block, MQARConfig,                 # noqa: E402
                       make_mqar_batch, eval_acc)


class SoftmaxAttention(nn.Module):
    """Standard causal scaled-dot-product attention, same projection shapes + RoPE as the RACE
    modules, same forward signature (returns (out, None)). The solvability ceiling."""
    def __init__(self, cfg, layer_idx=0, **kw):
        super().__init__()
        self.head_dim = cfg.head_dim
        self.num_heads = cfg.num_attention_heads
        self.nkv = cfg.num_key_value_heads
        self.groups = self.num_heads // self.nkv
        b = getattr(cfg, "attention_bias", False)
        self.q_proj = nn.Linear(cfg.hidden_size, self.num_heads * self.head_dim, bias=b)
        self.k_proj = nn.Linear(cfg.hidden_size, self.nkv * self.head_dim, bias=b)
        self.v_proj = nn.Linear(cfg.hidden_size, self.nkv * self.head_dim, bias=b)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, cfg.hidden_size, bias=b)

    def make_hash_trainable(self):       # no-op (interface parity with RACE modules)
        return self

    def forward(self, hidden_states, position_embeddings=None, **kw):
        B, T, _ = hidden_states.shape
        hd = self.head_dim
        q = self.q_proj(hidden_states).view(B, T, self.num_heads, hd).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, T, self.nkv, hd).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, T, self.nkv, hd).transpose(1, 2)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        k = repeat_kv(k, self.groups)
        v = repeat_kv(v, self.groups)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        o = o.transpose(1, 2).reshape(B, T, self.num_heads * hd)
        return self.o_proj(o), None


def build_attn(kind, cfg, li, race_L, race_K, seed, device):
    if kind == "softmax":
        return SoftmaxAttention(cfg, layer_idx=li)
    if kind == "delta":
        return DeltaRaceLlamaAttention(cfg, layer_idx=li, L=race_L, Kbits=race_K, device=device, seed=seed)
    if kind == "mean":
        return RaceLlamaAttention(cfg, layer_idx=li, L=race_L, Kbits=race_K, device=device, seed=seed)
    raise ValueError(kind)


class TinyLM(nn.Module):
    """Tiny causal LM with a pluggable attention kind (softmax / mean / delta)."""
    def __init__(self, cfg, attn_kind, num_layers, mlp_hidden, race_L, race_K, seed, device,
                 rope_theta=10000.0):
        super().__init__()
        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.blocks = nn.ModuleList([
            Block(build_attn(attn_kind, cfg, li, race_L, race_K, seed, device),
                  cfg.hidden_size, mlp_hidden) for li in range(num_layers)])
        self.ln_f = RMSNorm(cfg.hidden_size)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight
        self.rope_theta = rope_theta
        self.head_dim = cfg.head_dim

    def _rope(self, T, device, dtype):
        hd = self.head_dim
        inv = 1.0 / (self.rope_theta ** (torch.arange(0, hd, 2, device=device).float() / hd))
        ang = torch.outer(torch.arange(T, device=device).float(), inv)
        emb = torch.cat([ang, ang], dim=-1)
        return emb.cos().to(dtype)[None], emb.sin().to(dtype)[None]

    def forward(self, input_ids):
        x = self.embed(input_ids)
        B, T, _ = x.shape
        cos, sin = self._rope(T, x.device, x.dtype)
        pos = (cos.expand(B, -1, -1), sin.expand(B, -1, -1))
        for blk in self.blocks:
            x = blk(x, pos)
        return self.lm_head(self.ln_f(x))


@torch.no_grad()
def delta_gate_stats(model, ids):
    """Replicate the delta module's gate computation on the eval batch (forward-pre-hook captures
    the attn input) -> report learned beta (lr) and alpha (forget) mean/std. Confirms the delta
    mechanism is actually exercised (not collapsed to alpha~1, beta~const => ~mean)."""
    blk = model.blocks[0].attn
    if not isinstance(blk, DeltaRaceLlamaAttention):
        return None
    cap = {}
    h = blk.register_forward_pre_hook(lambda m, a: cap.__setitem__("h", a[0].detach()))
    model.eval(); model(ids); h.remove()
    hin = cap["h"]
    beta = torch.sigmoid(blk.W_beta(hin)).float()
    alpha = torch.sigmoid(blk.W_alpha(hin)).float() if getattr(blk, "W_alpha", None) is not None else torch.ones_like(beta)
    return {"beta_mean": beta.mean().item(), "beta_std": beta.std().item(),
            "alpha_mean": alpha.mean().item(), "alpha_std": alpha.std().item()}


def train_one(kind, args, seed, device):
    torch.manual_seed(seed)
    mqar = MQARConfig(num_kv_pairs=args.kv_pairs, num_queries=args.queries,
                      num_keys=args.num_keys, num_values=args.num_values)
    cfg = TinyCfg(args.hidden, args.heads, args.kv_heads, mqar.vocab_size)
    model = TinyLM(cfg, kind, args.layers, args.mlp_hidden, args.race_l, args.race_k,
                   seed, device).to(device)
    if args.trainable_hash and kind in ("mean", "delta"):
        for blk in model.blocks:
            blk.attn.make_hash_trainable()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
    gen = torch.Generator().manual_seed(seed + 7)
    eval_gen = torch.Generator().manual_seed(999)
    eval_ids, eval_tgt = make_mqar_batch(mqar, args.eval_batch, device, eval_gen)
    model.train()
    l0 = None
    for step in range(1, args.steps + 1):
        ids, tgt = make_mqar_batch(mqar, args.batch, device, gen)
        logits = model(ids)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), tgt.view(-1), ignore_index=-100)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if l0 is None:
            l0 = loss.item()
        if step % args.log_every == 0 or step == 1:
            print(f"  [{kind} s{seed}] step {step:5d} loss {loss.item():.4f} "
                  f"recall {eval_acc(model, eval_ids, eval_tgt):.4f}", flush=True)
    acc = eval_acc(model, eval_ids, eval_tgt)
    gates = delta_gate_stats(model, eval_ids) if kind == "delta" else None
    return acc, l0, loss.item(), gates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--eval-batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--log-every", type=int, default=500)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--kinds", nargs="+", default=["softmax", "mean", "delta"])
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--kv-heads", type=int, default=4)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--mlp-hidden", type=int, default=512)
    ap.add_argument("--race-l", type=int, default=6)     # S = 6*2^2 = 24
    ap.add_argument("--race-k", type=int, default=2)
    # SOLVABLE-but-stressing MQAR default: 4 pairs, 16 values (chance=1/16=0.0625)
    ap.add_argument("--kv-pairs", type=int, default=4)
    ap.add_argument("--queries", type=int, default=8)
    ap.add_argument("--num-keys", type=int, default=32)
    ap.add_argument("--num-values", type=int, default=16)
    ap.add_argument("--trainable-hash", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    chance = 1.0 / args.num_values
    print(f"device={device} torch={torch.__version__}", flush=True)
    print(f"args={vars(args)}  chance_recall={chance:.4f}", flush=True)

    res = {k: [] for k in args.kinds}
    gate_log = []
    for kind in args.kinds:
        for sd in args.seeds:
            print(f"\n===== {kind}  seed={sd} =====", flush=True)
            acc, l0, lf, gates = train_one(kind, args, sd, device)
            res[kind].append(acc)
            print(f"  -> {kind} s{sd} final_recall={acc:.4f} loss {l0:.3f}->{lf:.3f}", flush=True)
            if gates:
                gate_log.append((sd, gates))
                print(f"     gates: beta={gates['beta_mean']:.3f}±{gates['beta_std']:.3f} "
                      f"alpha={gates['alpha_mean']:.3f}±{gates['alpha_std']:.3f}", flush=True)

    def ms(xs):
        m = statistics.mean(xs); s = statistics.pstdev(xs) if len(xs) > 1 else 0.0
        return m, s
    print("\n================ MQAR DIAGNOSTIC RESULTS ================", flush=True)
    summ = {}
    for k in args.kinds:
        m, s = ms(res[k]); summ[k] = (m, s)
        print(f"  {k:8s}: recall {m:.4f} ± {s:.4f}   (seeds {[f'{x:.3f}' for x in res[k]]})", flush=True)

    sm = summ.get("softmax", (0, 0))[0]
    solvable = sm >= 0.9
    verdict = "INCONCLUSIVE (softmax did not solve the task -> harness too hard)"
    if "mean" in summ and "delta" in summ:
        dm, ds = summ["delta"]; mm, msd = summ["mean"]
        gap = dm - mm
        noise = 2 * math.sqrt(ds ** 2 + msd ** 2)   # ~2 sigma of the difference
        if solvable and gap > noise and gap > 0:
            verdict = f"GO: delta beats mean by {gap:+.4f} > 2sigma({noise:.4f}); per-bucket delta HELPS recall."
        elif solvable:
            verdict = (f"NO-GO: softmax solves it ({sm:.3f}) but delta-mean gap {gap:+.4f} "
                       f"within 2sigma({noise:.4f}) -> per-bucket delta does NOT capture the recall benefit.")
    if gate_log:
        bm = statistics.mean(g["beta_mean"] for _, g in gate_log)
        am = statistics.mean(g["alpha_mean"] for _, g in gate_log)
        print(f"  delta gates (avg over seeds): beta_mean={bm:.3f} alpha_mean={am:.3f} "
              f"({'ACTIVE' if (0.02 < bm < 0.98) else 'possibly collapsed'})", flush=True)
    print(f"\nSOFTMAX_SOLVABLE={solvable} (softmax recall {sm:.3f})", flush=True)
    print(f"VERDICT: {verdict}", flush=True)
    print(f"MQAR_DIAG_JSON {{'softmax': {summ.get('softmax',(0,0))[0]:.4f}, "
          f"'mean': {summ.get('mean',(0,0))[0]:.4f}, 'delta': {summ.get('delta',(0,0))[0]:.4f}, "
          f"'solvable': {solvable}}}", flush=True)


if __name__ == "__main__":
    main()
