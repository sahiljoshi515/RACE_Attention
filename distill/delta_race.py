"""Delta-RACE: soft-bucketed linear attention with a per-bucket GATED-DELTA value
memory (Widrow-Hoff overwrite + forget gate), the PyTorch reference the CUDA kernel
will mirror.

Baseline RACE keeps a per-bucket running MEAN, so colliding keys average together and
associative recall fails. Delta-RACE replaces that mean with a gated-delta memory:

    M_t[s] = (alpha_t - beta_t * p_{t,s}) * M_{t-1}[s] + (beta_t * p_{t,s}) * v_t
           = alpha_t * M_{t-1}[s] + beta_t * p_{t,s} * (v_t - M_{t-1}[s])    (alpha_t=1)
    out_t  = Σ_s probsQ[t,s] * M_t[s]

with beta_t = sigmoid(W_beta·h_t) (data-dependent learning rate) and alpha_t =
sigmoid(W_alpha·h_t) (forget gate), each a per-(B,H,T) scalar. Buckets are independent,
so this is S independent gated linear recurrences with scalar gate
g_{t,s} = (alpha_t - beta_t*p_{t,s}) and hd-vector input u_{t,s} = beta_t*p_{t,s}*v_t.
g_{t,s} can go negative (delta over-correction) and that is intentional -- do NOT clamp.

This module reuses RaceLlamaAttention's projections / soft-hash / RoPE / GQA.
"""
import os
import sys
import math
import torch
import torch.nn as nn

from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "scaling"))
from race_common import build_planes_protos       # noqa: E402

# Reuse the baseline mean-RACE module for apples-to-apples learning comparisons.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from race_llama_attention import RaceLlamaAttention  # noqa: E402,F401  (MeanRace path)


# ---------------------------------------------------------------------------
# Core delta-RACE scan (operates on already-hashed probs + values + gates)
# ---------------------------------------------------------------------------
# Shapes (per the spec):
#   probsQ, probsK : [N, T, S]   soft bucket membership of query / key
#   v              : [N, T, hd]  values
#   beta, alpha    : [N, T]      per-token scalar learning rate / forget gate
# Returns out : [N, T, hd].

def delta_race_scan_ref(probsQ, probsK, v, beta, alpha, eps=1e-6):
    """SEQUENTIAL ground-truth reference: explicit loop over T building M_t and out_t.
    Slow but obviously correct; the canonical definition of delta-RACE."""
    N, T, S = probsK.shape
    hd = v.shape[-1]
    dtype = v.dtype
    device = v.device

    M = torch.zeros(N, S, hd, dtype=dtype, device=device)        # M_0 = 0
    out = torch.empty(N, T, hd, dtype=dtype, device=device)
    for t in range(T):
        pk = probsK[:, t]                       # [N,S]
        pq = probsQ[:, t]                       # [N,S]
        vt = v[:, t]                            # [N,hd]
        bt = beta[:, t].unsqueeze(-1)           # [N,1]
        at = alpha[:, t].unsqueeze(-1)          # [N,1]
        bp = (bt * pk)                          # [N,S]   beta_t * p_{t,s}
        g = (at - bp).unsqueeze(-1)             # [N,S,1] scalar gate (may be < 0)
        u = bp.unsqueeze(-1) * vt.unsqueeze(1)  # [N,S,hd] input beta_t*p*v_t
        M = g * M + u                           # [N,S,hd] gated-delta update
        out[:, t] = torch.einsum("ns,nsd->nd", pq, M)  # readout Σ_s probsQ*M_t[s]
    return out


def delta_race_scan(probsQ, probsK, v, beta, alpha, eps=1e-6, chunk=128):
    """VECTORIZED / chunked gated scan -- the path training uses.

    Each bucket s is an independent gated linear recurrence
        M_t = g_{t,s} * M_{t-1} + u_{t,s},   g_{t,s} = alpha_t - beta_t*p_{t,s},
                                             u_{t,s} = beta_t*p_{t,s}*v_t.
    A fully-parallel closed form needs the gate-product ratios prod_{j=i+1..t} g_j,
    which is naturally written as G[t]/G[i] (cumprod-then-divide). But g_{t,s} can be
    negative AND can pass through ~0 (a bucket that is fully overwritten when
    alpha_t ~= beta_t*p), so 1/G[i] blows up and the parallel form produces NaN/inf.

    So this chunked path does the recurrence SEQUENTIALLY *within* each chunk (a small
    Python loop of length <=chunk), but stays fully vectorized over the batch (N),
    buckets (S) and head dim (hd) -- the heavy axes. Per-bucket state M is carried
    across chunks. No gate-product division anywhere, so it is stable for any gate
    (including negative / near-zero) and matches the reference to <1e-4 in fp32.

    `chunk` only controls how many timesteps' inputs are sliced/materialized at once
    (a memory knob); the result is independent of `chunk`.
    """
    N, T, S = probsK.shape
    hd = v.shape[-1]
    dtype = v.dtype
    device = v.device

    out = torch.empty(N, T, hd, dtype=dtype, device=device)
    M = torch.zeros(N, S, hd, dtype=dtype, device=device)        # carried state M_{t-1}

    for c0 in range(0, T, chunk):
        c1 = min(c0 + chunk, T)
        pk = probsK[:, c0:c1]                       # [N,L,S]
        pq = probsQ[:, c0:c1]                        # [N,L,S]
        vc = v[:, c0:c1]                             # [N,L,hd]
        bc = beta[:, c0:c1].unsqueeze(-1)            # [N,L,1]
        ac = alpha[:, c0:c1].unsqueeze(-1)           # [N,L,1]

        bp = bc * pk                                 # [N,L,S]   beta_t * p_{t,s}
        g = ac - bp                                  # [N,L,S]   gate (may be < 0)
        # u[:,j] = beta_t*p_{t,s} * v_t  -> [N,L,S,hd]
        u = bp.unsqueeze(-1) * vc.unsqueeze(2)       # [N,L,S,hd]

        L = c1 - c0
        for j in range(L):
            M = g[:, j].unsqueeze(-1) * M + u[:, j]  # [N,S,hd] gated-delta update
            out[:, c0 + j] = torch.einsum("ns,nsd->nd", pq[:, j], M)  # readout

    return out


# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------
class DeltaRaceLlamaAttention(nn.Module):
    """Delta-RACE drop-in mirroring RaceLlamaAttention's __init__/forward interface,
    plus W_beta / W_alpha gate projections. Training/prefill uses delta_race_scan.

    forward(hidden_states, position_embeddings, ...) -> (attn_output, None) with the
    same output shape as RaceLlamaAttention. Decode is TODO (see note at bottom)."""

    def __init__(self, config, layer_idx, L=2, Kbits=2, M=1, eps=1e-6, seed=0,
                 device="cuda", learn_alpha=True):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        self.hidden_size = config.hidden_size
        bias = getattr(config, "attention_bias", False)

        # Same projections as LlamaAttention (weights copied in from the teacher).
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=bias)

        # Gate projections: one scalar per head per token for beta (learning rate) and
        # alpha (forget gate). [hidden_size -> num_heads].
        self.W_beta = nn.Linear(self.hidden_size, self.num_heads, bias=True)
        self.W_alpha = nn.Linear(self.hidden_size, self.num_heads, bias=True)
        self.learn_alpha = learn_alpha
        # Init: beta toward a moderate learning rate, alpha toward ~1 (retain memory).
        nn.init.zeros_(self.W_beta.weight)
        nn.init.zeros_(self.W_beta.bias)          # sigmoid(0)=0.5 learning rate
        nn.init.zeros_(self.W_alpha.weight)
        nn.init.constant_(self.W_alpha.bias, 4.0)  # sigmoid(4)~=0.982 -> near-pure delta

        # RACE soft-hash: fixed random hyperplanes + prototypes (frozen buffers),
        # one learnable temperature. planes per-head (head_dim), shared across heads.
        assert M == 1, "DeltaRaceLlamaAttention currently supports M=1 only (no ensemble axis)"
        self.L, self.Kbits, self.M, self.eps = L, Kbits, M, eps
        self.R = 1 << Kbits
        self.S = L * self.R
        planes_T, protos_T = build_planes_protos(
            self.head_dim, Kbits, L, M=1, device=device, share_planes=True, seed=seed
        )
        self.register_buffer("planes_T", planes_T)   # [head_dim, L*Kbits]  (frozen)
        self.register_buffer("protos_T", protos_T)    # [Kbits, R]            (frozen)
        self.log_temp = nn.Parameter(torch.zeros(()))  # scale = sqrt(hd)*exp(log_temp)

    def _soft_hash(self, x):
        """x: [N,T,head_dim] -> probs [N,T,S] (fp32). Identical to RaceLlamaAttention."""
        N, T, hd = x.shape
        proj = (x @ self.planes_T.to(x.dtype)).view(N, T, self.L, self.Kbits)
        scale = math.sqrt(self.head_dim) * torch.exp(self.log_temp)
        logits = (proj.tanh() / scale) @ self.protos_T.to(proj.dtype)   # [N,T,L,R]
        probs = torch.softmax(logits.float(), dim=-1).reshape(N, T, self.S)
        return probs

    def forward(self, hidden_states, position_embeddings=None, attention_mask=None,
                past_key_values=None, **kwargs):
        B, T, _ = hidden_states.shape
        hd = self.head_dim
        q = self.q_proj(hidden_states).view(B, T, self.num_heads, hd).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, T, self.num_kv_heads, hd).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, T, self.num_kv_heads, hd).transpose(1, 2)

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        k = repeat_kv(k, self.num_key_value_groups)   # [B, num_heads, T, hd]
        v = repeat_kv(v, self.num_key_value_groups)

        N = B * self.num_heads
        qn = q.reshape(N, T, hd)
        kn = k.reshape(N, T, hd)
        vn = v.reshape(N, T, hd)

        # Batched soft-hash over [Q;K].
        probs = self._soft_hash(torch.cat([qn, kn], dim=0))   # [2N,T,S] fp32
        probsQ, probsK = probs[:N], probs[N:]

        # Gates: [B,T,num_heads] -> [N,T] with N ordered (B,H) to match the q/k/v
        # reshape (view(B,T,H,hd).transpose(1,2).reshape(B*H,T,hd) -> (B,H) major).
        beta = torch.sigmoid(self.W_beta(hidden_states).float())   # [B,T,H]
        if self.learn_alpha:
            alpha = torch.sigmoid(self.W_alpha(hidden_states).float())
        else:
            alpha = torch.ones_like(beta)
        beta = beta.permute(0, 2, 1).reshape(N, T)    # [N,T]  (B,H) major
        alpha = alpha.permute(0, 2, 1).reshape(N, T)

        out = delta_race_scan(probsQ, probsK, vn.float(), beta, alpha, self.eps)  # [N,T,hd]

        out = out.view(B, self.num_heads, T, hd).transpose(1, 2).reshape(B, T, self.num_heads * hd)
        out = self.o_proj(out.to(hidden_states.dtype))
        return out, None

    @torch.no_grad()
    def copy_projections_from(self, llama_attn):
        """Copy q/k/v/o weights (and biases if any) from a LlamaAttention module."""
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            getattr(self, name).load_state_dict(getattr(llama_attn, name).state_dict())
        return self

    def make_hash_trainable(self):
        """Convert frozen soft-hash buffers (planes_T, protos_T) into trainable
        nn.Parameters. Idempotent; keeps the current values."""
        for name in ("planes_T", "protos_T"):
            if name in self._buffers:
                val = self._buffers.pop(name).detach().clone()
                setattr(self, name, nn.Parameter(val))
        return self


# Apples-to-apples baseline: the existing mean-RACE module is imported above as
# RaceLlamaAttention. The learning test can instantiate both on the same config/seed
# (identical soft-hash geometry) and compare delta vs mean.
MeanRaceAttention = RaceLlamaAttention


# ---------------------------------------------------------------------------
# Self-tests (CPU, tiny). Run: python distill/delta_race.py
# ---------------------------------------------------------------------------
def _self_test():
    torch.manual_seed(0)
    dev = "cpu"

    # --- (1) vectorized scan == sequential reference, T not a multiple of chunk ---
    N, T, S, hd = 2, 37, 8, 16
    probsK = torch.softmax(torch.randn(N, T, S), dim=-1)
    probsQ = torch.softmax(torch.randn(N, T, S), dim=-1)
    v = torch.randn(N, T, hd)
    beta = torch.sigmoid(torch.randn(N, T))
    alpha = torch.sigmoid(torch.randn(N, T))

    ref = delta_race_scan_ref(probsQ, probsK, v, beta, alpha)
    vec = delta_race_scan(probsQ, probsK, v, beta, alpha, chunk=8)  # tiny chunk on purpose
    d_scan = (ref - vec).abs().max().item()
    ok_scan = d_scan < 1e-4
    print(f"[1] scan vec vs ref: max|diff|={d_scan:.3e}  -> {'PASS' if ok_scan else 'FAIL'}")

    # also check default-chunk path (chunk > T, single chunk)
    vec2 = delta_race_scan(probsQ, probsK, v, beta, alpha, chunk=128)
    d_scan2 = (ref - vec2).abs().max().item()
    ok_scan2 = d_scan2 < 1e-4
    print(f"    scan vec(chunk=128) vs ref: max|diff|={d_scan2:.3e}  -> {'PASS' if ok_scan2 else 'FAIL'}")

    # --- (2) causality: changing v at position t leaves out_{<t} bit-identical ---
    t_edit = 20
    v2 = v.clone()
    v2[:, t_edit] += 3.14159
    out_a = delta_race_scan_ref(probsQ, probsK, v, beta, alpha)
    out_b = delta_race_scan_ref(probsQ, probsK, v2, beta, alpha)
    causal_prefix_identical = torch.equal(out_a[:, :t_edit], out_b[:, :t_edit])
    changed_after = not torch.equal(out_a[:, t_edit:], out_b[:, t_edit:])
    ok_causal = causal_prefix_identical and changed_after
    print(f"[2] causality (ref): prefix<{t_edit} identical={causal_prefix_identical}, "
          f"changed>={t_edit}={changed_after} -> {'PASS' if ok_causal else 'FAIL'}")
    # same on the vectorized path
    outv_a = delta_race_scan(probsQ, probsK, v, beta, alpha, chunk=8)
    outv_b = delta_race_scan(probsQ, probsK, v2, beta, alpha, chunk=8)
    causal_vec = torch.equal(outv_a[:, :t_edit], outv_b[:, :t_edit])
    print(f"    causality (vec): prefix<{t_edit} identical={causal_vec} "
          f"-> {'PASS' if causal_vec else 'FAIL'}")

    # --- (3) module forward shape/dtype + bf16-autocast safe ---
    class _Cfg:
        hidden_size = 32
        num_attention_heads = 4
        num_key_value_heads = 2
        attention_bias = False
    cfg = _Cfg()
    Bsz, Tt = 2, 16
    attn = DeltaRaceLlamaAttention(cfg, layer_idx=0, L=2, Kbits=2, device=dev)
    hs = torch.randn(Bsz, Tt, cfg.hidden_size)
    # RoPE cos/sin: [B,T,head_dim].
    hdim = cfg.hidden_size // cfg.num_attention_heads
    pos = torch.arange(Tt).float()
    inv_freq = 1.0 / (10000 ** (torch.arange(0, hdim, 2).float() / hdim))
    ang = pos[:, None] * inv_freq[None, :]            # [T, hdim/2]
    emb = torch.cat([ang, ang], dim=-1)               # [T, hdim]
    cos = emb.cos()[None].expand(Bsz, -1, -1)
    sin = emb.sin()[None].expand(Bsz, -1, -1)

    out, w = attn(hs, position_embeddings=(cos, sin))
    ok_shape = (out.shape == (Bsz, Tt, cfg.hidden_size)) and (w is None)
    print(f"[3] forward shape: {tuple(out.shape)} expected {(Bsz, Tt, cfg.hidden_size)} "
          f"-> {'PASS' if ok_shape else 'FAIL'}")

    ok_autocast = True
    try:
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            out_bf, _ = attn(hs, position_embeddings=(cos, sin))
        ok_autocast = (out_bf.shape == (Bsz, Tt, cfg.hidden_size)) and torch.isfinite(out_bf.float()).all().item()
    except Exception as e:
        ok_autocast = False
        print(f"    autocast error: {e}")
    print(f"[3] bf16-autocast forward: dtype={out_bf.dtype if ok_autocast else 'n/a'} "
          f"-> {'PASS' if ok_autocast else 'FAIL'}")

    all_ok = ok_scan and ok_scan2 and ok_causal and causal_vec and ok_shape and ok_autocast
    print(f"\nALL CHECKS: {'PASS' if all_ok else 'FAIL'}")
    return all_ok


if __name__ == "__main__":
    ok = _self_test()
    sys.exit(0 if ok else 1)


# Decode (single-token incremental) is TODO: a decode step would advance the per-bucket
# M state by one token (M <- (alpha - beta*p)*M + beta*p*v; out = Σ probsQ*M) and must
# match this prefill scan token-for-token. Omitted here pending the CUDA kernel design.
