"""RaceLlamaAttention: a drop-in replacement for transformers LlamaAttention that
swaps the softmax(QK^T)V core for causal RACE attention (custom CUDA RaceCausalFn).

Matches the transformers 5.5.0 LlamaAttention.forward contract:
    forward(hidden_states, position_embeddings=None, attention_mask=None,
            past_key_values=None, **kwargs) -> (attn_output, attn_weights)

Keeps Llama's exact q/k/v/o projections, RoPE, and GQA (repeat_kv); only the
attention mechanism changes. Trainable: q/k/v/o (copied from Llama) + a scalar
log_temp. Frozen: the random hyperplanes + bucket prototypes (registered buffers).
Causality is inherent in the RACE prefix scan, so no additive attention mask is
used.
"""
import os
import sys
import math
import torch
import torch.nn as nn

from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "scaling"))
from race_causal_cuda import RaceCausalFn        # noqa: E402  (custom CUDA fwd + backward)
from race_common import build_planes_protos       # noqa: E402

# Optional fused decode kernel (scaling/ is on sys.path above; lives in
# scaling/race_decode_cuda.py). The import is cheap on CPU: it does NOT compile
# the CUDA ext (that happens lazily on the first race_decode_step call inside the
# wrapper). If the module is missing we silently fall back to the torch path.
try:
    from race_decode_cuda import race_decode_step as _race_decode_step  # noqa: E402
    _HAVE_DECODE_KERNEL = True
except Exception:
    _race_decode_step = None
    _HAVE_DECODE_KERNEL = False


class RaceLlamaAttention(nn.Module):
    def __init__(self, config, layer_idx, L=2, Kbits=2, M=1, eps=1e-6, seed=0,
                 device="cuda"):
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

        # RACE soft-hash: fixed random hyperplanes + prototypes (frozen buffers),
        # one learnable temperature. planes per-head (head_dim), shared across heads.
        assert M == 1, "RaceLlamaAttention currently supports M=1 only (no ensemble axis)"
        self.L, self.Kbits, self.M, self.eps = L, Kbits, M, eps
        self.R = 1 << Kbits
        self.S = L * self.R
        planes_T, protos_T = build_planes_protos(
            self.head_dim, Kbits, L, M=1, device=device, share_planes=True, seed=seed
        )
        self.register_buffer("planes_T", planes_T)   # [head_dim, L*Kbits]  (frozen)
        self.register_buffer("protos_T", protos_T)    # [Kbits, R]            (frozen)
        self.log_temp = nn.Parameter(torch.zeros(()))  # scale = sqrt(hd)*exp(log_temp)

        # Incremental-decode state (inference only; off during training so the
        # training forward is byte-identical). The causal RACE scan reduces to two
        # running prefix sums per layer: A=Σ probsK [N,S], B=Σ probsK⊗v [N,S,hd].
        # A prefill (T>1) captures them; a decode step (T==1) advances them. This is
        # what lets the kept softmax layers use a normal KV cache (the model can then
        # feed q_len==1 through the whole stack).
        self._capture_decode_state = False
        self._dec_A = None
        self._dec_B = None
        # Cached host-side scale (= sqrt(hd)*exp(log_temp)) for the decode kernel.
        # log_temp is frozen at inference, so we compute it once instead of paying a
        # .item() D2H sync per layer per token (21 syncs/token otherwise serialize
        # against the growing kept-softmax KV work and erase the win at long context).
        self._dec_scale = None
        # When the fused decode kernel imported cleanly, prefer it for T==1 decode
        # steps. Per-module toggle so tests/eval can force the torch reference path.
        # Auto-disables (once) on any kernel build/runtime failure (see forward).
        self._use_decode_kernel = _HAVE_DECODE_KERNEL

    def enable_decode_cache(self, flag=True):
        """Turn on incremental-decode state capture (inference). Resets any state."""
        self._capture_decode_state = flag
        self._dec_A = None
        self._dec_B = None
        self._dec_scale = None   # recompute lazily (log_temp may have changed)
        return self

    def reset_decode_state(self):
        """Clear the running prefix sums (keeps capture enabled). Call before each new
        sequence's prefill so decode state never leaks across sequences."""
        self._dec_A = None
        self._dec_B = None
        self._dec_scale = None   # recompute lazily (log_temp may have changed)
        return self

    def _soft_hash(self, x):
        """x: [N,T,head_dim] -> probs [N,T,S] (fp32)."""
        N, T, hd = x.shape
        proj = (x @ self.planes_T.to(x.dtype)).view(N, T, self.L, self.Kbits)
        scale = math.sqrt(self.head_dim) * torch.exp(self.log_temp)
        logits = (proj.tanh() / scale) @ self.protos_T.to(proj.dtype)   # [N,T,L,R]
        probs = torch.softmax(logits.float(), dim=-1).reshape(N, T, self.S)
        return probs   # contiguous already (softmax output); kernel re-contiguouses if needed

    def _decode_kernel_step(self, qn, kn, vn, N):
        """Fused single-token decode via the CUDA kernel. Replaces the torch decode
        block (the ``if T==1 ...`` branch in forward, lines ~probsK/probsQ split +
        running-sum readout): soft-hash of q/k, IN-PLACE advance of _dec_A/_dec_B, and
        readout into a preallocated fp32 out [N,hd]. Returns out [N,1,hd], or None if
        the kernel raised (in which case _use_decode_kernel is cleared so the caller
        falls through to the torch reference exactly once)."""
        try:
            # Host-computed scale matches the kernel contract (sqrt(hd)*exp(log_temp)).
            # Cached once: log_temp is frozen at inference, so the .item() D2H sync runs
            # on the first decode step only, not per layer per token.
            if self._dec_scale is None:
                self._dec_scale = math.sqrt(self.head_dim) * float(torch.exp(self.log_temp))
            # Preallocated, contiguous fp32 output (wrapper does NOT .contiguous() out).
            out_nd = torch.empty(N, self.head_dim, device=qn.device, dtype=torch.float32)
            _race_decode_step(
                qn[:, 0].float(), kn[:, 0].float(), vn[:, 0].float(),
                self.planes_T.float(), self.protos_T.float(),
                self._dec_A, self._dec_B, out_nd,
                self._dec_scale, self.eps, self.L, self.Kbits,
            )
            return out_nd.unsqueeze(1)   # [N,1,hd]
        except Exception:
            # Build/runtime failure: degrade gracefully, once. _dec_A/_dec_B updates are
            # in place; on failure the torch fallback recomputes them for this step.
            self._use_decode_kernel = False
            return None

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

        # Fused-decode fast path. A T==1 decode step with captured prefix state is the
        # launch-bound hot path; the kernel fuses soft-hash + in-place A/B update +
        # readout in one launch, so we skip the (otherwise redundant) torch soft-hash
        # on line ~probs= below. It mirrors the torch decode math (the if-branch
        # ~probsK/probsQ split + running-sum block); _dec_A/_dec_B are advanced IN
        # PLACE to exactly the values the torch path would produce, so subsequent steps
        # (kernel or torch) are interchangeable. Gated by _use_decode_kernel and CUDA;
        # any failure disables the kernel once and falls through to the torch path.
        is_decode = (T == 1 and self._capture_decode_state and self._dec_A is not None)
        if is_decode and self._use_decode_kernel and qn.is_cuda:
            out = self._decode_kernel_step(qn, kn, vn, N)
            if out is not None:
                out = out.view(B, self.num_heads, T, hd).transpose(1, 2).reshape(B, T, self.num_heads * hd)
                out = self.o_proj(out.to(hidden_states.dtype))
                return out, None
            # out is None -> kernel disabled itself; fall through to torch reference.

        # One batched soft-hash over [Q;K] halves the kernel launches (the per-token
        # hot path is launch-bound during decode). Slices stay contiguous.
        probs = self._soft_hash(torch.cat([qn, kn], dim=0))   # [2N,T,S] fp32
        probsQ, probsK = probs[:N], probs[N:]

        if T == 1 and self._capture_decode_state and self._dec_A is not None:
            # Incremental decode: advance the running prefix sums by the new token and
            # read out. Exactly race_prefix_ref's per-step math (B/(A+eps), then probsQ·).
            pk = probsK[:, 0]                           # [N,S]
            pq = probsQ[:, 0]                           # [N,S]
            v1 = vn[:, 0].float()                       # [N,hd]
            self._dec_A = self._dec_A + pk                                   # [N,S]
            self._dec_B = self._dec_B + pk.unsqueeze(-1) * v1.unsqueeze(1)   # [N,S,hd]
            E = self._dec_B / (self._dec_A.unsqueeze(-1) + self.eps)         # [N,S,hd]
            out = torch.einsum("ns,nsd->nd", pq, E).unsqueeze(1)            # [N,1,hd]
        else:
            out = RaceCausalFn.apply(probsK, probsQ, vn.float(), self.eps)   # [N,T,hd] fp32
            if self._capture_decode_state:
                # Capture end-of-prefix state for the subsequent decode steps. einsum
                # avoids materializing B_pref [N,T,S,hd] (sum over T directly).
                self._dec_A = probsK.sum(1)                                  # [N,S]
                self._dec_B = torch.einsum("nts,ntd->nsd", probsK, vn.float())  # [N,S,hd]

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
        """Convert the frozen soft-hash geometry buffers (planes_T, protos_T) into
        trainable nn.Parameters so the LSH hyperplanes/prototypes can be learned.
        Gradients reach them through _soft_hash -> probsK/probsQ -> RaceCausalFn
        (the custom CUDA op only needs grads w.r.t. probs; autograd handles the rest).
        Idempotent; keeps the current values (e.g. loaded from a checkpoint)."""
        for name in ("planes_T", "protos_T"):
            if name in self._buffers:                      # currently a frozen buffer
                val = self._buffers.pop(name).detach().clone()
                setattr(self, name, nn.Parameter(val))     # -> trainable Parameter
        return self
