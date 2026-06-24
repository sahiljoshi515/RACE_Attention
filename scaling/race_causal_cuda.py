"""Causal RACE attention backed by the CUDA kernels (chunked parallel scan).

Exposes:
  * RaceCausalFn : autograd Function wrapping race_fused_fwd (forward) and
    race_backward (exact forward-scan backward). Operates on the collapsed bucket
    probabilities probsK/probsQ [N,T,S] and V2 [N,T,D] (all fp32).
  * RaceCausalCuda : nn.Module with the [B,H,T,D] interface; soft-hash projection
    then the fused causal scan.
"""
import os
import sys
import math
import torch
import torch.nn as nn
from torch.autograd import Function

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "kernels", "gpu"))
from race_cuda_build import load_ext  # noqa: E402

from race_common import build_planes_protos, soft_hash_probs  # noqa: E402

_EXT = None


def _ext():
    global _EXT
    if _EXT is None:
        _EXT = load_ext(verbose=True)
    return _EXT


def fwd_chunk(T):
    """Heuristic time-chunk size for the chunked kernels.

    Targets ~32-64 chunks so the readout phases have enough blocks to fill the
    SMs while keeping chunks large enough to amortize the partial-sum recompute.
    Clamped to [2048, 65536]; single chunk for short T.
    """
    if T <= 4096:
        return int(T)
    e = int(math.ceil(math.log2(max(1.0, T / 48.0))))
    e = max(11, min(16, e))   # 2048 .. 65536
    return min(1 << e, int(T))


class RaceCausalFn(Function):
    """out = causal RACE(probsK, probsQ, V2). Exact grads w.r.t. probsK/probsQ/V2."""

    @staticmethod
    def forward(ctx, probsK, probsQ, V2, eps):
        ext = _ext()
        probsK = probsK.contiguous().float()
        probsQ = probsQ.contiguous().float()
        V2 = V2.contiguous().float()
        C = fwd_chunk(probsK.shape[1])
        out = ext.race_fused_fwd(probsK, probsQ, V2, float(eps), C)
        ctx.save_for_backward(probsK, probsQ, V2)
        ctx.eps = float(eps)
        ctx.chunk = C
        return out

    @staticmethod
    def backward(ctx, grad_out):
        ext = _ext()
        probsK, probsQ, V2 = ctx.saved_tensors
        grad_out = grad_out.contiguous().float()
        gpk, gpq, gv = ext.race_backward(probsK, probsQ, V2, grad_out, ctx.eps, ctx.chunk)
        return gpk, gpq, gv, None


def race_cuda_fused(probsK, probsQ, V2, eps=1e-6):
    return RaceCausalFn.apply(probsK, probsQ, V2, eps)


class RaceCausalCuda(nn.Module):
    """Causal RACE over Q,K,V: [B,H,T,D] -> [B,H,T,D] using the CUDA kernels.

    eps=1e-6 matches the hardcoded constant in misc/race.py.
    """

    def __init__(self, d_k, Kbits, L, M, device="cuda", share_planes=True,
                 eps=1e-6, seed=0):
        super().__init__()
        self.d_k, self.Kbits, self.L, self.M = d_k, Kbits, L, M
        self.share_planes = share_planes
        self.eps = eps
        planes_T, protos_T = build_planes_protos(
            d_k, Kbits, L, M, device=device, share_planes=share_planes, seed=seed
        )
        self.register_buffer("planes_T", planes_T)
        self.register_buffer("protos_T", protos_T)

    def forward(self, Q, K, V):
        B, H, T, D = Q.shape
        probsK, probsQ, V2, N = soft_hash_probs(
            Q, K, V, self.planes_T, self.protos_T,
            self.L, self.Kbits, self.M, self.share_planes,
        )
        out_NTD = RaceCausalFn.apply(probsK, probsQ, V2, self.eps)      # [N,T,D] fp32
        out = out_NTD.view(self.M, B, H, T, D).mean(dim=0)             # [B,H,T,D]
        return out
