"""CPU RACE causal attention, mirroring race_causal_cuda.RaceCausalFn.

The CUDA kernel fuses the whole causal RACE op. On CPU we compose it from the
validated `race_prefix_mean_flat` building block (kernels/cpu/race_pref.cpp) plus
a cheap probsQ gather, matching race_common.race_prefix_ref exactly:

    E[n,s,t,d] = (Σ_{τ<=t} probsK[n,τ,s]·V2[n,τ,d]) / (Σ_{τ<=t} probsK[n,τ,s] + eps)
    out[n,t,d] = Σ_s probsQ[n,t,s] · E[n,s,t,d]

Exposes RaceCausalCPUFn (autograd Function, exact grads) with the same
(probsK[N,T,S], probsQ[N,T,S], V2[N,T,D], eps) -> out[N,T,D] signature as the
CUDA RaceCausalFn, so it is a drop-in replacement on CPU tensors.
"""
import os
import sys
import torch
from torch.autograd import Function

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "kernels", "cpu"))
from race_ext import load_race  # noqa: E402

_RACE = None


def _race():
    global _RACE
    if _RACE is None:
        _RACE = load_race(verbose=False)
    return _RACE


def _flatten(probsK, V2):
    N, T, S = probsK.shape
    D = V2.shape[2]
    probsK_flat = probsK.permute(0, 2, 1).reshape(N * S, T).contiguous()          # [(n,s), t]
    V_flat = V2.unsqueeze(1).expand(N, S, T, D).reshape(N * S, T, D).contiguous()  # [(n,s), t, d]
    return probsK_flat, V_flat, N, T, S, D


class RaceCausalCPUFn(Function):
    """out = causal RACE(probsK, probsQ, V2) on CPU. Exact grads w.r.t. all three."""

    @staticmethod
    def forward(ctx, probsK, probsQ, V2, eps):
        ext = _race()
        probsK = probsK.contiguous().float()
        probsQ = probsQ.contiguous().float()
        V2 = V2.contiguous().float()
        probsK_flat, V_flat, N, T, S, D = _flatten(probsK, V2)
        E_flat = ext.race_prefix_mean_flat(probsK_flat, V_flat, float(eps))        # [(n,s),t,d]
        E = E_flat.view(N, S, T, D)
        out = torch.einsum("nts,nstd->ntd", probsQ, E)                            # [N,T,D]
        ctx.save_for_backward(probsK_flat, V_flat, probsQ, E)
        ctx.eps = float(eps)
        ctx.shape = (N, T, S, D)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        ext = _race()
        probsK_flat, V_flat, probsQ, E = ctx.saved_tensors
        N, T, S, D = ctx.shape
        grad_out = grad_out.contiguous().float()
        # d out / d probsQ
        grad_probsQ = torch.einsum("ntd,nstd->nts", grad_out, E).contiguous()      # [N,T,S]
        # d out / d E  -> feed through the prefix-mean backward
        gradE = torch.einsum("nts,ntd->nstd", probsQ, grad_out).reshape(N * S, T, D).contiguous()
        gW_flat, gV_flat = ext.race_prefix_mean_flat_bw(probsK_flat, V_flat, gradE, ctx.eps)
        grad_probsK = gW_flat.view(N, S, T).permute(0, 2, 1).reshape(N, T, S).contiguous()
        grad_V2 = gV_flat.view(N, S, T, D).sum(dim=1).contiguous()                 # V2 broadcast over S
        return grad_probsK, grad_probsQ, grad_V2, None


def race_cpu(probsK, probsQ, V2, eps=1e-6):
    return RaceCausalCPUFn.apply(probsK, probsQ, V2, eps)
