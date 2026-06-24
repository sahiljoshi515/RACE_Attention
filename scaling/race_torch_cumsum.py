"""Causal RACE attention via the pure-PyTorch cumsum scan (the reference path).

Same soft-hash projection as RaceCausalCuda, but the causal scan is the
materialized cumsum from misc/race.py (race_common.race_prefix_ref).  This
materializes B_pref [N,T,S,D] and therefore OOMs at long sequence lengths --
that is the limitation the fused CUDA kernel removes.
"""
import torch
import torch.nn as nn

from race_common import build_planes_protos, soft_hash_probs, race_prefix_ref


class RaceCumsumCausal(nn.Module):
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
        # fp32 scan (matches the fp32 CUDA kernel core)
        out_NTD, _, _ = race_prefix_ref(probsK, probsQ, V2.float(), self.eps)  # [N,T,D]
        out = out_NTD.view(self.M, B, H, T, D).mean(dim=0)                     # [B,H,T,D]
        return out
