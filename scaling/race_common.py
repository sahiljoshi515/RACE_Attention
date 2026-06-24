"""Shared pieces for the causal-RACE benchmark / correctness harnesses.

Everything here mirrors the soft-hash + causal-prefix math in
``misc/race.py::BatchedACE`` (the canonical PyTorch cumsum implementation), so
the CUDA kernel path and the cumsum path are fed *identical* probabilities and
differ only in how the causal scan is executed.
"""
import math
import itertools
import torch
import torch.nn.functional as F


def build_planes_protos(d_k, Kbits, L, M, device="cuda", share_planes=True,
                        dtype=torch.float32, seed=0):
    """Random hyperplanes + bucket prototypes, matching BatchedACE.__init__.

    share_planes=True  -> planes_T [d_k, L*Kbits]   (one shared sketch)
    share_planes=False -> planes_T [M, d_k, L*Kbits] (independent per ensemble)
    protos_T: [Kbits, R] with R = 2**Kbits, corners of {-1,+1}^Kbits.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    if share_planes:
        planes = torch.randn(L, Kbits, d_k, generator=g)
        planes_T = planes.view(L * Kbits, d_k).T.contiguous()              # [d_k, L*Kbits]
    else:
        planes = torch.randn(M, L, Kbits, d_k, generator=g)
        planes_T = planes.view(M, L * Kbits, d_k).transpose(1, 2).contiguous()  # [M, d_k, L*Kbits]
    planes_T = planes_T.to(device=device, dtype=dtype)
    corners = torch.tensor(list(itertools.product([-1.0, 1.0], repeat=Kbits)),
                           dtype=dtype, device=device)
    protos_T = corners.T.contiguous()                                      # [Kbits, R]
    return planes_T, protos_T


def soft_hash_probs(Q, K, V, planes_T, protos_T, L, Kbits, M, share_planes=True):
    """Project Q,K -> soft bucket probabilities (causal RACE soft-hash).

    Q,K,V: [B,H,T,D].  Returns
      probsK, probsQ : [N,T,S]  (fp32, S = L * 2**Kbits, L-outer / R-inner)
      V2             : [N,T,D]  (same dtype as V)
      N              : M*B*H  (ensemble-major: ordering (M,B,H))
    """
    B, H, T, D = Q.shape
    R = protos_T.shape[1]
    scale = math.sqrt(D)
    N = M * B * H

    def packM(Z):
        # [B,H,T,D] -> [M,B,H,T,D] -> [N,T,D] with N ordered (M,B,H)
        return Z.unsqueeze(0).expand(M, -1, -1, -1, -1).reshape(N, T, D)

    Km, Qm, Vm = packM(K), packM(Q), packM(V)
    pT = planes_T.to(Qm.dtype)
    if share_planes:
        projK = Km @ pT                                   # [N,T,L*Kbits]
        projQ = Qm @ pT
    else:
        LK = pT.shape[-1]
        # [M,D,LK] -> [N,D,LK]  (N = (M,B,H))
        pTe = pT.unsqueeze(1).unsqueeze(1).expand(M, B, H, D, LK).reshape(N, D, LK)
        projK = torch.bmm(Km, pTe)
        projQ = torch.bmm(Qm, pTe)

    prT = protos_T.to(projK.dtype)
    projK = projK.view(N, T, L, Kbits)
    projQ = projQ.view(N, T, L, Kbits)
    logitsK = (projK.tanh() / scale) @ prT                # [N,T,L,R]
    logitsQ = (projQ.tanh() / scale) @ prT
    # softmax in fp32 (matches the kernel's fp32 probs requirement)
    probsK = F.softmax(logitsK.float(), dim=-1).reshape(N, T, L * R)
    probsQ = F.softmax(logitsQ.float(), dim=-1).reshape(N, T, L * R)
    return probsK, probsQ, Vm, N


def race_prefix_ref(probsK, probsQ, V2, eps=1e-6):
    """Pure-PyTorch causal RACE prefix scan (the cumsum reference).

    probsK, probsQ: [N,T,S];  V2: [N,T,D].  Returns (out[N,T,D], A_final[N,S],
    B_final[N,S,D]).  This is exactly the kernel's math with the L/R bucket axes
    already collapsed into S, so it doubles as the correctness ground truth.
    Materializes B_pref [N,T,S,D] -> OOMs at long T (intended).
    """
    A_pref = probsK.cumsum(dim=1)                                       # [N,T,S]
    B_pref = (probsK.unsqueeze(-1) * V2.unsqueeze(2)).cumsum(dim=1)     # [N,T,S,D]
    E_pref = B_pref / (A_pref.unsqueeze(-1) + eps)                      # [N,T,S,D]
    out = torch.einsum("nts,ntsd->ntd", probsQ, E_pref)                # [N,T,D]
    A_final = A_pref[:, -1].contiguous()                               # [N,S]
    B_final = B_pref[:, -1].contiguous()                               # [N,S,D]
    return out, A_final, B_final
