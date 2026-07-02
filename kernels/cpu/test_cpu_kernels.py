"""Correctness + gradient tests for the CPU RACE kernels.

References are computed in float64 (the kernels accumulate in double internally),
so tolerances reflect float32 output rounding, not float32-cumsum drift.

  race_pref  : race_prefix_mean_flat / _bw   (RACE causal prefix-mean block)
  linear_pref: causal_dot_product / _backward (Katharopoulos causal linear attn)

Run:  python test_cpu_kernels.py     (exit code 0 = all pass, 1 = failure)
"""
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from race_ext import load_race, load_linear

torch.manual_seed(0)
race = load_race(verbose=False)
linear = load_linear(verbose=False)

FAILS = []


def relerr(a, b):
    a = a.double(); b = b.double()
    return (a - b).abs().max().item() / (b.abs().max().item() + 1e-12)


def check(name, err, tol):
    ok = err < tol
    print(f"[{'PASS' if ok else 'FAIL'}] {name:42s} relerr={err:.2e} (tol {tol:.0e})")
    if not ok:
        FAILS.append(name)


# =============================================================================
# 1) race_prefix_mean_flat  (forward):  E_t = (Σ_{τ<=t} w_τ v_τ) / (Σ_{τ<=t} w_τ + eps)
# =============================================================================
def ref_prefix_mean_flat(w, v, eps):
    w = w.double(); v = v.double()
    A = w.cumsum(1)                          # [NS,T]
    B = (w.unsqueeze(-1) * v).cumsum(1)      # [NS,T,D]
    return B / (A.unsqueeze(-1) + eps)


eps = 1e-6
print("\n== race_prefix_mean_flat forward ==")
for (NS, T, D) in [(1, 1, 1), (4, 1, 8), (8, 16, 32), (32, 512, 64),
                   (16, 2048, 64), (2, 4096, 128)]:
    w = torch.rand(NS, T, dtype=torch.float32)
    v = torch.randn(NS, T, D, dtype=torch.float32)
    E = race.race_prefix_mean_flat(w.contiguous(), v.contiguous(), eps)
    check(f"race_fwd NS={NS} T={T} D={D}", relerr(E, ref_prefix_mean_flat(w, v, eps)), 1e-4)


# =============================================================================
# 2) race backward vs float64 autograd of the reference
# =============================================================================
class RaceFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, v, eps):
        w = w.contiguous(); v = v.contiguous()
        E = race.race_prefix_mean_flat(w, v, float(eps))
        ctx.save_for_backward(w, v)
        ctx.eps = float(eps)
        return E

    @staticmethod
    def backward(ctx, g):
        w, v = ctx.saved_tensors
        gW, gV = race.race_prefix_mean_flat_bw(w, v, g.contiguous(), ctx.eps)
        return gW, gV, None


print("\n== race_prefix_mean_flat backward ==")
for (NS, T, D) in [(4, 8, 8), (16, 256, 32), (64, 1024, 64), (8, 2048, 128)]:
    w = torch.rand(NS, T, dtype=torch.float32)
    v = torch.randn(NS, T, D, dtype=torch.float32)
    Wl = torch.randn(NS, T, D)
    w1 = w.double().clone().requires_grad_(True)
    v1 = v.double().clone().requires_grad_(True)
    (ref_prefix_mean_flat(w1, v1, eps) * Wl.double()).sum().backward()
    w2 = w.clone().requires_grad_(True)
    v2 = v.clone().requires_grad_(True)
    (RaceFn.apply(w2, v2, eps) * Wl).sum().backward()
    check(f"race_bw gW NS={NS} T={T} D={D}", relerr(w2.grad, w1.grad), 1e-3)
    check(f"race_bw gV NS={NS} T={T} D={D}", relerr(v2.grad, v1.grad), 1e-3)


# =============================================================================
# 3) linear_pref causal_dot_product (forward): out_t = Q_t · Σ_{s<=t} K_s ⊗ V_s
# =============================================================================
def ref_causal_linear(Q, K, V):
    Q = Q.double(); K = K.double(); V = V.double()
    KV = torch.einsum('nhle,nhlm->nhlem', K, V).cumsum(2)   # [N,H,L,E,M]
    return torch.einsum('nhle,nhlem->nhlm', Q, KV)


print("\n== linear_pref causal_dot_product forward ==")
for (N, H, L, E, M) in [(1, 1, 1, 4, 4), (2, 2, 16, 8, 8),
                        (2, 4, 256, 32, 32), (1, 2, 2048, 64, 64)]:
    Q = torch.randn(N, H, L, E); K = torch.randn(N, H, L, E); V = torch.randn(N, H, L, M)
    out = torch.zeros(N, H, L, M)
    linear.causal_dot_product(Q.contiguous(), K.contiguous(), V.contiguous(), out)
    check(f"linear_fwd N={N} H={H} L={L}", relerr(out, ref_causal_linear(Q, K, V)), 1e-3)


class LinFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V):
        Q = Q.contiguous(); K = K.contiguous(); V = V.contiguous()
        out = torch.zeros(Q.shape[0], Q.shape[1], Q.shape[2], V.shape[3])
        linear.causal_dot_product(Q, K, V, out)
        ctx.save_for_backward(Q, K, V)
        return out

    @staticmethod
    def backward(ctx, g):
        Q, K, V = ctx.saved_tensors
        gQ = torch.zeros_like(Q); gK = torch.zeros_like(K); gV = torch.zeros_like(V)
        linear.causal_dot_backward(Q, K, V, g.contiguous(), gQ, gK, gV)
        return gQ, gK, gV


print("\n== linear_pref causal_dot backward ==")
for (N, H, L, E, M) in [(2, 2, 16, 8, 8), (2, 4, 256, 32, 32), (1, 2, 1024, 64, 64)]:
    Q = torch.randn(N, H, L, E); K = torch.randn(N, H, L, E); V = torch.randn(N, H, L, M)
    Wl = torch.randn(N, H, L, M)
    q1, k1, v1 = (x.double().clone().requires_grad_(True) for x in (Q, K, V))
    (ref_causal_linear(q1, k1, v1) * Wl.double()).sum().backward()
    q2, k2, v2 = (x.clone().requires_grad_(True) for x in (Q, K, V))
    (LinFn.apply(q2, k2, v2) * Wl).sum().backward()
    check(f"linear_bw gQ N={N} L={L}", relerr(q2.grad, q1.grad), 2e-3)
    check(f"linear_bw gK N={N} L={L}", relerr(k2.grad, k1.grad), 2e-3)
    check(f"linear_bw gV N={N} L={L}", relerr(v2.grad, v1.grad), 2e-3)


print("\n== SUMMARY ==")
print(f"torch threads={torch.get_num_threads()}  OMP_NUM_THREADS={os.getenv('OMP_NUM_THREADS')}")
if FAILS:
    print("FAILURES:", FAILS)
    sys.exit(1)
print("ALL CPU KERNEL TESTS PASSED")
sys.exit(0)
