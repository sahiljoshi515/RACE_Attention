"""Validate the CPU RACE composite (RaceCausalCPUFn) against race_common's pure
PyTorch reference race_prefix_ref (the same ground truth the CUDA kernel targets).
Runs on CPU only. Exit 0 = all pass."""
import os
import sys
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from race_common import race_prefix_ref          # noqa: E402
from race_causal_cpu import RaceCausalCPUFn       # noqa: E402

torch.manual_seed(0)
FAILS = []


def relerr(a, b):
    a = a.double(); b = b.double()
    return (a - b).abs().max().item() / (b.abs().max().item() + 1e-12)


def check(name, err, tol):
    ok = err < tol
    print(f"[{'PASS' if ok else 'FAIL'}] {name:38s} relerr={err:.2e} (tol {tol:.0e})")
    if not ok:
        FAILS.append(name)


eps = 1e-6
print("== CPU RACE composite forward vs race_prefix_ref ==")
for (N, T, S, D) in [(2, 8, 6, 8), (4, 64, 24, 32), (12, 256, 24, 64), (12, 2048, 24, 128)]:
    probsK = torch.rand(N, T, S)
    probsQ = torch.rand(N, T, S)
    V2 = torch.randn(N, T, D)
    out = RaceCausalCPUFn.apply(probsK, probsQ, V2, eps)
    ref, _, _ = race_prefix_ref(probsK.double(), probsQ.double(), V2.double(), eps)
    check(f"fwd N={N} T={T} S={S} D={D}", relerr(out, ref), 1e-4)

print("\n== CPU RACE composite backward vs autograd of race_prefix_ref ==")
for (N, T, S, D) in [(2, 8, 6, 8), (4, 64, 24, 32), (6, 256, 24, 64)]:
    probsK = torch.rand(N, T, S)
    probsQ = torch.rand(N, T, S)
    V2 = torch.randn(N, T, D)
    Wl = torch.randn(N, T, D)
    pK1 = probsK.double().clone().requires_grad_(True)
    pQ1 = probsQ.double().clone().requires_grad_(True)
    V1 = V2.double().clone().requires_grad_(True)
    ref, _, _ = race_prefix_ref(pK1, pQ1, V1, eps)
    (ref * Wl.double()).sum().backward()
    pK2 = probsK.clone().requires_grad_(True)
    pQ2 = probsQ.clone().requires_grad_(True)
    V2b = V2.clone().requires_grad_(True)
    out = RaceCausalCPUFn.apply(pK2, pQ2, V2b, eps)
    (out * Wl).sum().backward()
    check(f"bw gK N={N} T={T}", relerr(pK2.grad, pK1.grad), 1e-3)
    check(f"bw gQ N={N} T={T}", relerr(pQ2.grad, pQ1.grad), 1e-3)
    check(f"bw gV2 N={N} T={T}", relerr(V2b.grad, V1.grad), 1e-3)

print("\n== SUMMARY ==")
if FAILS:
    print("FAILURES:", FAILS)
    sys.exit(1)
print("ALL CPU RACE COMPOSITE TESTS PASSED")
sys.exit(0)
