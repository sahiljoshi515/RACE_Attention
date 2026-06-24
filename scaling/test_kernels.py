"""Correctness tests: the causal RACE CUDA kernels vs the fp64 cumsum reference.

The PyTorch cumsum implementation (race_common.race_prefix_ref, == misc/race.py)
is the EXACT specification; the CUDA kernels must match it. We compare, in fp64
ground truth:
  * forward out (race_fused_fwd) across several chunk sizes,
  * backward gradProbsK / gradProbsQ / gradV (RaceCausalFn),
  * end-to-end module gradients (RaceCausalCuda vs the cumsum module),
across many shapes, edge cases (T=1, tiny T, odd D), S in {8,24,64}, determinism.

Run: python scaling/test_kernels.py   (needs a CUDA GPU + nvcc to build the ext).
Exit 0 iff all checks pass.
"""
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from race_common import race_prefix_ref                                  # noqa: E402
from race_causal_cuda import RaceCausalFn, RaceCausalCuda, _ext           # noqa: E402
from race_torch_cumsum import RaceCumsumCausal                           # noqa: E402

DEV = "cuda"
N_FAIL = 0
N_PASS = 0
LINES = []
PARAM_SETS = [(2, 2), (3, 3), (4, 4)]            # (L, Kbits) -> S = 8, 24, 64


def relerr(a, b):
    return (a.double() - b.double()).abs().max().item() / (b.double().abs().max().item() + 1e-30)


def maxabs(a, b):
    return (a.double() - b.double()).abs().max().item()


def check(name, got, ref, tol, abs_floor=2e-5):
    # Pass on relative error, OR on a TIGHT absolute floor (only for genuinely
    # near-zero references, e.g. the intrinsic T=1 gradProbsK cancellation).
    # abs_floor is ~100x below typical gradient magnitudes so a grossly-wrong
    # (e.g. all-zero) gradient cannot pass via the fallback.
    global N_FAIL, N_PASS
    re = relerr(got, ref)
    ok = re <= tol or maxabs(got, ref) <= abs_floor
    LINES.append(f"  [{'PASS' if ok else 'FAIL'}] {name:36s} relerr={re:.2e} (tol {tol:.0e})")
    if ok:
        N_PASS += 1
    else:
        N_FAIL += 1
    return ok


def make_inputs(N, T, L, R, D, seed, dist="softmax"):
    g = torch.Generator().manual_seed(seed)
    if dist == "softmax":
        lk = torch.randn(N, T, L, R, generator=g, dtype=torch.float64)
        lq = torch.randn(N, T, L, R, generator=g, dtype=torch.float64)
        pk = torch.softmax(lk, dim=-1).reshape(N, T, L * R)
        pq = torch.softmax(lq, dim=-1).reshape(N, T, L * R)
    else:  # un-normalized positive (stress)
        pk = torch.rand(N, T, L * R, generator=g, dtype=torch.float64)
        pq = torch.rand(N, T, L * R, generator=g, dtype=torch.float64)
    V = torch.randn(N, T, D, generator=g, dtype=torch.float64)
    go = torch.randn(N, T, D, generator=g, dtype=torch.float64)
    return pk, pq, V, go


def run_case(N, T, L, Kbits, D, seed=0, eps=1e-6, dist="softmax", fwd_tol=2e-5, grad_tol=3e-3):
    R = 1 << Kbits
    S = L * R
    LINES.append(f"--- N={N} T={T} S={S}(L={L},K={Kbits}) D={D} dist={dist} ---")
    pk, pq, V, go = make_inputs(N, T, L, R, D, seed, dist)

    # fp64 ground truth (forward + autograd backward)
    pk64 = pk.detach().clone().requires_grad_(True)
    pq64 = pq.detach().clone().requires_grad_(True)
    v64 = V.detach().clone().requires_grad_(True)
    out_t, _, _ = race_prefix_ref(pk64, pq64, v64, eps)
    (out_t * go).sum().backward()
    gpk_t, gpq_t, gv_t = pk64.grad, pq64.grad, v64.grad

    ext = _ext()
    pkc, pqc, vc = pk.float().cuda().contiguous(), pq.float().cuda().contiguous(), V.float().cuda().contiguous()

    # forward across several chunk sizes
    for C in (1024, 4096, 8192):
        out_k = ext.race_fused_fwd(pkc, pqc, vc, eps, C)
        check(f"fwd out C={C}", out_k, out_t.cuda(), fwd_tol)
    # determinism
    a = ext.race_fused_fwd(pkc, pqc, vc, eps, 4096)
    b = ext.race_fused_fwd(pkc, pqc, vc, eps, 4096)
    check("fwd determinism", a, b, 1e-6)

    # backward via the autograd Function
    pkx = pk.float().cuda().detach().clone().requires_grad_(True)
    pqx = pq.float().cuda().detach().clone().requires_grad_(True)
    vx = V.float().cuda().detach().clone().requires_grad_(True)
    out = RaceCausalFn.apply(pkx, pqx, vx, eps)
    (out * go.float().cuda()).sum().backward()
    check("gradProbsK", pkx.grad, gpk_t.cuda(), grad_tol)
    check("gradProbsQ", pqx.grad, gpq_t.cuda(), grad_tol)
    check("gradV", vx.grad, gv_t.cuda(), grad_tol)
    torch.cuda.empty_cache()


def run_module_e2e(H, T, L, Kbits, D, B=1, seed=0, tol=2e-3):
    """End-to-end: RaceCausalCuda vs the cumsum module through the full soft-hash
    projection + ensemble mean (same seed => identical planes/probs)."""
    S = L * (1 << Kbits)
    LINES.append(f"--- E2E module H={H} T={T} S={S} D={D} ---")
    torch.manual_seed(seed)
    Q = torch.randn(B, H, T, D, device=DEV, dtype=torch.float32)
    K = torch.randn(B, H, T, D, device=DEV, dtype=torch.float32)
    V = torch.randn(B, H, T, D, device=DEV, dtype=torch.float32)
    mc = RaceCausalCuda(D, Kbits, L, 1, device=DEV, seed=0)
    mm = RaceCumsumCausal(D, Kbits, L, 1, device=DEV, seed=0)
    qc, kc, vc = (x.clone().requires_grad_(True) for x in (Q, K, V))
    qm, km, vm = (x.clone().requires_grad_(True) for x in (Q, K, V))
    oc = mc(qc, kc, vc)
    om = mm(qm, km, vm)
    go = torch.randn_like(om)
    (oc * go).sum().backward()
    (om * go).sum().backward()
    check("e2e out", oc, om, 2e-5)
    check("e2e gradQ", qc.grad, qm.grad, tol)
    check("e2e gradK", kc.grad, km.grad, tol)
    check("e2e gradV", vc.grad, vm.grad, tol)
    torch.cuda.empty_cache()


def main():
    print("torch", torch.__version__, "GPU", torch.cuda.get_device_name(0))

    # 1) shape sweep across S, N, D
    for (L, K) in PARAM_SETS:
        for N in (1, 2, 4):
            for D in (16, 64, 128):
                run_case(N, 1024, L, K, D, seed=N + D)

    # 2) T sweep (incl edge T=1,2,7 and large T where the old kernel was catastrophic)
    for (L, K) in PARAM_SETS:
        for T in (1, 2, 7, 64, 256, 4096, 16384, 65536):
            run_case(2, T, L, K, 64, seed=T, grad_tol=5e-3)

    # 3) odd / non-power-of-two and large D
    for (L, K) in PARAM_SETS:
        for D in (96, 130, 256):
            run_case(2, 512, L, K, D, seed=D)

    # 4) un-normalized positive distribution (stress)
    for (L, K) in PARAM_SETS:
        run_case(2, 1024, L, K, 64, seed=7, dist="raw")

    # 5) end-to-end module gradients (projection + scan + ensemble mean)
    for (L, K) in PARAM_SETS:
        for (H, T, D) in ((4, 256, 64), (8, 1024, 128), (4, 2048, 96)):
            run_module_e2e(H, T, L, K, D, seed=H + T)

    print("\n".join(LINES))
    print(f"\n==== {N_PASS} passed, {N_FAIL} failed ====")
    sys.exit(1 if N_FAIL else 0)


if __name__ == "__main__":
    main()
