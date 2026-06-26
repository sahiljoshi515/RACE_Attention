"""INDEPENDENT verifier for delta_race.py.

I write my OWN sequential reference of the gated-delta recurrence straight from the
spec text (not copied from delta_race.py), then compare it to BOTH delta_race_scan
and delta_race_scan_ref across many random shapes / dtypes, and check strict
causality via autograd Jacobian zeros and edge cases.
"""
import sys, os
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from delta_race import delta_race_scan, delta_race_scan_ref


# ---- MY OWN reference, written directly from the spec ----------------------
# Spec:
#   M_0[s] = 0
#   M_t[s] = (alpha_t - beta_t * p_{t,s}) * M_{t-1}[s] + (beta_t * p_{t,s}) * v_t
#   out_t  = sum_s probsQ[t,s] * M_t[s]
def my_ref(probsQ, probsK, v, beta, alpha):
    N, T, S = probsK.shape
    hd = v.shape[-1]
    M = torch.zeros(N, S, hd, dtype=v.dtype, device=v.device)
    outs = []
    for t in range(T):
        p = probsK[:, t]                       # [N,S]
        bp = beta[:, t, None] * p              # [N,S]
        g = alpha[:, t, None] - bp             # [N,S]
        # update each bucket independently
        M = g[..., None] * M + bp[..., None] * v[:, t, None, :]   # [N,S,hd]
        o = (probsQ[:, t, :, None] * M).sum(dim=1)                # [N,hd]
        outs.append(o)
    return torch.stack(outs, dim=1)            # [N,T,hd]


def rand_probs(N, T, S, gen):
    return torch.softmax(torch.randn(N, T, S, generator=gen, dtype=torch.float64).float(), dim=-1)


def maxdiff(a, b):
    return (a.float() - b.float()).abs().max().item()


def main():
    gen = torch.Generator().manual_seed(1234)
    print("=== (a) my_ref vs delta_race_scan AND delta_race_scan_ref ===")
    worst_vec = 0.0
    worst_refimpl = 0.0
    shapes = []
    for T in (1, 7, 8, 9, 37, 128, 130, 200):
        for S in (8, 24):
            for hd in (16, 128):
                shapes.append((2, T, S, hd))
    # add an odd N
    shapes.append((3, 53, 24, 16))

    fail = []
    for (N, T, S, hd) in shapes:
        probsK = rand_probs(N, T, S, gen)
        probsQ = rand_probs(N, T, S, gen)
        v = torch.randn(N, T, hd, generator=gen)
        beta = torch.sigmoid(torch.randn(N, T, generator=gen))
        alpha = torch.sigmoid(torch.randn(N, T, generator=gen))

        ref_me = my_ref(probsQ, probsK, v, beta, alpha)
        ref_impl = delta_race_scan_ref(probsQ, probsK, v, beta, alpha)
        # try several chunks incl. non-multiples of T
        for ch in (1, 8, 16, 128, 512):
            vec = delta_race_scan(probsQ, probsK, v, beta, alpha, chunk=ch)
            d = maxdiff(ref_me, vec)
            worst_vec = max(worst_vec, d)
            if d >= 1e-4:
                fail.append((N, T, S, hd, ch, d))
        d2 = maxdiff(ref_me, ref_impl)
        worst_refimpl = max(worst_refimpl, d2)
        if d2 >= 1e-4:
            fail.append((N, T, S, hd, "refimpl", d2))
    print(f"  fp32 worst |my_ref - delta_race_scan|      = {worst_vec:.3e}")
    print(f"  fp32 worst |my_ref - delta_race_scan_ref|  = {worst_refimpl:.3e}")
    print(f"  fp32 failures (>=1e-4): {fail}")

    # ---- bf16 autocast ----
    print("=== (a') bf16 autocast (CPU) my_ref vs both impls ===")
    worst_bf = 0.0
    for (N, T, S, hd) in [(2, 37, 8, 16), (2, 130, 24, 128), (2, 1, 8, 16)]:
        probsK = rand_probs(N, T, S, gen)
        probsQ = rand_probs(N, T, S, gen)
        v = torch.randn(N, T, hd, generator=gen)
        beta = torch.sigmoid(torch.randn(N, T, generator=gen))
        alpha = torch.sigmoid(torch.randn(N, T, generator=gen))
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            ref_impl = delta_race_scan_ref(probsQ, probsK, v, beta, alpha)
            vec = delta_race_scan(probsQ, probsK, v, beta, alpha, chunk=64)
        # compare the two impls to each other under autocast and check finite
        d = maxdiff(ref_impl, vec)
        worst_bf = max(worst_bf, d)
        finite = torch.isfinite(vec.float()).all().item() and torch.isfinite(ref_impl.float()).all().item()
        print(f"  shape={(N,T,S,hd)} |ref_impl-vec|={d:.3e} finite={finite}")
    print(f"  bf16 worst |ref_impl - vec| = {worst_bf:.3e}")

    # ---- (b) strict causality via autograd: d out_i / d v_j must be 0 for j>i ----
    print("=== (b) strict causality: autograd d out_i / d v_j == 0 for j>i ===")
    N, T, S, hd = 2, 12, 8, 4
    probsK = rand_probs(N, T, S, gen)
    probsQ = rand_probs(N, T, S, gen)
    beta = torch.sigmoid(torch.randn(N, T, generator=gen))
    alpha = torch.sigmoid(torch.randn(N, T, generator=gen))

    def causality_jacobian(scan_fn, label):
        v = torch.randn(N, T, hd, generator=gen, requires_grad=True)
        out = scan_fn(probsQ, probsK, v, beta, alpha)   # [N,T,hd]
        viol = 0.0
        for i in range(T):
            # scalar = out at time i (sum over N,hd); grad wrt v should be 0 for j>i
            g, = torch.autograd.grad(out[:, i].sum(), v, retain_graph=True)
            future = g[:, i + 1:]
            viol = max(viol, future.abs().max().item() if future.numel() else 0.0)
        print(f"  {label}: max |d out_i/d v_{{j>i}}| = {viol:.3e}")
        return viol
    cviol_vec = causality_jacobian(lambda *a: delta_race_scan(*a, chunk=5), "delta_race_scan(chunk=5)")
    cviol_ref = causality_jacobian(delta_race_scan_ref, "delta_race_scan_ref")

    # ---- (c) edge cases ----
    print("=== (c) edge cases ===")
    # T=1
    probsK = rand_probs(2, 1, 8, gen); probsQ = rand_probs(2, 1, 8, gen)
    v = torch.randn(2, 1, 16, generator=gen)
    beta = torch.sigmoid(torch.randn(2, 1, generator=gen)); alpha = torch.sigmoid(torch.randn(2, 1, generator=gen))
    d_t1 = maxdiff(my_ref(probsQ, probsK, v, beta, alpha), delta_race_scan(probsQ, probsK, v, beta, alpha))
    print(f"  T=1: |my_ref-vec|={d_t1:.3e}")

    # beta=0 => M stays 0 (g=alpha, u=0, M_0=0) => out=0
    N, T, S, hd = 2, 20, 8, 16
    probsK = rand_probs(N, T, S, gen); probsQ = rand_probs(N, T, S, gen)
    v = torch.randn(N, T, hd, generator=gen)
    beta0 = torch.zeros(N, T); alpha = torch.sigmoid(torch.randn(N, T, generator=gen))
    out0 = delta_race_scan(probsQ, probsK, v, beta0, alpha)
    out0r = delta_race_scan_ref(probsQ, probsK, v, beta0, alpha)
    print(f"  beta=0: max|out_vec|={out0.abs().max().item():.3e} max|out_ref|={out0r.abs().max().item():.3e} (expect 0)")

    # alpha=1 => pure delta. Verify against my_ref with alpha=1 explicitly.
    alpha1 = torch.ones(N, T)
    beta = torch.sigmoid(torch.randn(N, T, generator=gen))
    d_a1 = maxdiff(my_ref(probsQ, probsK, v, beta, alpha1),
                   delta_race_scan(probsQ, probsK, v, beta, alpha1))
    # also independently: pure-delta recurrence M = M + beta*p*(v - M)
    Mpd = torch.zeros(N, S, hd)
    outs = []
    for t in range(T):
        p = probsK[:, t]; bp = beta[:, t, None] * p
        Mpd = Mpd + bp[..., None] * (v[:, t, None, :] - Mpd)
        outs.append((probsQ[:, t, :, None] * Mpd).sum(1))
    puredelta = torch.stack(outs, 1)
    d_pd = maxdiff(puredelta, delta_race_scan(probsQ, probsK, v, beta, alpha1))
    print(f"  alpha=1: |my_ref(a=1)-vec|={d_a1:.3e}  |independent pure-delta - vec|={d_pd:.3e}")

    # ---- adversarial: gate passes through ~0 (alpha ~ beta*p) ----
    print("=== (d) adversarial near-zero/negative gate stability ===")
    N, T, S, hd = 2, 200, 8, 16
    probsK = rand_probs(N, T, S, gen); probsQ = rand_probs(N, T, S, gen)
    v = torch.randn(N, T, hd, generator=gen)
    beta = torch.full((N, T), 0.9)
    alpha = probsK.max(dim=-1).values * 0.9   # alpha ~ beta * max_p so g near 0 / negative
    refm = my_ref(probsQ, probsK, v, beta, alpha)
    worst_adv = 0.0
    for ch in (1, 16, 128, 512):
        vec = delta_race_scan(probsQ, probsK, v, beta, alpha, chunk=ch)
        worst_adv = max(worst_adv, maxdiff(refm, vec))
    print(f"  adversarial worst |my_ref-vec|={worst_adv:.3e} finite={torch.isfinite(refm).all().item()}")

    print("\n=== SUMMARY ===")
    print(f"fp32 worst vec diff={worst_vec:.3e}, refimpl diff={worst_refimpl:.3e}, "
          f"bf16={worst_bf:.3e}, adversarial={worst_adv:.3e}")
    print(f"causality vec={cviol_vec:.3e} ref={cviol_ref:.3e}")
    print(f"T=1={d_t1:.3e} beta0_vec={out0.abs().max().item():.3e} "
          f"alpha1={d_a1:.3e} puredelta={d_pd:.3e}")


if __name__ == "__main__":
    main()
