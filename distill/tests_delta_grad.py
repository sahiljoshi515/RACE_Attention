"""Gradient + optimization verification for delta-RACE (distill/delta_race.py).

(a) torch.autograd.gradcheck on delta_race_scan with double precision tiny shapes.
(b) Backprop through DeltaRaceLlamaAttention.forward: all params get finite non-zero
    grads; no NaN/inf under bf16 autocast.
(c) overfit sanity: a single DeltaRaceLlamaAttention layer drives a trivial regression
    loss toward 0 with Adam in <200 steps.

Runs CPU for (a) (double precision) and CPU+CUDA for (b)/(c). Pass --device cuda to
exercise the bf16-autocast path on a GPU (SLURM).
"""
import argparse
import math
import sys

import torch

from delta_race import (
    delta_race_scan,
    delta_race_scan_ref,
    DeltaRaceLlamaAttention,
)


def make_rope(B, T, hdim, device, dtype=torch.float32):
    pos = torch.arange(T, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (10000 ** (torch.arange(0, hdim, 2, device=device).float() / hdim))
    ang = pos[:, None] * inv_freq[None, :]
    emb = torch.cat([ang, ang], dim=-1)
    cos = emb.cos()[None].expand(B, -1, -1).to(dtype)
    sin = emb.sin()[None].expand(B, -1, -1).to(dtype)
    return cos, sin


class _Cfg:
    hidden_size = 32
    num_attention_heads = 4
    num_key_value_heads = 2
    attention_bias = False


# ---------------------------------------------------------------------------
# (a) gradcheck on the scan (double precision, tiny)
# ---------------------------------------------------------------------------
def test_gradcheck(device="cpu"):
    print("=" * 70)
    print("(a) torch.autograd.gradcheck on delta_race_scan (float64, tiny)")
    print("=" * 70)
    torch.manual_seed(0)
    # tiny + non-multiple-of-chunk T so the chunk-carry path is exercised.
    N, T, S, hd = 2, 7, 4, 3
    dt = torch.float64

    # raw (unconstrained) inputs -> map to valid domains inside the closure so
    # gradcheck perturbs free variables (probs are simplex via softmax, gates via sigmoid).
    logitsQ = torch.randn(N, T, S, dtype=dt, device=device, requires_grad=True)
    logitsK = torch.randn(N, T, S, dtype=dt, device=device, requires_grad=True)
    v = torch.randn(N, T, hd, dtype=dt, device=device, requires_grad=True)
    beta_raw = torch.randn(N, T, dtype=dt, device=device, requires_grad=True)
    alpha_raw = torch.randn(N, T, dtype=dt, device=device, requires_grad=True)

    def f(lQ, lK, vv, br, ar, chunk):
        pq = torch.softmax(lQ, dim=-1)
        pk = torch.softmax(lK, dim=-1)
        b = torch.sigmoid(br)
        a = torch.sigmoid(ar)
        return delta_race_scan(pq, pk, vv, b, a, chunk=chunk)

    results = {}
    for chunk in (3, 128):  # chunk<T (multi-chunk carry) and chunk>T (single chunk)
        ok = torch.autograd.gradcheck(
            lambda lQ, lK, vv, br, ar: f(lQ, lK, vv, br, ar, chunk),
            (logitsQ, logitsK, v, beta_raw, alpha_raw),
            eps=1e-6, atol=1e-5, rtol=1e-4, nondet_tol=0.0,
        )
        results[chunk] = ok
        print(f"  gradcheck(chunk={chunk}) -> {'PASS' if ok else 'FAIL'}")

    # second-order: gradgradcheck (training may use grad penalties / higher order)
    try:
        gg = torch.autograd.gradgradcheck(
            lambda lQ, lK, vv, br, ar: f(lQ, lK, vv, br, ar, 3),
            (logitsQ, logitsK, v, beta_raw, alpha_raw),
            atol=1e-4, rtol=1e-3,
        )
        print(f"  gradgradcheck(chunk=3) -> {'PASS' if gg else 'FAIL'}")
    except Exception as e:
        gg = False
        print(f"  gradgradcheck error: {e}")

    # also: ref-vs-vec gradient agreement on the *actual* used inputs (sanity)
    pq = torch.softmax(logitsQ.detach().clone().requires_grad_(True), dim=-1)
    pass_all = all(results.values()) and gg
    print(f"  -> gradcheck all chunks PASS = {all(results.values())}")
    return pass_all, results, gg


# ---------------------------------------------------------------------------
# ref-vs-vec gradient agreement (extra correctness on the backward path)
# ---------------------------------------------------------------------------
def test_grad_ref_vs_vec(device="cpu"):
    print("=" * 70)
    print("    ref-vs-vec gradient agreement (float64)")
    print("=" * 70)
    torch.manual_seed(1)
    N, T, S, hd = 2, 37, 8, 16
    dt = torch.float64

    def build():
        pq = torch.softmax(torch.randn(N, T, S, dtype=dt), dim=-1)
        pk = torch.softmax(torch.randn(N, T, S, dtype=dt), dim=-1)
        vv = torch.randn(N, T, hd, dtype=dt)
        b = torch.sigmoid(torch.randn(N, T, dtype=dt))
        a = torch.sigmoid(torch.randn(N, T, dtype=dt))
        return pq, pk, vv, b, a

    torch.manual_seed(7)
    pq, pk, vv, b, a = build()
    grads = {}
    for name, fn in (("ref", delta_race_scan_ref), ("vec", delta_race_scan)):
        ins = [t.clone().requires_grad_(True) for t in (pq, pk, vv, b, a)]
        out = fn(*ins)
        # arbitrary scalar loss with structure
        loss = (out * torch.sin(out)).sum()
        loss.backward()
        grads[name] = [i.grad.clone() for i in ins]

    maxd = max((gr - gv).abs().max().item() for gr, gv in zip(grads["ref"], grads["vec"]))
    ok = maxd < 1e-8
    print(f"  max|grad_ref - grad_vec| = {maxd:.3e} -> {'PASS' if ok else 'FAIL'}")
    return ok, maxd


# ---------------------------------------------------------------------------
# (b) backprop through the module: all params finite non-zero grads, bf16 safe
# ---------------------------------------------------------------------------
def test_module_backprop(device="cpu", use_autocast=False):
    print("=" * 70)
    print(f"(b) backprop through DeltaRaceLlamaAttention.forward "
          f"(device={device}, autocast_bf16={use_autocast})")
    print("=" * 70)
    torch.manual_seed(0)
    cfg = _Cfg()
    B, T = 2, 24
    hdim = cfg.hidden_size // cfg.num_attention_heads
    attn = DeltaRaceLlamaAttention(cfg, layer_idx=0, L=2, Kbits=2,
                                   device=device, learn_alpha=True).to(device)
    attn.make_hash_trainable()  # so planes_T/protos_T also receive grads
    hs = torch.randn(B, T, cfg.hidden_size, device=device, requires_grad=True)
    cos, sin = make_rope(B, T, hdim, device)

    autocast_dtype = torch.bfloat16
    ctx = torch.autocast(device_type=device, dtype=autocast_dtype) if use_autocast else _nullctx()
    with ctx:
        out, _ = attn(hs, position_embeddings=(cos, sin))
    target = torch.randn_like(out.float())
    loss = ((out.float() - target) ** 2).mean()

    loss_finite = torch.isfinite(loss).item()
    out_finite = torch.isfinite(out.float()).all().item()
    print(f"  loss={loss.item():.4f} finite={loss_finite}  out_finite={out_finite}  out_dtype={out.dtype}")

    loss.backward()

    expect = ["q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight",
              "W_beta.weight", "W_beta.bias", "W_alpha.weight", "W_alpha.bias",
              "log_temp", "planes_T", "protos_T"]
    rows = []
    all_ok = True
    for n, p in attn.named_parameters():
        if p.grad is None:
            finite = False
            nrm = float("nan")
            nonzero = False
        else:
            g = p.grad
            finite = torch.isfinite(g).all().item()
            nrm = g.norm().item()
            nonzero = (g.abs().sum().item() > 0)
        ok = finite and nonzero
        if any(n.endswith(e) or n == e for e in expect):
            all_ok = all_ok and ok
        rows.append((n, nrm, finite, nonzero, ok))

    for n, nrm, finite, nonzero, ok in rows:
        print(f"    {n:24s} |grad|={nrm:.3e}  finite={finite} nonzero={nonzero} {'OK' if ok else 'BAD'}")
    print(f"  hs.grad finite={torch.isfinite(hs.grad).all().item()}")
    print(f"  -> all expected params finite & non-zero = {all_ok}")
    return all_ok and loss_finite and out_finite, rows


class _nullctx:
    def __enter__(self): return None
    def __exit__(self, *a): return False


# ---------------------------------------------------------------------------
# (c) overfit sanity: drive a trivial regression loss toward 0 with Adam <200 steps
# ---------------------------------------------------------------------------
def test_overfit(device="cpu", use_autocast=False, steps=200):
    print("=" * 70)
    print(f"(c) overfit: single DeltaRaceLlamaAttention -> loss->0 in <{steps} steps "
          f"(device={device}, autocast_bf16={use_autocast})")
    print("=" * 70)
    torch.manual_seed(0)
    cfg = _Cfg()
    B, T = 2, 16
    hdim = cfg.hidden_size // cfg.num_attention_heads
    attn = DeltaRaceLlamaAttention(cfg, layer_idx=0, L=2, Kbits=2,
                                   device=device, learn_alpha=True).to(device)
    attn.make_hash_trainable()

    # fixed input + fixed random target -> a single (input,target) pair to memorize.
    hs = torch.randn(B, T, cfg.hidden_size, device=device)
    cos, sin = make_rope(B, T, hdim, device)
    target = torch.randn(B, T, cfg.hidden_size, device=device)

    opt = torch.optim.Adam(attn.parameters(), lr=3e-3)
    losses = []
    autocast_dtype = torch.bfloat16
    for step in range(steps):
        opt.zero_grad()
        ctx = torch.autocast(device_type=device, dtype=autocast_dtype) if use_autocast else _nullctx()
        with ctx:
            out, _ = attn(hs, position_embeddings=(cos, sin))
        loss = ((out.float() - target) ** 2).mean()
        loss.backward()
        opt.step()
        losses.append(loss.item())

    l0, lf = losses[0], losses[-1]
    lmin = min(losses)
    ratio = lf / (l0 + 1e-12)
    ok = (lf < 0.05 * l0) or (lf < 1e-3)
    print(f"  loss[0]={l0:.4f}  loss[-1]={lf:.6f}  min={lmin:.6f}  final/initial={ratio:.4f}")
    # sparse loss curve
    idxs = list(range(0, steps, max(1, steps // 20)))
    if (steps - 1) not in idxs:
        idxs.append(steps - 1)
    print("  loss curve:")
    for i in idxs:
        print(f"    step {i:4d}: {losses[i]:.6f}")
    print(f"  -> overfit toward 0 = {ok}")
    return ok, losses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--steps", type=int, default=200)
    args = ap.parse_args()
    dev = args.device

    print(f"torch {torch.__version__}  device={dev}  cuda_avail={torch.cuda.is_available()}")

    results = {}
    # (a) gradcheck always on CPU float64 (deterministic, exact)
    ga, gres, gg = test_gradcheck("cpu")
    results["gradcheck"] = ga
    gv_ok, gv_d = test_grad_ref_vs_vec("cpu")
    results["grad_ref_vs_vec"] = gv_ok

    # (b) backprop fp32 + bf16-autocast (autocast only meaningful/clean on cuda)
    b_fp32, _ = test_module_backprop(dev, use_autocast=False)
    results["backprop_fp32"] = b_fp32
    b_bf16, _ = test_module_backprop(dev, use_autocast=True)
    results["backprop_bf16_autocast"] = b_bf16

    # (c) overfit fp32 (+ bf16 if cuda)
    c_fp32, _ = test_overfit(dev, use_autocast=False, steps=args.steps)
    results["overfit_fp32"] = c_fp32
    if dev == "cuda":
        c_bf16, _ = test_overfit(dev, use_autocast=True, steps=args.steps)
        results["overfit_bf16_autocast"] = c_bf16

    print("\n" + "=" * 70)
    print("SUMMARY")
    for k, vok in results.items():
        print(f"  {k:30s} {'PASS' if vok else 'FAIL'}")
    all_ok = all(results.values())
    print(f"ALL: {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
