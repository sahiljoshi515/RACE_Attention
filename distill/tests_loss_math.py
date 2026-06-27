"""Adversarial verification of the loss math in global_utils.py.

We re-derive kl_loss / hidden_loss / ce_loss from scratch (NOT trusting the impl)
and compare on tiny random fp32 + bf16 tensors. We also exercise the combined-loss
assembly that distill_global.py uses (the `(hw*h + kw*kl + cw*ce)/accum` line) and
several edge cases designed to BREAK the implementation:

  * KL chunking equivalence for --kl-chunk in {1, 7, 2048, 0}  (chunked == unchunked
    == independent reference) for T in {1, 2}.
  * KL direction is KL(softmax(teacher/T) || softmax(student/T)) with teacher = the
    *target* (detached) and the T^2 scale + batchmean over B*T rows.
  * Gradient flows to STUDENT only; teacher.grad is None.
  * hidden_loss = mean over replaced layers of per-layer MSE; teacher detached.
  * ce_loss == F.cross_entropy on shifted (next-token) labels.
  * Edge cases: T=1, single row, all-equal logits -> KL == 0, KL(p||p) == 0.

CPU-only, deterministic. Run:  python distill/tests_loss_math.py
Does NOT modify any script under test.
"""
import os
import sys
import math

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from global_utils import kl_loss, hidden_loss, ce_loss  # noqa: E402

torch.manual_seed(1234)
DEV = "cpu"

# accumulate (name, passed, detail) so the run reports every check + max-diffs
RESULTS = []


def record(name, passed, detail=""):
    RESULTS.append((name, bool(passed), detail))
    tag = "PASS" if passed else "FAIL"
    print(f"[{tag}] {name}  {detail}")


# --------------------------------------------------------------------------- #
# independent references (written from scratch; no reuse of the impl)         #
# --------------------------------------------------------------------------- #
def ref_kl(student_logits, teacher_logits, T):
    """T^2 * mean_over_rows( sum_v p_t * (log p_t - log p_s) ), with
    p_t = softmax(teacher/T), p_s = softmax(student/T). Computed in fp64 for a
    high-precision oracle, fully manual (no F.kl_div, no F.log_softmax)."""
    V = student_logits.size(-1)
    s = student_logits.detach().double().reshape(-1, V) / T
    t = teacher_logits.detach().double().reshape(-1, V) / T
    # manual stable log-softmax
    log_ps = s - torch.logsumexp(s, dim=-1, keepdim=True)
    log_pt = t - torch.logsumexp(t, dim=-1, keepdim=True)
    p_t = log_pt.exp()
    per_row = (p_t * (log_pt - log_ps)).sum(dim=-1)   # KL(p_t || p_s) per row
    return (per_row.mean() * (T * T)).item()


def ref_mse(a, b):
    """Plain elementwise mean-squared-error in fp64."""
    a = a.detach().double()
    b = b.detach().double()
    return ((a - b) ** 2).mean().item()


def ref_ce(logits, input_ids):
    """next-token CE = mean over t of -log softmax(logits_t)[x_{t+1}], fp64, manual."""
    V = logits.size(-1)
    lg = logits.detach().double()[:, :-1].reshape(-1, V)
    lbl = input_ids[:, 1:].reshape(-1)
    log_p = lg - torch.logsumexp(lg, dim=-1, keepdim=True)
    nll = -log_p[torch.arange(lbl.numel()), lbl]
    return nll.mean().item()


# --------------------------------------------------------------------------- #
# (a) kl_loss                                                                 #
# --------------------------------------------------------------------------- #
def test_kl():
    print("\n=== (a) kl_loss ===")
    B, Tlen, V = 3, 5, 17          # N = B*Tlen = 15 rows; tiny vocab
    for dtype in (torch.float32, torch.bfloat16):
        s_logits = torch.randn(B, Tlen, V, dtype=dtype, device=DEV)
        t_logits = torch.randn(B, Tlen, V, dtype=dtype, device=DEV)
        N = B * Tlen
        for T in (1.0, 2.0):
            ref = ref_kl(s_logits, t_logits, T)
            unchunked = kl_loss(s_logits, t_logits, T=T, chunk=0).item()
            # the impl upcasts to fp32; a tiny tolerance vs the fp64 oracle.
            tol = 1e-5 if dtype == torch.float32 else 2e-2
            record(f"kl unchunked == fp64 ref  dt={dtype} T={T}",
                   abs(unchunked - ref) < tol,
                   f"impl={unchunked:.8f} ref={ref:.8f} |diff|={abs(unchunked-ref):.2e}")

            # chunked == unchunked, BIT-EXACT expectation in fp32 (same fp32 math, just summed
            # differently -> allow a tiny float-association epsilon). Test the stated values.
            for ck in (1, 7, 2048, 0):
                chunked = kl_loss(s_logits, t_logits, T=T, chunk=ck).item()
                d = abs(chunked - unchunked)
                # chunk>=N or chunk==0 takes the unchunked branch -> must be IDENTICAL.
                exact = (ck == 0 or ck >= N)
                ok = (d == 0.0) if exact else (d < 1e-6)
                record(f"kl chunk={ck} == unchunked  dt={dtype} T={T}",
                       ok, f"chunked={chunked:.8f} |diff|={d:.2e} exact={exact}")

    # KL direction: teacher is the TARGET. Make student near-uniform, teacher peaked.
    # KL(peaked || uniform) >> KL(uniform || peaked); verify impl matches the teacher-target one.
    V = 8
    peaked = torch.zeros(1, V); peaked[0, 0] = 10.0      # teacher: nearly one-hot
    uniform = torch.zeros(1, V)                           # student: exactly uniform
    impl = kl_loss(uniform, peaked, T=1.0, chunk=0).item()      # KL(softmax(teacher)||softmax(student))
    ref_correct = ref_kl(uniform, peaked, 1.0)
    ref_wrong = ref_kl(peaked, uniform, 1.0)             # the reversed direction
    record("kl direction = KL(teacher||student)",
           abs(impl - ref_correct) < 1e-5 and abs(impl - ref_wrong) > 1.0,
           f"impl={impl:.5f} correct={ref_correct:.5f} reversed={ref_wrong:.5f}")

    # T^2 scaling check: at fixed distributions, the (1/T)-divided logits change p's,
    # but verify the explicit T*T multiplier is present by comparing to ref at T=2.
    s2 = torch.randn(2, 6); t2 = torch.randn(2, 6)
    impl_T2 = kl_loss(s2, t2, T=2.0, chunk=0).item()
    # strip the scale: divide by T^2 and compare to a batchmean KL of the /T logits.
    log_s = F.log_softmax(s2 / 2.0, dim=-1)
    p_t = F.softmax(t2 / 2.0, dim=-1)
    manual = (F.kl_div(log_s, p_t, reduction="batchmean") * 4.0).item()
    record("kl explicit T^2 scale", abs(impl_T2 - manual) < 1e-6,
           f"impl={impl_T2:.6f} manual={manual:.6f}")

    # batchmean over B*T rows: doubling the rows (same per-row KL) must NOT change the mean.
    s3 = torch.randn(1, 4, 9); t3 = torch.randn(1, 4, 9)
    one = kl_loss(s3, t3, T=1.0, chunk=0).item()
    s3d = torch.cat([s3, s3], dim=1); t3d = torch.cat([t3, t3], dim=1)
    two = kl_loss(s3d, t3d, T=1.0, chunk=0).item()
    record("kl batchmean (per-row mean, row-count invariant)", abs(one - two) < 1e-6,
           f"4rows={one:.6f} 8rows={two:.6f}")

    # gradient flows to STUDENT only; teacher.grad is None.
    s_g = torch.randn(2, 5, 11, requires_grad=True)
    t_g = torch.randn(2, 5, 11, requires_grad=True)
    L = kl_loss(s_g, t_g, T=1.5, chunk=3)
    L.backward()
    record("kl grad -> student only (teacher.grad None)",
           s_g.grad is not None and s_g.grad.abs().sum() > 0 and t_g.grad is None,
           f"student_grad_sum={s_g.grad.abs().sum():.4f} teacher_grad={t_g.grad}")

    # chunked path also keeps the student graph (grad nonzero through chunking).
    s_gc = torch.randn(4, 4, 7, requires_grad=True)
    t_gc = torch.randn(4, 4, 7)
    kl_loss(s_gc, t_gc, T=1.0, chunk=1).backward()
    record("kl chunked path is differentiable to student",
           s_gc.grad is not None and torch.isfinite(s_gc.grad).all() and s_gc.grad.abs().sum() > 0,
           f"grad_sum={s_gc.grad.abs().sum():.4f}")


# --------------------------------------------------------------------------- #
# (a-edge) edge cases                                                         #
# --------------------------------------------------------------------------- #
def test_kl_edges():
    print("\n=== (a-edge) kl_loss edge cases ===")
    # single row (N==1): all chunk values must agree and match ref.
    s = torch.randn(1, 1, 13); t = torch.randn(1, 1, 13)
    base = kl_loss(s, t, T=1.0, chunk=0).item()
    ref = ref_kl(s, t, 1.0)
    record("kl single-row matches ref", abs(base - ref) < 1e-5,
           f"impl={base:.6f} ref={ref:.6f}")
    for ck in (1, 7, 2048, 0):
        v = kl_loss(s, t, T=1.0, chunk=ck).item()
        record(f"kl single-row chunk={ck} == unchunked", v == base, f"diff={abs(v-base):.2e}")

    # all-equal logits (teacher==student==uniform) -> KL == 0 exactly.
    z = torch.zeros(2, 3, 10)
    record("kl all-equal logits -> 0", kl_loss(z, z, T=1.0, chunk=0).item() == 0.0,
           f"val={kl_loss(z, z, T=1.0, chunk=0).item()}")

    # KL(p || p) == 0 for arbitrary identical logits (any T), including chunked.
    p = torch.randn(3, 4, 12)
    for T in (1.0, 2.0):
        for ck in (0, 1, 5):
            v = kl_loss(p, p, T=T, chunk=ck).item()
            record(f"kl(p||p)==0  T={T} chunk={ck}", abs(v) < 1e-6, f"val={v:.2e}")

    # T=1 sanity: explicit ref already covered above; confirm direct equality to manual F.kl_div.
    s1 = torch.randn(2, 8); t1 = torch.randn(2, 8)
    impl = kl_loss(s1, t1, T=1.0, chunk=0).item()
    man = (F.kl_div(F.log_softmax(s1, -1), F.softmax(t1, -1), reduction="batchmean")).item()
    record("kl T=1 == F.kl_div(batchmean)", abs(impl - man) < 1e-6,
           f"impl={impl:.6f} manual={man:.6f}")


# --------------------------------------------------------------------------- #
# (b) hidden_loss                                                             #
# --------------------------------------------------------------------------- #
def test_hidden():
    print("\n=== (b) hidden_loss ===")
    replaced = [2, 5, 9]
    B, Tlen, H = 2, 4, 16
    for dtype in (torch.float32, torch.bfloat16):
        s_store = {"h_out": {}}
        t_store = {"h_out": {}}
        ref_per = {}
        for i in replaced:
            s = torch.randn(B, Tlen, H, dtype=dtype, requires_grad=(dtype == torch.float32))
            t = torch.randn(B, Tlen, H, dtype=dtype)          # teacher: no grad (detached upstream)
            s_store["h_out"][i] = s
            t_store["h_out"][i] = t
            ref_per[i] = ref_mse(s, t)
        loss, per = hidden_loss(s_store, t_store, replaced)
        ref_mean = sum(ref_per.values()) / len(replaced)
        tol = 1e-5 if dtype == torch.float32 else 1e-2
        record(f"hidden mean == mean of per-layer MSE  dt={dtype}",
               abs(loss.item() - ref_mean) < tol,
               f"impl={loss.item():.8f} ref={ref_mean:.8f} |diff|={abs(loss.item()-ref_mean):.2e}")
        # per-layer MSE values match the independent reference.
        worst = max(abs(per[i]["hidden_mse"] - ref_per[i]) for i in replaced)
        record(f"hidden per-layer MSE == ref  dt={dtype}", worst < tol,
               f"max_per_layer_diff={worst:.2e}")

    # gradient flows to the student hidden states; teacher hidden has no grad.
    s_store = {"h_out": {}}; t_store = {"h_out": {}}
    s_tensors = {}
    for i in replaced:
        s = torch.randn(B, Tlen, H, requires_grad=True)
        t = torch.randn(B, Tlen, H)                  # detached
        s_store["h_out"][i] = s; t_store["h_out"][i] = t
        s_tensors[i] = s
    loss, _ = hidden_loss(s_store, t_store, replaced)
    loss.backward()
    grad_ok = all(s_tensors[i].grad is not None and s_tensors[i].grad.abs().sum() > 0 for i in replaced)
    record("hidden grad -> student tensors", grad_ok,
           f"per-layer grad sums {[round(s_tensors[i].grad.abs().sum().item(),3) for i in replaced]}")

    # single replaced layer: mean over 1 layer == that layer's MSE.
    s1 = {"h_out": {0: torch.randn(1, 3, 8)}}
    t1 = {"h_out": {0: torch.randn(1, 3, 8)}}
    l1, p1 = hidden_loss(s1, t1, [0])
    record("hidden single-layer mean == its MSE", abs(l1.item() - p1[0]["hidden_mse"]) < 1e-6,
           f"mean={l1.item():.6f} per={p1[0]['hidden_mse']:.6f}")

    # identical hidden -> MSE 0.
    same = torch.randn(2, 2, 8)
    l0, _ = hidden_loss({"h_out": {0: same.clone()}}, {"h_out": {0: same.clone()}}, [0])
    record("hidden identical -> 0", l0.item() == 0.0, f"val={l0.item()}")


# --------------------------------------------------------------------------- #
# (c) ce_loss                                                                 #
# --------------------------------------------------------------------------- #
def test_ce():
    print("\n=== (c) ce_loss ===")
    B, Tlen, V = 3, 7, 23
    for dtype in (torch.float32, torch.bfloat16):
        logits = torch.randn(B, Tlen, V, dtype=dtype)
        ids = torch.randint(0, V, (B, Tlen))
        impl = ce_loss(logits, ids).item()
        # reference via torch.nn.functional on shifted labels (the canonical recipe).
        sl = logits[:, :-1].float().reshape(-1, V)
        lbl = ids[:, 1:].reshape(-1)
        torch_ce = F.cross_entropy(sl, lbl).item()
        my = ref_ce(logits, ids)
        tol = 1e-5 if dtype == torch.float32 else 1e-2
        record(f"ce == F.cross_entropy(shifted)  dt={dtype}", abs(impl - torch_ce) < tol,
               f"impl={impl:.8f} torch={torch_ce:.8f} |diff|={abs(impl-torch_ce):.2e}")
        record(f"ce == manual -log p(x_t+1|x<=t)  dt={dtype}", abs(impl - my) < tol,
               f"impl={impl:.8f} manual={my:.8f} |diff|={abs(impl-my):.2e}")

    # explicit next-token semantics: position t predicts token t+1 (not t).
    # craft logits so position t strongly predicts ids[t+1] -> CE ~ 0.
    Vv = 10
    ids = torch.tensor([[1, 4, 7, 2, 9]])
    logits = torch.full((1, 5, Vv), -10.0)
    for t in range(4):                      # only first T-1 positions matter
        logits[0, t, ids[0, t + 1]] = 20.0   # position t predicts NEXT token
    impl = ce_loss(logits, ids).item()
    record("ce uses NEXT token (shift by 1)", impl < 1e-3, f"ce={impl:.6e}")

    # ce differentiable to logits.
    lg = torch.randn(2, 5, 12, requires_grad=True)
    ids2 = torch.randint(0, 12, (2, 5))
    ce_loss(lg, ids2).backward()
    record("ce differentiable", lg.grad is not None and lg.grad.abs().sum() > 0,
           f"grad_sum={lg.grad.abs().sum():.4f}")

    # single-row (B=1) still fine.
    lg1 = torch.randn(1, 4, 6); id1 = torch.randint(0, 6, (1, 4))
    r = ce_loss(lg1, id1).item(); ref = ref_ce(lg1, id1)
    record("ce single-row matches ref", abs(r - ref) < 1e-5, f"impl={r:.6f} ref={ref:.6f}")


# --------------------------------------------------------------------------- #
# (d) combined loss assembly (mirrors distill_global.py line 560-561)         #
# --------------------------------------------------------------------------- #
def test_combined():
    print("\n=== (d) combined loss + weights + /accum ===")
    replaced = [1, 3]
    B, Tlen, H, V = 2, 4, 16, 19
    s_store = {"h_out": {}}; t_store = {"h_out": {}}
    for i in replaced:
        s_store["h_out"][i] = torch.randn(B, Tlen, H, requires_grad=True)
        t_store["h_out"][i] = torch.randn(B, Tlen, H)
    s_logits = torch.randn(B, Tlen, V, requires_grad=True)
    t_logits = torch.randn(B, Tlen, V)
    ids = torch.randint(0, V, (B, Tlen))

    hw, kw, cw, accum = 1.0, 0.5, 0.25, 4
    h_loss, _ = hidden_loss(s_store, t_store, replaced)
    k_loss = kl_loss(s_logits, t_logits, T=2.0, chunk=7)
    c_loss = ce_loss(s_logits, ids)
    # the exact assembly from distill_global.py:560-561
    loss = (hw * h_loss + kw * k_loss + cw * c_loss) / accum

    # independent reconstruction from the refs + impl scalar terms
    ref_total = (hw * h_loss.item() + kw * k_loss.item() + cw * c_loss.item()) / accum
    record("combined = (hw*h + kw*kl + cw*ce)/accum", abs(loss.item() - ref_total) < 1e-6,
           f"loss={loss.item():.8f} recon={ref_total:.8f}")

    # /accum scaling: accum=1 must equal the un-divided sum.
    loss1 = (hw * h_loss + kw * k_loss + cw * c_loss) / 1
    summ = hw * h_loss.item() + kw * k_loss.item() + cw * c_loss.item()
    record("combined accum=1 == raw weighted sum", abs(loss1.item() - summ) < 1e-6,
           f"loss={loss1.item():.8f} sum={summ:.8f}")

    # weights actually weight: zeroing kl_weight drops the KL contribution exactly.
    loss_nokl = (hw * h_loss + 0.0 * k_loss + cw * c_loss) / accum
    record("kl_weight=0 removes KL term",
           abs(loss_nokl.item() - (hw * h_loss.item() + cw * c_loss.item()) / accum) < 1e-6,
           f"val={loss_nokl.item():.8f}")

    # combined grad reaches BOTH hidden states and logits.
    loss.backward()
    g_hidden = all(s_store["h_out"][i].grad is not None for i in replaced)
    g_logit = s_logits.grad is not None and s_logits.grad.abs().sum() > 0
    record("combined grad reaches hidden + logits", g_hidden and g_logit,
           f"hidden_ok={g_hidden} logit_grad_sum={s_logits.grad.abs().sum():.4f}")


def main():
    test_kl()
    test_kl_edges()
    test_hidden()
    test_ce()
    test_combined()

    print("\n================= SUMMARY =================")
    n_pass = sum(1 for _, p, _ in RESULTS if p)
    n_fail = len(RESULTS) - n_pass
    for name, p, detail in RESULTS:
        if not p:
            print(f"  FAILED: {name}  {detail}")
    print(f"{n_pass}/{len(RESULTS)} checks passed; {n_fail} failed.")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
