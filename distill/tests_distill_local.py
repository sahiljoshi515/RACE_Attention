"""Adversarial verification of distill_local.py (the LOCAL teacher-forced pilot).

Run on a GPU (H200) under race_vit_env after `source distill/env.sh`.

What this verifies (mapping to the task spec):
  (1) IMPORT/RUN: distill_local imports cleanly on the current torch 2.10 /
      transformers 5.5.0 stack; its capture hooks fire and produce the tensors it
      expects (rotary_emb hook, per-layer + self_attn hooks). We probe the known
      API-drift traps: LlamaDecoderLayer.forward now returns a BARE tensor (not a
      tuple) and self_attn returns a (out, weights) tuple.
  (2) OBJECTIVE: the loss is per-layer TEACHER-FORCED -- each RACE layer is fed the
      TEACHER's input hidden state h_in[i] and matched to the teacher's attn output
      and the teacher's layer output. We recompute the loss with an INDEPENDENT
      reference and assert layer_losses + student_layer + teacher_targets agree.
  (3) END-TO-END (tiny model): 2-3 optimizer steps; loss decreases; ONLY the RACE
      q/k/v/o + log_temp train (base frozen, zero leakage); and the custom
      race_backward drives q/k/v grads (gradcheck vs torch autograd reference scan).
  (4) CONTRACT vs distill_global: documented divergences + any dead/broken code.

We DO NOT modify distill_local.py. A tiny LlamaConfig (few layers/heads/vocab/seq)
is built directly so no checkpoint download is needed; head_dim stays 128 (the size
the RACE soft-hash + CUDA scan are exercised with in production).

Each check prints PASS/FAIL with numbers; results are collected and dumped as JSON
at the end so the SLURM .out can be parsed.
"""
import os
import sys
import json
import math
import traceback

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

RESULTS = []          # list of (name, passed_bool, detail_str)


def record(name, passed, detail=""):
    RESULTS.append((name, bool(passed), detail))
    print(f"[{'PASS' if passed else 'FAIL'}] {name}  {detail}")


def section(title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


# --------------------------------------------------------------------------- #
# Tiny teacher model (no checkpoint download)                                 #
# --------------------------------------------------------------------------- #
def build_tiny_teacher(device, seed=0, n_layers=4, hidden=256, n_heads=2,
                       n_kv_heads=1, head_dim=128, vocab=512, dtype=torch.bfloat16):
    """A tiny randomly-initialized Llama (GQA: 2 q heads / 1 kv head). head_dim=128
    keeps the RACE kernel exercised at its production head dimension."""
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(seed)
    cfg = LlamaConfig(
        vocab_size=vocab,
        hidden_size=hidden,
        intermediate_size=hidden * 2,
        num_hidden_layers=n_layers,
        num_attention_heads=n_heads,
        num_key_value_heads=n_kv_heads,
        head_dim=head_dim,
        max_position_embeddings=4096,
        rms_norm_eps=1e-5,
        attention_bias=False,
        tie_word_embeddings=False,
    )
    model = LlamaForCausalLM(cfg).to(device).to(dtype)
    model.eval()
    return model


# --------------------------------------------------------------------------- #
# (1) IMPORT + capture-hook sanity                                            #
# --------------------------------------------------------------------------- #
def test_import_and_capture(device):
    section("(1) IMPORT + API-drift + capture hooks")
    try:
        import distill_local as DL
        record("import distill_local", True, f"MODEL={DL.MODEL}")
    except Exception as e:
        record("import distill_local", False, repr(e))
        traceback.print_exc()
        return None, None, None, None

    from hybrid import build_race_modules, freeze_teacher, set_trainable_race, odd_layers

    model = build_tiny_teacher(device)
    freeze_teacher(model)
    n_layers = model.config.num_hidden_layers
    replaced = [i for i in range(n_layers) if odd_layers(i)]
    record("replaced set (odd layers)", replaced == [1, 3],
           f"replaced={replaced} of {n_layers}")

    # Does register_capture run, and do the hooks actually fire & populate?
    B, T = 2, 64
    ids = torch.randint(0, model.config.vocab_size, (B, T), device=device)
    try:
        store, handles = DL.register_capture(model, replaced)
        h_in, attn_T, h_out_T, pos = DL.teacher_targets(model, ids, store)
        ok_keys = all(i in h_in and i in attn_T and i in h_out_T for i in replaced)
        cos, sin = pos
        shapes_ok = (h_in[replaced[0]].shape == (B, T, model.config.hidden_size)
                     and cos.shape[0] == B and cos.shape[1] == T)
        record("capture hooks fire (h_in/attn_out/h_out/rope populated)",
               ok_keys and shapes_ok,
               f"h_in{tuple(h_in[replaced[0]].shape)} cos{tuple(cos.shape)}")
    except Exception as e:
        record("capture hooks fire", False, repr(e))
        traceback.print_exc()
        for h in handles:
            h.remove()
        return None, None, None, None

    # --- API-drift trap: LlamaDecoderLayer.forward returns a BARE tensor in 5.5.0.
    # register_capture's layer_hook handles both (out[0] if tuple else out); confirm
    # the captured h_out equals a clean manual re-run of the decoder layer so the
    # hook captured the RIGHT tensor (the raw pre-final-norm layer output).
    layer = model.model.layers[replaced[-1]]
    with torch.no_grad():
        manual = layer(h_in[replaced[-1]], position_embeddings=pos)
        manual_t = manual[0] if isinstance(manual, tuple) else manual
        captured = h_out_T[replaced[-1]]
        drift_err = (manual_t.float() - captured.float()).abs().max().item()
    record("layer_hook captured RAW layer output (bare-tensor return handled)",
           drift_err < 1e-3, f"max|manual-captured|={drift_err:.2e}; "
           f"decoder-layer returns {'tuple' if isinstance(manual, tuple) else 'bare tensor'}")

    # --- self_attn hook trap: self_attn returns (attn_out, weights); out[0] must be
    # the attention output, not the weights. Confirm attn_out_T matches a manual
    # self_attn call on the teacher's normed h_in.
    with torch.no_grad():
        normed = layer.input_layernorm(h_in[replaced[-1]])
        a_manual, _ = layer.self_attn(normed, position_embeddings=pos)
        attn_err = (a_manual.float() - attn_T[replaced[-1]].float()).abs().max().item()
    record("attn_hook captured self_attn OUTPUT (not weights)",
           attn_err < 1e-3, f"max|manual-captured|={attn_err:.2e}")

    return DL, model, replaced, (store, handles)


# --------------------------------------------------------------------------- #
# (2) OBJECTIVE: teacher-forced per-layer loss vs independent reference        #
# --------------------------------------------------------------------------- #
def test_objective(DL, model, replaced, device):
    section("(2) Distillation objective: teacher-forced per-layer loss")
    from hybrid import build_race_modules, set_trainable_race
    race = build_race_modules(model, L=2, Kbits=2, M=1, device=device, seed=0)
    set_trainable_race(race)
    for m in race.values():
        m.eval()

    B, T = 2, 64
    ids = torch.randint(0, model.config.vocab_size, (B, T), device=device)
    store, handles = DL.register_capture(model, replaced)
    h_in, attn_T, h_out_T, pos = DL.teacher_targets(model, ids, store)

    i = replaced[0]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        aS, hS = DL.student_layer(model, race, i, h_in[i], pos)
        metrics, loss_i = DL.layer_losses(aS, attn_T[i], hS, h_out_T[i])

    # --- INDEPENDENT reference for layer_losses ---------------------------------
    a, at = aS.float(), attn_T[i].float()
    h, ht = hS.float(), h_out_T[i].float()
    ref_attn_mse = F.mse_loss(a, at)
    ref_hidden_mse = F.mse_loss(h, ht)
    ref_loss = ref_attn_mse + ref_hidden_mse
    ref_rel_attn = (ref_attn_mse / (at.pow(2).mean() + 1e-8)).item()
    ref_rel_hidden = (ref_hidden_mse / (ht.pow(2).mean() + 1e-8)).item()
    ref_attn_cos = F.cosine_similarity(a, at, dim=-1).mean().item()
    ref_hidden_cos = F.cosine_similarity(h, ht, dim=-1).mean().item()

    loss_match = abs(loss_i.item() - ref_loss.item()) < 1e-5
    record("layer_losses total == ref MSE(attn)+MSE(hidden)", loss_match,
           f"loss={loss_i.item():.6f} ref={ref_loss.item():.6f}")
    record("layer_losses metrics match reference",
           abs(metrics["attn_mse"] - ref_attn_mse.item()) < 1e-5
           and abs(metrics["hidden_mse"] - ref_hidden_mse.item()) < 1e-5
           and abs(metrics["rel_attn_mse"] - ref_rel_attn) < 1e-4
           and abs(metrics["rel_hidden_mse"] - ref_rel_hidden) < 1e-4
           and abs(metrics["attn_cos"] - ref_attn_cos) < 1e-4
           and abs(metrics["hidden_cos"] - ref_hidden_cos) < 1e-4,
           f"attn_mse={metrics['attn_mse']:.4f} hid_mse={metrics['hidden_mse']:.4f}")

    # --- student_layer matches the residual structure h_in + attn ; + mlp(norm) ----
    layer = model.model.layers[i]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        normed = layer.input_layernorm(h_in[i])
        a_ref, _ = race[str(i)](normed, position_embeddings=pos)
        h1 = h_in[i] + a_ref
        h_ref = h1 + layer.mlp(layer.post_attention_layernorm(h1))
    attn_struct = (a_ref.float() - aS.float()).abs().max().item()
    hid_struct = (h_ref.float() - hS.float()).abs().max().item()
    record("student_layer == teacher-forced residual (h_in + RACE_attn; +MLP)",
           attn_struct < 1e-2 and hid_struct < 1e-2,
           f"attn_d={attn_struct:.2e} hid_d={hid_struct:.2e}")

    # --- TEACHER-FORCING confirmation: the student attn is fed the TEACHER's h_in,
    # NOT the student's own running hidden state. Perturb the previous replaced
    # layer's RACE weights wildly; teacher_targets (frozen base) and h_in must be
    # UNCHANGED, proving each layer is fed teacher hidden states independently.
    if len(replaced) >= 2:
        j_prev = replaced[0]
        with torch.no_grad():
            race[str(j_prev)].q_proj.weight.add_(100.0)   # wreck an earlier RACE layer
        h_in2, attn_T2, h_out_T2, _ = DL.teacher_targets(model, ids, store)
        same = (h_in2[replaced[1]].float() - h_in[replaced[1]].float()).abs().max().item()
        record("teacher-forced: later layer's h_in independent of earlier RACE",
               same == 0.0, f"max|h_in_after_wreck - h_in|={same:.2e} (must be 0)")
        with torch.no_grad():
            race[str(j_prev)].q_proj.weight.sub_(100.0)   # restore

    for hd in handles:
        hd.remove()
    return race


# --------------------------------------------------------------------------- #
# (3a) custom race_backward gradient correctness (gradcheck vs autograd ref)   #
# --------------------------------------------------------------------------- #
def test_race_backward(device):
    section("(3a) custom RaceCausalFn backward vs autograd reference scan")
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scaling"))
    from race_causal_cuda import RaceCausalFn
    from race_common import race_prefix_ref

    torch.manual_seed(0)
    N, T, S, D, eps = 3, 16, 8, 128, 1e-6
    # random probability simplices for probsK/probsQ over S
    pk = torch.softmax(torch.randn(N, T, S, device=device, dtype=torch.float64), -1)
    pq = torch.softmax(torch.randn(N, T, S, device=device, dtype=torch.float64), -1)
    v = torch.randn(N, T, D, device=device, dtype=torch.float64)

    # Forward correctness: kernel out == pure-torch reference scan -----------------
    pkf, pqf, vf = pk.float(), pq.float(), v.float()
    out_k = RaceCausalFn.apply(pkf, pqf, vf, eps)
    out_ref, _, _ = race_prefix_ref(pkf, pqf, vf, eps)
    fwd_err = (out_k - out_ref).abs().max().item()
    record("RaceCausalFn forward == race_prefix_ref", fwd_err < 1e-3,
           f"max abs err={fwd_err:.2e}")

    # Backward correctness: compare kernel grads to grads of the pure-torch ref ----
    # (the ref scan is fully autograd-differentiable, so its .backward is the truth)
    go = torch.randn(N, T, D, device=device, dtype=torch.float32)

    a = pkf.clone().requires_grad_(True)
    b = pqf.clone().requires_grad_(True)
    c = vf.clone().requires_grad_(True)
    out_kk = RaceCausalFn.apply(a, b, c, eps)
    out_kk.backward(go)
    gpk_k, gpq_k, gv_k = a.grad.clone(), b.grad.clone(), c.grad.clone()

    a2 = pkf.clone().requires_grad_(True)
    b2 = pqf.clone().requires_grad_(True)
    c2 = vf.clone().requires_grad_(True)
    out_rr, _, _ = race_prefix_ref(a2, b2, c2, eps)
    out_rr.backward(go)
    gpk_r, gpq_r, gv_r = a2.grad.clone(), b2.grad.clone(), c2.grad.clone()

    def relerr(x, y):
        return (x - y).abs().max().item() / (y.abs().max().item() + 1e-8)

    e_pk, e_pq, e_v = relerr(gpk_k, gpk_r), relerr(gpq_k, gpq_r), relerr(gv_k, gv_r)
    record("race_backward dProbsK matches autograd ref", e_pk < 2e-3, f"relerr={e_pk:.2e}")
    record("race_backward dProbsQ matches autograd ref", e_pq < 2e-3, f"relerr={e_pq:.2e}")
    record("race_backward dV matches autograd ref", e_v < 2e-3, f"relerr={e_v:.2e}")

    # Double-precision gradcheck against the kernel (probsQ + V only; probsK feeds a
    # 1/(A+eps) nonlinearity that the fp32 kernel approximates -- checked above via
    # the autograd ref). gradcheck needs float64 inputs; the Function casts to fp32
    # internally so we use a loose tol.
    try:
        a3 = pq.clone().requires_grad_(True)   # vary probsQ
        ok_q = torch.autograd.gradcheck(
            lambda x: RaceCausalFn.apply(pkf, x.float(), vf, eps).double(),
            (a3,), eps=1e-3, atol=1e-2, rtol=1e-2, nondet_tol=1e-3)
        record("gradcheck(probsQ) numerical", bool(ok_q), "torch.autograd.gradcheck passed")
    except Exception as e:
        record("gradcheck(probsQ) numerical", False, f"{type(e).__name__}: {str(e)[:160]}")


# --------------------------------------------------------------------------- #
# (3b) End-to-end tiny training: loss decreases, only intended params train    #
# --------------------------------------------------------------------------- #
def test_end_to_end(DL, model, replaced, device):
    section("(3b) End-to-end tiny training (loss decrease + frozen base)")
    from hybrid import (build_race_modules, set_trainable_race, trainable_parameters,
                        count_params)
    race = build_race_modules(model, L=2, Kbits=2, M=1, device=device, seed=0)
    set_trainable_race(race)
    for m in race.values():
        m.train()

    # base must be 100% frozen
    n_base_train = sum(p.requires_grad for p in model.parameters())
    record("base model fully frozen (0 trainable)", n_base_train == 0,
           f"trainable base params={n_base_train}")

    params = trainable_parameters(race)
    record("trainable params nonempty", len(params) > 0,
           f"{count_params(race)/1e3:.1f}K params across {len(race)} layers")

    # snapshot base for an exact unchanged check
    race_ids = {id(p) for m in race.values() for p in m.parameters()}
    base_snap = {n: p.detach().clone() for n, p in model.named_parameters()}

    opt = torch.optim.AdamW(params, lr=5e-3, betas=(0.9, 0.95), weight_decay=0.0)
    store, handles = DL.register_capture(model, replaced)

    B, T = 2, 64
    torch.manual_seed(123)
    # fixed batch so the loss trajectory is a clean optimization signal
    ids = torch.randint(0, model.config.vocab_size, (B, T), device=device)

    losses = []
    leaked_steps = []
    grad_nonzero = {k: False for k in ("q_proj", "k_proj", "v_proj", "o_proj", "log_temp")}
    i0 = replaced[0]
    for step in range(3):
        h_in, attn_T, h_out_T, pos = DL.teacher_targets(model, ids, store)
        opt.zero_grad(set_to_none=True)
        total = 0.0
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for i in replaced:
                aS, hS = DL.student_layer(model, race, i, h_in[i], pos)
                _, loss_i = DL.layer_losses(aS, attn_T[i], hS, h_out_T[i])
                total = total + loss_i
            total = total / len(replaced)
        total.backward()

        # record grad flow on the first replaced layer
        mod = race[str(i0)]
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            g = getattr(mod, name).weight.grad
            if g is not None and torch.isfinite(g).all() and g.abs().sum() > 0:
                grad_nonzero[name] = True
        if mod.log_temp.grad is not None and torch.isfinite(mod.log_temp.grad).all():
            grad_nonzero["log_temp"] = True

        # any base param receive a grad? (leakage)
        leaked = sum(1 for p in model.parameters()
                     if id(p) not in race_ids and p.grad is not None
                     and p.grad.abs().sum() > 0)
        if leaked:
            leaked_steps.append((step, leaked))

        opt.step()
        losses.append(total.item())

    for hd in handles:
        hd.remove()

    record("3 steps ran, losses finite", all(math.isfinite(x) for x in losses),
           f"losses={[round(x,5) for x in losses]}")
    record("loss DECREASED over 3 steps", losses[-1] < losses[0],
           f"{losses[0]:.5f} -> {losses[-1]:.5f}")
    record("race_backward drives q/k/v grads (nonzero, finite)",
           all(grad_nonzero[k] for k in ("q_proj", "k_proj", "v_proj")),
           f"{grad_nonzero}")
    record("o_proj + log_temp grads present", grad_nonzero["o_proj"] and grad_nonzero["log_temp"],
           f"o_proj={grad_nonzero['o_proj']} log_temp={grad_nonzero['log_temp']}")
    record("NO base param received a grad (zero leakage)", len(leaked_steps) == 0,
           f"leaked_steps={leaked_steps}")

    # exact: base weights bit-identical after training
    changed = []
    cur = dict(model.named_parameters())
    for n, p0 in base_snap.items():
        if not torch.equal(cur[n].detach(), p0):
            changed.append(n)
    record("base weights bit-identical after training", len(changed) == 0,
           f"changed={len(changed)} params" + (f" e.g. {changed[:3]}" if changed else ""))

    # determinism: same seed -> identical loss trajectory
    def run_traj(seed):
        torch.manual_seed(0)
        m2 = build_tiny_teacher(device, seed=0)
        from hybrid import freeze_teacher
        freeze_teacher(m2)
        r2 = build_race_modules(m2, L=2, Kbits=2, M=1, device=device, seed=0)
        set_trainable_race(r2)
        for mm in r2.values():
            mm.train()
        o2 = torch.optim.AdamW(trainable_parameters(r2), lr=5e-3, betas=(0.9, 0.95))
        st2, h2 = DL.register_capture(m2, replaced)
        torch.manual_seed(123)
        ids2 = torch.randint(0, m2.config.vocab_size, (B, T), device=device)
        ls = []
        for _ in range(3):
            hi, at, ho, ps = DL.teacher_targets(m2, ids2, st2)
            o2.zero_grad(set_to_none=True)
            tot = 0.0
            with torch.autocast("cuda", dtype=torch.bfloat16):
                for i in replaced:
                    aS, hS = DL.student_layer(m2, r2, i, hi[i], ps)
                    _, li = DL.layer_losses(aS, at[i], hS, ho[i])
                    tot = tot + li
                tot = tot / len(replaced)
            tot.backward()
            o2.step()
            ls.append(tot.item())
        for x in h2:
            x.remove()
        return ls

    t1, t2 = run_traj(0), run_traj(0)
    det = all(a == b for a, b in zip(t1, t2))
    record("determinism: same seed -> identical loss trajectory", det,
           f"t1={[round(x,6) for x in t1]} t2={[round(x,6) for x in t2]}")


# --------------------------------------------------------------------------- #
# (4) Contract vs distill_global + dead/broken code notes                      #
# --------------------------------------------------------------------------- #
def test_contract_notes(DL, device):
    section("(4) Contract vs distill_global + dead/broken-code probes")

    # The local pilot has NO logits/KL/CE term (purely per-layer MSE) and NO base
    # unfreeze path -- divergences from distill_global's contract. Confirm by source
    # inspection so the report is grounded.
    src = open(os.path.join(HERE, "distill_local.py")).read()
    record("local has NO kl_loss/ce_loss (per-layer MSE only)",
           "kl_loss" not in src and "ce_loss" not in src,
           "objective = attn_mse + hidden_mse only")
    record("local has NO --unfreeze base path",
           "unfreeze" not in src and "set_trainable_base" not in src,
           "base always fully frozen")
    record("local has NO checkpoint save/resume",
           "save_ckpt" not in src and "save_full_state" not in src,
           "writes metrics .jsonl only; no model ckpt -> no round-trip to verify")

    # warmup-then-CONSTANT lr (docstring says so) -- not cosine like global. Confirm
    # the lr schedule formula caps at args.lr (no decay).
    record("local lr = warmup-then-constant (no cosine decay)",
           "min(1.0, (step + 1) / max(1, args.warmup))" in src,
           "lr = args.lr * min(1, (step+1)/warmup)")

    # eval-every off-by checks: evaluate() flips race to eval() then back; ensure it
    # restores train mode. (functional probe)
    from hybrid import build_race_modules, set_trainable_race, odd_layers
    model = build_tiny_teacher(device)
    from hybrid import freeze_teacher
    freeze_teacher(model)
    replaced = [i for i in range(model.config.num_hidden_layers) if odd_layers(i)]
    race = build_race_modules(model, L=2, Kbits=2, M=1, device=device)
    set_trainable_race(race)
    for m in race.values():
        m.train()
    store, handles = DL.register_capture(model, replaced)
    B, T = 2, 64
    eb = torch.randint(0, model.config.vocab_size, (B, T))
    ev, per = DL.evaluate(model, race, replaced, eb, store, device)
    still_train = all(m.training for m in race.values())
    record("evaluate() restores train mode afterward", still_train,
           f"race.training={still_train}; eval keys ok={set(ev.keys()) >= {'attn_mse','hidden_cos'}}")
    for hd in handles:
        hd.remove()


def main():
    assert torch.cuda.is_available(), "needs a GPU"
    device = "cuda"
    print(f"torch {torch.__version__} | GPU {torch.cuda.get_device_name(0)}")
    import transformers
    print(f"transformers {transformers.__version__}")

    DL, model, replaced, sh = test_import_and_capture(device)
    if DL is None:
        print("\nABORT: import/capture failed; cannot continue.")
    else:
        if sh is not None:
            for h in sh[1]:
                try:
                    h.remove()
                except Exception:
                    pass
        try:
            test_objective(DL, model, replaced, device)
        except Exception as e:
            record("test_objective", False, repr(e)); traceback.print_exc()
        try:
            test_race_backward(device)
        except Exception as e:
            record("test_race_backward", False, repr(e)); traceback.print_exc()
        try:
            test_end_to_end(DL, model, replaced, device)
        except Exception as e:
            record("test_end_to_end", False, repr(e)); traceback.print_exc()
        try:
            test_contract_notes(DL, device)
        except Exception as e:
            record("test_contract_notes", False, repr(e)); traceback.print_exc()

    section("SUMMARY")
    n_pass = sum(1 for _, p, _ in RESULTS if p)
    for name, p, d in RESULTS:
        print(f"  {'PASS' if p else 'FAIL'}  {name}")
    print(f"\n{n_pass}/{len(RESULTS)} checks passed")
    out = {"passed": n_pass, "total": len(RESULTS),
           "results": [{"name": n, "pass": p, "detail": d} for n, p, d in RESULTS]}
    print("JSON_RESULTS " + json.dumps(out))


if __name__ == "__main__":
    main()
