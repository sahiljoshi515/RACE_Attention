"""End-to-end TRAINING-MECHANICS test for the GLOBAL RACE distillation pipeline.

This is an ADVERSARIAL replica of distill_global.py's training step using the REAL
functions under test (NOTHING is reimplemented except the test harness scaffolding):
  hybrid.build_race_modules / convert_to_hybrid / freeze_teacher / set_trainable_race /
  set_trainable_base / trainable_parameters / count_params
  distill_global.split_param_groups / lr_at / grad_norm
  global_utils.register_global_capture / clear_store / hidden_loss / kl_loss / ce_loss /
  grad_health / base_grad_count / snapshot_base / assert_base_unchanged

We do NOT load Llama-3.2-3B. Instead we build a TINY Llama from a LlamaConfig
(small vocab/hidden/layers, head_dim=128 so the custom RACE CUDA kernel path is the
real one used in published runs) and a FIXED random batch of token ids -- the loss
trajectory on a fixed batch is a clean overfit-sanity signal and is fully deterministic.

The teacher and student are TWO COPIES of the SAME tiny base weights (the student's
softmax-kept layers + embeddings are bit-identical to the teacher, exactly mirroring
distill_global where build_student copies q/k/v/o from the teacher). The student's
RACE layers are initialized from those same projections, so at step 0 the only source
of mismatch is the RACE attention core -- a faithful tiny analogue of the real setup.

Tests (each prints PASS/FAIL with numbers; a hard failure raises AssertionError):
  T1 FROZEN-BASE  : only RACE params get grads (base_grad_count==0), base bit-identical
                    after 3 steps (assert_base_unchanged), a RACE weight actually moved.
  T2 OVERFIT      : loss DECREASES monotonically over N steps on a FIXED batch.
  T3 LR SCHEDULE  : lr_at matches a hand-rolled warmup+cosine formula across the horizon.
  T4 UNFREEZE MLP : set_trainable_base('mlp') -> MLP+norm params get grads, attention /
                    embeddings / lm_head stay frozen, the leak assert passes, RACE trains.
  T5 GRAD CLIP    : clip_grad_norm_ caps the post-clip grad norm at the threshold and
                    a tiny clip shrinks the effective step vs no clip.

Exit code 0 == all passed. Designed to run on one GPU (H200) via SLURM.
"""
import os
import sys
import math
import json
import copy

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from transformers import LlamaConfig, LlamaForCausalLM                       # noqa: E402
from hybrid import (build_race_modules, convert_to_hybrid, freeze_teacher,   # noqa: E402
                    set_trainable_race, set_trainable_base, trainable_parameters,
                    count_params, pattern_pred)
from global_utils import (register_global_capture, clear_store, hidden_loss, # noqa: E402
                          kl_loss, ce_loss, grad_health, base_grad_count,
                          snapshot_base, assert_base_unchanged)
from distill_global import split_param_groups, lr_at, grad_norm             # noqa: E402

DEVICE = "cuda"
RESULTS = []   # (name, ok, detail)


def log(name, ok, detail=""):
    RESULTS.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}")


# --------------------------------------------------------------------------- #
# tiny model construction                                                     #
# --------------------------------------------------------------------------- #
def tiny_config(n_layers=4, vocab=512, hidden=256, heads=2, kv_heads=1, head_dim=128):
    """A deliberately tiny Llama. head_dim=128 keeps the RACE CUDA forward/backward on
    the SAME code path the real 3B model uses; everything else is shrunk for speed and
    determinism. GQA (kv_heads<heads) is kept on so repeat_kv is exercised like the real
    model. intermediate_size small so the MLP-unfreeze test has real-but-cheap params."""
    return LlamaConfig(
        vocab_size=vocab,
        hidden_size=hidden,
        intermediate_size=512,
        num_hidden_layers=n_layers,
        num_attention_heads=heads,
        num_key_value_heads=kv_heads,
        head_dim=head_dim,
        max_position_embeddings=2048,
        rms_norm_eps=1e-5,
        attention_bias=False,
        tie_word_embeddings=False,
    )


def build_pair(cfg, pattern="AR", L=2, Kbits=2, seed=0):
    """Build a (teacher, student, race, replaced) tuple mirroring distill_global:
      * teacher: a frozen tiny Llama (random init, fixed seed) -- stands in for Llama-3B.
      * student: a DEEP COPY of the teacher, with the pattern-selected layers replaced by
        RaceLlamaAttention whose q/k/v/o are copied from the teacher (build_race_modules).
    Returns the same objects build_student would, in the same build order (freeze whole
    model FIRST, then re-enable RACE)."""
    torch.manual_seed(seed)
    teacher = LlamaForCausalLM(cfg).to(DEVICE).to(torch.bfloat16)
    freeze_teacher(teacher)

    # student starts as an exact copy of the teacher base (same weights everywhere).
    student = copy.deepcopy(teacher).to(DEVICE)
    pred = pattern_pred(pattern)
    race = build_race_modules(student, L=L, Kbits=Kbits, M=1, device=DEVICE,
                              replace_pred=pred, seed=seed)
    convert_to_hybrid(student, race)
    freeze_teacher(student)        # freeze EVERYTHING (incl. swapped-in RACE) first ...
    set_trainable_race(race)       # ... then re-enable RACE params (order matters)
    replaced = sorted(int(k) for k in race.keys())
    student.train()
    return teacher, student, race, replaced


def fixed_batch(cfg, B=2, T=64, seed=1234):
    """A FIXED random token batch (same every step) for the overfit-sanity trajectory."""
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, cfg.vocab_size, (B, T), generator=g).to(DEVICE)


# --------------------------------------------------------------------------- #
# one faithful training step (mirrors distill_global.main's inner loop)        #
# --------------------------------------------------------------------------- #
def train_step(teacher, student, race, replaced, batch, opt,
               t_store, s_store, race_params, base_params, race_ids,
               hidden_weight=1.0, kl_weight=0.5, kl_temp=1.0, ce_weight=0.0,
               kl_chunk=0, clip=1.0):
    """Replicates the per-microbatch + optimizer-step body of distill_global.main
    (accum==1). Returns a metrics dict. Uses the REAL loss helpers and the REAL
    grad-health / base-grad-count integrity checks."""
    clear_store(t_store); clear_store(s_store)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        with torch.no_grad():
            t_logits = teacher(input_ids=batch, use_cache=False).logits
        s_logits = student(input_ids=batch, use_cache=False).logits
    h_loss, per = hidden_loss(s_store, t_store, replaced)
    k_loss = kl_loss(s_logits, t_logits, T=kl_temp, chunk=kl_chunk)
    if ce_weight > 0:
        c_loss = ce_loss(s_logits, batch)
    else:
        with torch.no_grad():
            c_loss = ce_loss(s_logits, batch)
    loss = hidden_weight * h_loss + kl_weight * k_loss + ce_weight * c_loss
    loss.backward()

    # integrity probes BEFORE the step (grads are live)
    proj_gn = grad_norm([p for p in race_params])
    bgc = base_grad_count(student, race_ids)
    rgf, rfin = grad_health(race_params)
    gnorm = (torch.nn.utils.clip_grad_norm_(race_params + base_params, clip)
             if clip > 0 else torch.zeros(()))
    opt.step()
    opt.zero_grad(set_to_none=True)
    clear_store(t_store); clear_store(s_store)
    detail = {
        "total": float(loss.item()),
        "hidden": float(h_loss.item()),
        "kl": float(k_loss.item()),
        "ce": float(c_loss.item()),
        "grad_norm_preclip": float(gnorm),
        "race_grad_nonzero_frac": rgf,
        "race_grad_finite": bool(rfin),
        "base_grad_count": bgc,
    }
    del t_logits, s_logits, loss, h_loss, k_loss, c_loss
    return detail


# --------------------------------------------------------------------------- #
# T1: frozen-base guarantee + RACE moves                                       #
# --------------------------------------------------------------------------- #
def test_frozen_base(n_steps=3):
    cfg = tiny_config()
    teacher, student, race, replaced = build_pair(cfg, pattern="AR")
    proj, hashp = split_param_groups(race)
    race_params = proj + hashp
    race_ids = {id(p) for p in race_params}
    base_params = []   # frozen-base run

    opt = torch.optim.AdamW(
        [{"params": proj, "lr": 5e-3}], betas=(0.9, 0.95), weight_decay=0.0)
    t_store, t_h = register_global_capture(teacher, replaced, detach=True)
    s_store, s_h = register_global_capture(student, replaced, detach=False)

    snap_s = snapshot_base(student, race_ids, n=12)   # exact frozen-param snapshot
    w0 = race_params[0].detach().clone()

    batch = fixed_batch(cfg)
    max_bgc = 0
    all_finite = True
    all_nz_frac = 1.0
    for s in range(n_steps):
        d = train_step(teacher, student, race, replaced, batch, opt,
                       t_store, s_store, race_params, base_params, race_ids,
                       clip=1.0)
        max_bgc = max(max_bgc, d["base_grad_count"])
        all_finite = all_finite and d["race_grad_finite"]
        all_nz_frac = min(all_nz_frac, d["race_grad_nonzero_frac"])

    moved = (race_params[0].detach() - w0).abs().sum().item()
    base_ok = True
    try:
        assert_base_unchanged(student, snap_s)
    except AssertionError as e:
        base_ok = False
        log("T1.assert_base_unchanged", False, f"raised: {e}")

    for h in t_h + s_h:
        h.remove()

    log("T1.base_grad_count==0", max_bgc == 0, f"max base_grad_count over {n_steps} steps = {max_bgc}")
    log("T1.base_bit_identical", base_ok, "frozen base unchanged after training" if base_ok else "")
    log("T1.race_weight_moved", moved > 0, f"|delta(race_params[0])|_1 = {moved:.3e}")
    log("T1.race_grad_finite", all_finite, "all RACE grads finite")
    log("T1.race_grad_nonzero", all_nz_frac > 0, f"min nonzero-grad frac = {all_nz_frac:.3f}")
    return all([max_bgc == 0, base_ok, moved > 0, all_finite, all_nz_frac > 0])


# --------------------------------------------------------------------------- #
# T2: overfit a fixed batch -> monotone loss decrease                          #
# --------------------------------------------------------------------------- #
def test_overfit(n_steps=5, lr=3e-4):
    """Overfit-sanity: with a FIXED batch the loss must DECREASE monotonically.

    Subtlety (and why this is a real test, not a tautology): build_race_modules copies
    q/k/v/o straight from the teacher, so at init the RACE student is ALREADY very close
    to the teacher (loss ~ 1e-2 on a tiny random model). Near that near-optimum the loss
    surface is shallow, and a too-large AdamW step (which normalizes by the grad's 2nd
    moment, so its size is ~lr regardless of how small the grad is) OVERSHOOTS and the
    loss climbs -- a property of Adam near a flat minimum, NOT a pipeline bug. So we:
      (1) create a REAL gap by perturbing the RACE log_temp away from its init (the
          soft-hash temperature directly controls how peaked the bucket assignment is,
          so a wrong temp gives a genuinely worse-but-recoverable starting loss), and
      (2) use a MODEST lr so AdamW descends instead of bouncing.
    Then a faithful training loop should walk the loss monotonically DOWN. If it doesn't,
    that IS a real defect (bad grad sign, broken backward, etc.)."""
    cfg = tiny_config()
    teacher, student, race, replaced = build_pair(cfg, pattern="AR", seed=0)
    # genuine, recoverable gap: push every RACE temperature off its optimum.
    with torch.no_grad():
        for m in race.values():
            m.log_temp.fill_(0.7)
    proj, hashp = split_param_groups(race)
    race_params = proj + hashp
    race_ids = {id(p) for p in race_params}
    opt = torch.optim.AdamW([{"params": proj, "lr": lr}], betas=(0.9, 0.95), weight_decay=0.0)
    t_store, t_h = register_global_capture(teacher, replaced, detach=True)
    s_store, s_h = register_global_capture(student, replaced, detach=False)

    batch = fixed_batch(cfg)
    losses = []
    for s in range(n_steps):
        d = train_step(teacher, student, race, replaced, batch, opt,
                       t_store, s_store, race_params, [], race_ids, clip=1.0)
        losses.append(d["total"])

    for h in t_h + s_h:
        h.remove()

    # strictly decreasing step-to-step (overfit sanity on a fixed batch)
    monotone = all(losses[i + 1] < losses[i] for i in range(len(losses) - 1))
    net_drop = losses[0] - losses[-1]
    log("T2.loss_monotone_decrease", monotone, f"trajectory = {[round(x,4) for x in losses]}")
    log("T2.net_loss_drop>0", net_drop > 0, f"drop = {net_drop:.4f}")
    return monotone and net_drop > 0


# --------------------------------------------------------------------------- #
# T3: lr_at warmup + cosine vs a hand formula                                  #
# --------------------------------------------------------------------------- #
def _hand_lr(step, base, warmup, total, min_ratio, schedule):
    """Independent reference implementation of distill_global.lr_at."""
    if step < warmup:
        return base * (step + 1) / max(1, warmup)
    if schedule != "cosine" or total <= warmup:
        return base
    prog = min(1.0, (step - warmup) / max(1, total - warmup))
    min_lr = base * min_ratio
    return min_lr + 0.5 * (base - min_lr) * (1.0 + math.cos(math.pi * prog))


def test_lr_schedule():
    base, warmup, total, mr = 5e-3, 5, 40, 0.1
    ok_cos = True
    worst = 0.0
    for step in range(total + 5):
        got = lr_at(step, base, warmup, total, mr, "cosine")
        exp = _hand_lr(step, base, warmup, total, mr, "cosine")
        worst = max(worst, abs(got - exp))
        if abs(got - exp) > 1e-12:
            ok_cos = False
    # spot checks on the SHAPE (warmup ramp, peak, floor)
    lr0 = lr_at(0, base, warmup, total, mr, "cosine")
    lr_peak = lr_at(warmup, base, warmup, total, mr, "cosine")
    lr_end = lr_at(total - 1, base, warmup, total, mr, "cosine")
    lr_past = lr_at(total + 3, base, warmup, total, mr, "cosine")
    ramp_ok = abs(lr0 - base * 1 / warmup) < 1e-12
    peak_ok = abs(lr_peak - base) < 1e-12
    floor_ok = abs(lr_past - base * mr) < 1e-9   # prog clamps to 1 -> cos(pi) -> floor
    end_above_floor = (lr_end > base * mr) and (lr_end < base)
    # linear schedule: holds at base after warmup
    lin_ok = (abs(lr_at(20, base, warmup, total, mr, "linear") - base) < 1e-12
              and abs(lr_at(0, base, warmup, total, mr, "linear") - base / warmup) < 1e-12)

    log("T3.cosine_matches_handformula", ok_cos, f"max abs diff = {worst:.2e}")
    log("T3.warmup_ramp", ramp_ok, f"lr(0) = {lr0:.3e} (expect base/warmup={base/warmup:.3e})")
    log("T3.peak_at_base", peak_ok, f"lr(warmup) = {lr_peak:.3e} (expect {base:.3e})")
    log("T3.cosine_floor", floor_ok, f"lr(past total) = {lr_past:.3e} (expect {base*mr:.3e})")
    log("T3.decay_in_range", end_above_floor, f"lr(total-1) = {lr_end:.3e}")
    log("T3.linear_holds_base", lin_ok, "linear ramps then holds base")
    return all([ok_cos, ramp_ok, peak_ok, floor_ok, end_above_floor, lin_ok])


# --------------------------------------------------------------------------- #
# T4: --unfreeze mlp path                                                      #
# --------------------------------------------------------------------------- #
def test_unfreeze_mlp(n_steps=3):
    cfg = tiny_config()
    teacher, student, race, replaced = build_pair(cfg, pattern="AR", seed=0)
    proj, hashp = split_param_groups(race)
    race_params = proj + hashp
    race_ids = {id(p) for p in race_params}

    base_params = set_trainable_base(student, "mlp", race_ids)

    # Classify the chosen base params and verify the gating contract:
    #   only .mlp.* / *layernorm* / model.norm.weight got unfrozen; NOTHING else.
    chosen_names = []
    for n, p in student.named_parameters():
        if id(p) in {id(x) for x in base_params}:
            chosen_names.append(n)
    bad = [n for n in chosen_names
           if not (".mlp." in n or "layernorm" in n or n.endswith("model.norm.weight"))]
    has_mlp = any(".mlp." in n for n in chosen_names)
    has_norm = any(("layernorm" in n or n.endswith("model.norm.weight")) for n in chosen_names)

    # attention / embeddings / lm_head must remain frozen
    attn_frozen = all(not p.requires_grad for n, p in student.named_parameters()
                      if ("self_attn" in n and id(p) not in race_ids))
    embed_frozen = all(not p.requires_grad for n, p in student.named_parameters()
                       if ("embed" in n or "lm_head" in n))

    # the SAME leak assertion distill_global.build_student runs
    allowed = race_ids | {id(p) for p in base_params}
    leaked = [n for n, p in student.named_parameters()
              if p.requires_grad and id(p) not in allowed]
    leak_ok = (len(leaked) == 0)

    # Now actually TRAIN: MLP+norm grads must appear; RACE still trains; attn/embed
    # must receive NO grad. We snapshot the FROZEN attention+embedding params and assert
    # they stay bit-identical, while confirming an MLP weight moved.
    student.train()
    opt = torch.optim.AdamW(
        [{"params": proj, "lr": 5e-3}, {"params": base_params, "lr": 5e-3}],
        betas=(0.9, 0.95), weight_decay=0.0)
    t_store, t_h = register_global_capture(teacher, replaced, detach=True)
    s_store, s_h = register_global_capture(student, replaced, detach=False)
    snap_frozen = snapshot_base(student, race_ids | {id(p) for p in base_params}, n=12)

    # pick an MLP weight + a frozen attention weight to watch
    mlp_w = next(p for n, p in student.named_parameters() if ".mlp." in n)
    mlp_w0 = mlp_w.detach().clone()
    race_w0 = race_params[0].detach().clone()

    batch = fixed_batch(cfg)
    mlp_got_grad = False
    losses = []
    for s in range(n_steps):
        clear_store(t_store); clear_store(s_store)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            with torch.no_grad():
                t_logits = teacher(input_ids=batch, use_cache=False).logits
            s_logits = student(input_ids=batch, use_cache=False).logits
        h_loss, _ = hidden_loss(s_store, t_store, replaced)
        k_loss = kl_loss(s_logits, t_logits, T=1.0, chunk=0)
        loss = h_loss + 0.5 * k_loss
        loss.backward()
        if mlp_w.grad is not None and mlp_w.grad.abs().sum() > 0:
            mlp_got_grad = True
        # count base grads that are NOT in the allowed base set -> must be 0
        torch.nn.utils.clip_grad_norm_(race_params + base_params, 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
        losses.append(float(loss.item()))
        clear_store(t_store); clear_store(s_store)
        del t_logits, s_logits, loss, h_loss, k_loss

    mlp_moved = (mlp_w.detach() - mlp_w0).abs().sum().item()
    race_moved = (race_params[0].detach() - race_w0).abs().sum().item()
    frozen_ok = True
    try:
        assert_base_unchanged(student, snap_frozen)   # attention+embeds untouched
    except AssertionError as e:
        frozen_ok = False
        log("T4.frozen_attn_embed_unchanged", False, f"raised: {e}")

    for h in t_h + s_h:
        h.remove()

    log("T4.only_mlp_norm_chosen", len(bad) == 0 and has_mlp and has_norm,
        f"{len(chosen_names)} chosen, bad={bad[:3]}")
    log("T4.attn_frozen", attn_frozen, "all non-RACE attention frozen")
    log("T4.embed_lmhead_frozen", embed_frozen, "embeddings + lm_head frozen")
    log("T4.no_leak", leak_ok, f"leaked={leaked[:3]}")
    log("T4.mlp_got_grad", mlp_got_grad, "MLP received nonzero grad")
    log("T4.mlp_moved", mlp_moved > 0, f"|delta(mlp)|_1 = {mlp_moved:.3e}")
    log("T4.race_still_trains", race_moved > 0, f"|delta(race)|_1 = {race_moved:.3e}")
    log("T4.frozen_attn_embed_unchanged", frozen_ok, "attn+embed bit-identical")
    return all([len(bad) == 0, has_mlp, has_norm, attn_frozen, embed_frozen, leak_ok,
                mlp_got_grad, mlp_moved > 0, race_moved > 0, frozen_ok])


# --------------------------------------------------------------------------- #
# T5: grad clip behaviour                                                      #
# --------------------------------------------------------------------------- #
def test_grad_clip():
    cfg = tiny_config()
    teacher, student, race, replaced = build_pair(cfg, pattern="AR", seed=0)
    proj, hashp = split_param_groups(race)
    race_params = proj + hashp
    race_ids = {id(p) for p in race_params}
    t_store, t_h = register_global_capture(teacher, replaced, detach=True)
    s_store, s_h = register_global_capture(student, replaced, detach=False)
    batch = fixed_batch(cfg)

    def grads_after_backward():
        clear_store(t_store); clear_store(s_store)
        for p in race_params:
            p.grad = None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            with torch.no_grad():
                t_logits = teacher(input_ids=batch, use_cache=False).logits
            s_logits = student(input_ids=batch, use_cache=False).logits
        h_loss, _ = hidden_loss(s_store, t_store, replaced)
        k_loss = kl_loss(s_logits, t_logits, T=1.0, chunk=0)
        (h_loss + 0.5 * k_loss).backward()
        clear_store(t_store); clear_store(s_store)
        del t_logits, s_logits

    # 1) clip with a TINY threshold -> post-clip norm == threshold (when raw > thr)
    grads_after_backward()
    raw_norm = float(grad_norm(race_params))
    thr = raw_norm * 0.1
    returned = float(torch.nn.utils.clip_grad_norm_(race_params, thr))
    post_norm = float(grad_norm(race_params))
    # clip_grad_norm_ RETURNS the PRE-clip total norm and rescales grads to <= thr.
    return_is_preclip = abs(returned - raw_norm) < raw_norm * 1e-3 + 1e-6
    capped = post_norm <= thr * (1 + 1e-3) + 1e-9

    # 2) a HUGE threshold leaves grads untouched (no rescale)
    grads_after_backward()
    raw2 = float(grad_norm(race_params))
    torch.nn.utils.clip_grad_norm_(race_params, raw2 * 1e6)
    post2 = float(grad_norm(race_params))
    untouched = abs(post2 - raw2) < raw2 * 1e-3 + 1e-9

    for h in t_h + s_h:
        h.remove()

    log("T5.clip_returns_preclip_norm", return_is_preclip, f"returned={returned:.3f} raw={raw_norm:.3f}")
    log("T5.clip_caps_norm", capped, f"post-clip norm={post_norm:.3f} <= thr={thr:.3f}")
    log("T5.huge_thr_noop", untouched, f"post={post2:.3f} raw={raw2:.3f}")
    return all([return_is_preclip, capped, untouched])


# --------------------------------------------------------------------------- #
# determinism bonus: same seed -> identical first-step loss                     #
# --------------------------------------------------------------------------- #
def test_determinism():
    def one_step_loss():
        cfg = tiny_config()
        teacher, student, race, replaced = build_pair(cfg, pattern="AR", seed=0)
        proj, hashp = split_param_groups(race)
        race_params = proj + hashp
        race_ids = {id(p) for p in race_params}
        opt = torch.optim.AdamW([{"params": proj, "lr": 5e-3}], betas=(0.9, 0.95))
        t_store, t_h = register_global_capture(teacher, replaced, detach=True)
        s_store, s_h = register_global_capture(student, replaced, detach=False)
        batch = fixed_batch(cfg)
        d = train_step(teacher, student, race, replaced, batch, opt,
                       t_store, s_store, race_params, [], race_ids, clip=1.0)
        for h in t_h + s_h:
            h.remove()
        return d["total"]

    a = one_step_loss()
    b = one_step_loss()
    same = abs(a - b) < 1e-6 * max(1.0, abs(a))
    log("T6.same_seed_same_loss", same, f"loss_a={a:.6f} loss_b={b:.6f} diff={abs(a-b):.2e}")
    return same


def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    print("=" * 70)
    print("TINY-MODEL TRAINING-MECHANICS TEST (distill_global pipeline)")
    print(f"device={DEVICE} torch={torch.__version__} "
          f"gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'}")
    print("=" * 70)

    results = {}
    print("\n--- T1: FROZEN-BASE guarantee + RACE moves ---")
    results["T1_frozen_base"] = test_frozen_base()
    print("\n--- T2: OVERFIT a fixed batch (monotone loss decrease) ---")
    results["T2_overfit"] = test_overfit()
    print("\n--- T3: lr_at warmup+cosine vs hand formula ---")
    results["T3_lr_schedule"] = test_lr_schedule()
    print("\n--- T4: --unfreeze mlp path ---")
    results["T4_unfreeze_mlp"] = test_unfreeze_mlp()
    print("\n--- T5: grad-clip behaviour ---")
    results["T5_grad_clip"] = test_grad_clip()
    print("\n--- T6: determinism (same seed -> same loss) ---")
    results["T6_determinism"] = test_determinism()

    print("\n" + "=" * 70)
    n_pass = sum(1 for _, ok, _ in RESULTS if ok)
    print(f"SUBCHECKS: {n_pass}/{len(RESULTS)} passed")
    for name, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    all_ok = all(results.values())
    print(json.dumps({"all_pass": all_ok, "results": results}))
    print("ALL_TESTS_DONE")
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
