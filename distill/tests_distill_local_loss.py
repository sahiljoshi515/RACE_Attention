"""Focused re-test of the two FAILs in tests_distill_local.py, to classify them as
real distill_local bugs vs tiny-model/test-harness artifacts.

(A) rotary cos shape: confirm the capture store IS fully populated and correct; the
    earlier FAIL was a bad assertion (cos is [1,T,head_dim] by design, not [B,...]).

(B) loss-decrease: the earlier run used lr=5e-3 on a near-minimum init (RACE copied
    from the teacher's q/k/v/o on a RANDOM tiny teacher -> starting loss ~0.012,
    AdamW overshoots). Re-run with a SANE optimization setup:
      * perturb the RACE q/k/v/o away from the teacher init so there is real signal
        to recover (otherwise we are optimizing noise around a minimum),
      * small lr, more steps, full-batch GD on a FIXED batch,
    and assert the teacher-forced MSE objective DECREASES. This isolates whether the
    custom-backward gradient is a genuine DESCENT direction (sign-correct).
"""
import os
import sys
import math
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from tests_distill_local import build_tiny_teacher


def main():
    assert torch.cuda.is_available()
    device = "cuda"
    import distill_local as DL
    from hybrid import (build_race_modules, freeze_teacher, set_trainable_race,
                        trainable_parameters, odd_layers)

    print(f"torch {torch.__version__} | GPU {torch.cuda.get_device_name(0)}")

    # (A) capture store fully populated, with CORRECT (documented) shapes.
    model = build_tiny_teacher(device)
    freeze_teacher(model)
    replaced = [i for i in range(model.config.num_hidden_layers) if odd_layers(i)]
    store, handles = DL.register_capture(model, replaced)
    B, T = 2, 64
    ids = torch.randint(0, model.config.vocab_size, (B, T), device=device)
    h_in, attn_T, h_out_T, pos = DL.teacher_targets(model, ids, store)
    cos, sin = pos
    hd = model.config.head_dim
    # rotary cos/sin are [1,T,head_dim] (batch-broadcast) -- correct, not a bug
    capture_ok = (
        all(i in h_in and i in attn_T and i in h_out_T for i in replaced)
        and h_in[replaced[0]].shape == (B, T, model.config.hidden_size)
        and attn_T[replaced[0]].shape == (B, T, model.config.hidden_size)
        and cos.shape == (1, T, hd) and sin.shape == (1, T, hd)
    )
    print(f"[{'PASS' if capture_ok else 'FAIL'}] (A) capture store fully populated; "
          f"cos{tuple(cos.shape)} sin{tuple(sin.shape)} (rotary is [1,T,hd] by design)")
    for h in handles:
        h.remove()

    # (B) loss-decrease with a sane setup -----------------------------------------
    race = build_race_modules(model, L=2, Kbits=2, M=1, device=device, seed=0)
    set_trainable_race(race)
    # Perturb away from the teacher init so there is a real target to descend toward.
    torch.manual_seed(7)
    with torch.no_grad():
        for m in race.values():
            for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                w = getattr(m, name).weight
                w.add_(0.05 * torch.randn_like(w))
    for m in race.values():
        m.train()

    params = trainable_parameters(race)
    opt = torch.optim.AdamW(params, lr=2e-4, betas=(0.9, 0.95), weight_decay=0.0)
    store, handles = DL.register_capture(model, replaced)
    torch.manual_seed(123)
    ids = torch.randint(0, model.config.vocab_size, (B, T), device=device)

    losses = []
    for step in range(40):
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
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        losses.append(total.item())
    for h in handles:
        h.remove()

    finite = all(math.isfinite(x) for x in losses)
    decreased = losses[-1] < losses[0]
    # also: trajectory is broadly monotone-ish (min near the end)
    min_at_end = min(range(len(losses)), key=lambda k: losses[k]) >= len(losses) // 2
    print(f"loss[0]={losses[0]:.5f} loss[-1]={losses[-1]:.5f} min={min(losses):.5f}")
    print(f"trajectory (every 5): {[round(losses[k],5) for k in range(0,40,5)]}")
    print(f"[{'PASS' if (finite and decreased) else 'FAIL'}] (B) teacher-forced MSE "
          f"DECREASES with sane lr (custom backward is a descent direction); "
          f"min_at_end={min_at_end}")

    print("DONE_LOSS_RETEST")


if __name__ == "__main__":
    main()
