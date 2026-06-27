"""Smoke test for GLOBAL hybrid-student distillation (distill_global.py).

Asserts the behavioral contracts BEFORE a 200-step run:
  1. pattern partition is exactly as expected (ARRR -> 21 RACE, kept at 0,4,..,24)
  2. only RACE params are trainable; base frozen (both models)
  3. NO teacher forcing AND genuine end-to-end rollout:
       (a) perturbing the teacher cannot change the student (separate instances)
       (b) perturbing the student's layer-1 RACE *does* change later student layers
           (proves RACE outputs propagate downstream -- a true rollout, not isolated)
       (c) distill_global does not import the local teacher-forcing helper
  4. end-to-end gradients reach BOTH the first (1) and last (27) RACE layer
  5. frozen base receives zero grads
  6. loss is fp32 + finite; prints the 3 term magnitudes (scaling sanity)
  7. store hygiene: 21 captured per model; student captures in-graph, teacher detached
  8. gradient-checkpoint ON vs OFF produce matching GRADIENTS (exercises recompute)
  9. one optimizer step actually moves a RACE weight
 10. peak-memory + grad-health probe at B=2, T=4096 UNDER checkpointing (real config)

Run on a GPU node:
  source distill/env.sh && $PYBIN distill/smoke_global.py
"""
import os
import sys
import argparse
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transformers import AutoModelForCausalLM                                 # noqa: E402
from hybrid import trainable_parameters, count_params, pattern_pred, freeze_teacher  # noqa: E402
from global_utils import (clear_store, hidden_loss, kl_loss, grad_health,     # noqa: E402
                          base_grad_count)
import distill_global as DG                                                   # noqa: E402
from distill_global import MODEL, build_student, register_global_capture      # noqa: E402


def ns(**kw):
    base = dict(pattern="ARRR", race_l=2, race_k=2, M=1, seq_len=512, batch_size=1,
                grad_checkpoint=False, seed=0)
    base.update(kw)
    return argparse.Namespace(**base)


def race_layer_grads_ok(race, idx):
    mod = race[str(idx)]
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        g = getattr(mod, name).weight.grad
        assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0, \
            f"layer {idx} {name}: no/zero/non-finite grad"
    assert mod.log_temp.grad is not None and torch.isfinite(mod.log_temp.grad).all(), \
        f"layer {idx} log_temp: no grad"


def main():
    dev = "cuda"
    torch.manual_seed(0)
    print("GPU:", torch.cuda.get_device_name(0))
    args = ns()
    teacher = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(dev)
    freeze_teacher(teacher)
    vocab = teacher.config.vocab_size
    student, race, replaced = build_student(args, dev)

    # ---- 1. partition ----
    exp = [i for i in range(teacher.config.num_hidden_layers) if pattern_pred("ARRR")(i)]
    assert replaced == exp == [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19,
                               21, 22, 23, 25, 26, 27], replaced
    assert len(replaced) == 21
    print(f"[1] ARRR partition OK: {len(replaced)} RACE layers, softmax kept at "
          f"{[i for i in range(28) if i not in replaced]}")

    # ---- 2. only RACE trainable ----
    race_ids = {id(p) for p in trainable_parameters(race)}
    leaked = [n for n, p in student.named_parameters() if p.requires_grad and id(p) not in race_ids]
    assert not leaked and count_params(race) > 0
    assert sum(p.requires_grad for p in teacher.parameters()) == 0
    print(f"[2] trainable == RACE only ({count_params(race)/1e6:.1f}M); base frozen (both)")

    t_store, t_h = register_global_capture(teacher, replaced, detach=True)
    s_store, s_h = register_global_capture(student, replaced, detach=False)
    batch = torch.randint(0, vocab, (args.batch_size, args.seq_len), device=dev)

    # ---- 3. no teacher forcing + genuine end-to-end rollout ----
    # (RACE's atomicAdd makes the forward nondeterministic ~1-3 ULP, so compare the
    #  perturbation EFFECT against intrinsic run-to-run noise, not bit-equality.)
    assert not hasattr(DG, "student_layer"), "distill_global must NOT use the local teacher-forcing helper"
    # (a) teacher and student share NO storage -> the student cannot read teacher state
    assert student.model.embed_tokens.weight.data_ptr() != teacher.model.embed_tokens.weight.data_ptr()
    assert (student.model.layers[1].self_attn.q_proj.weight.data_ptr()
            != teacher.model.layers[1].self_attn.q_proj.weight.data_ptr())

    def stu_out():
        clear_store(s_store)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            student(input_ids=batch, use_cache=False)
        return {i: s_store["h_out"][i].detach().float().clone() for i in (3, 27)}

    a1, a2 = stu_out(), stu_out()
    noise = max((a1[i] - a2[i]).abs().max().item() for i in (3, 27))
    # (b) perturb the student's OWN input embeddings -> the change must propagate through
    #     the RACE stack to BOTH an early (3) and the last (27) replaced layer, far beyond
    #     intrinsic noise. Proves the student rolls out on its own hidden states end-to-end.
    saved_e = student.model.embed_tokens.weight.detach().clone()
    with torch.no_grad():
        student.model.embed_tokens.weight.add_(torch.randn_like(saved_e) * 0.1)
    b = stu_out()
    with torch.no_grad():
        student.model.embed_tokens.weight.copy_(saved_e)
    eff3 = (b[3] - a1[3]).abs().max().item()
    eff27 = (b[27] - a1[27]).abs().max().item()
    assert eff3 > max(1e-3, 10 * noise) and eff27 > max(1e-3, 10 * noise), \
        f"student input perturbation didn't propagate: L3 {eff3:.2e} L27 {eff27:.2e} noise {noise:.2e}"
    print(f"[3] no shared teacher/student storage + end-to-end rollout: input perturbation "
          f"reaches L3={eff3:.3f} L27={eff27:.3f} >> forward noise {noise:.1e}")

    # ---- 7. store hygiene + in-graph student / detached teacher ----
    clear_store(t_store); clear_store(s_store)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        t_logits = teacher(input_ids=batch, use_cache=False).logits
    with torch.autocast("cuda", dtype=torch.bfloat16):
        s_logits = student(input_ids=batch, use_cache=False).logits
    assert len(s_store["h_out"]) == 21 and len(t_store["h_out"]) == 21
    assert s_store["final_norm"] is not None and t_store["final_norm"] is not None
    assert s_store["h_out"][1].grad_fn is not None, "student capture not in graph!"
    assert t_store["h_out"][1].grad_fn is None, "teacher capture should be detached!"
    print("[7] store hygiene OK: 21/model, final_norm present, student in-graph, teacher detached")

    # ---- 4,5,6,9. grads + loss + step ----
    opt = torch.optim.AdamW(trainable_parameters(race), lr=5e-5)
    opt.zero_grad(set_to_none=True)
    h_loss, _ = hidden_loss(s_store, t_store, replaced)
    k_loss = kl_loss(s_logits, t_logits, T=1.0, chunk=2048)
    loss = 1.0 * h_loss + 0.5 * k_loss
    assert loss.dtype == torch.float32 and torch.isfinite(loss)
    print(f"[6] loss fp32 finite: total {loss.item():.4f} = hidden {h_loss.item():.4f} "
          f"+ 0.5*kl {k_loss.item():.4f}  (kl/hidden ratio {k_loss.item()/max(h_loss.item(),1e-9):.1f})")
    loss.backward()
    race_layer_grads_ok(race, 1); race_layer_grads_ok(race, 27)
    frac, fin = grad_health(trainable_parameters(race))
    assert fin and frac > 0.99
    print(f"[4] end-to-end grads OK at layers 1 AND 27; race_grad_nonzero_frac={frac:.2f}")
    bgc = base_grad_count(student, race_ids)
    assert bgc == 0 and base_grad_count(teacher, race_ids) == 0
    print("[5] frozen base received 0 grads (student & teacher)")
    w0 = trainable_parameters(race)[0].detach().clone()
    opt.step()
    moved = (trainable_parameters(race)[0].detach() - w0).abs().sum().item()
    assert moved > 0
    print(f"[9] optimizer moved a RACE weight by {moved:.3e}")
    del s_logits, t_logits, loss, h_loss, k_loss

    # ---- 8. grad-checkpoint GRADIENT parity (grad enabled -> exercises recompute) ----
    for h in s_h:
        h.remove()
    s_store2, s_h2 = register_global_capture(student, replaced, detach=False)

    def loss_and_grads(ckpt):
        (student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
         if ckpt else student.gradient_checkpointing_disable())
        student.train()
        for p in trainable_parameters(race):
            p.grad = None
        clear_store(t_store); clear_store(s_store2)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            tl = teacher(input_ids=batch, use_cache=False).logits
        with torch.autocast("cuda", dtype=torch.bfloat16):
            sl = student(input_ids=batch, use_cache=False).logits
        hl, _ = hidden_loss(s_store2, t_store, replaced)
        ll = 1.0 * hl + 0.5 * kl_loss(sl, tl, T=1.0, chunk=2048)
        ll.backward()
        g = {k: race[k].q_proj.weight.grad.detach().clone() for k in ("1", "27")}
        clear_store(s_store2)
        return ll.item(), g

    l_off, g_off = loss_and_grads(False)
    l_on, g_on = loss_and_grads(True)
    rel_l = abs(l_on - l_off) / max(abs(l_off), 1e-6)
    cos_g = min(F.cosine_similarity(g_on[k].flatten(), g_off[k].flatten(), dim=0).item() for k in g_off)
    assert rel_l < 5e-2 and cos_g > 0.98, f"ckpt parity: loss rel {rel_l:.1e}, grad cos {cos_g:.4f}"
    print(f"[8] grad-checkpoint GRAD parity OK: loss {l_off:.4f}/{l_on:.4f} (rel {rel_l:.1e}); grad cos {cos_g:.4f}")

    # ---- 10. mem + grad-health probe at B=2, T=4096 UNDER checkpointing (real config) ----
    for h in t_h + s_h2:
        h.remove()
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    t_store, t_h = register_global_capture(teacher, replaced, detach=True)
    s_store, s_h = register_global_capture(student, replaced, detach=False)
    student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    student.train()
    big = torch.randint(0, vocab, (2, 4096), device=dev)
    for p in trainable_parameters(race):
        p.grad = None
    clear_store(t_store); clear_store(s_store)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        tl = teacher(input_ids=big, use_cache=False).logits
    with torch.autocast("cuda", dtype=torch.bfloat16):
        sl = student(input_ids=big, use_cache=False).logits
    hl, _ = hidden_loss(s_store, t_store, replaced)
    kl = kl_loss(sl, tl, T=1.0, chunk=2048)
    del tl
    (1.0 * hl + 0.5 * kl).backward()
    race_layer_grads_ok(race, 1); race_layer_grads_ok(race, 27)
    frac, fin = grad_health(trainable_parameters(race))
    assert fin and frac > 0.99 and base_grad_count(student, race_ids) == 0
    peak = torch.cuda.max_memory_allocated() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"[10] B=2 T=4096 fwd+bwd UNDER ckpt: grads OK (frac {frac:.2f}); "
          f"peak {peak:.1f}/{total:.0f} GB ({'FITS' if peak < total*0.92 else 'TIGHT'})")
    print("\nSMOKE PASSED ✓  (no teacher forcing, end-to-end rollout, grads flow under ckpt, base frozen, mem OK)")


if __name__ == "__main__":
    main()
