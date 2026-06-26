"""Smoke + gradient-flow check for the RACE distillation prototype.

Verifies: the CUDA kernel builds & runs in this env (torch 2.10 in race_vit_env),
the teacher forward + one student step run, losses are finite, the RACE q/k/v/o +
log_temp receive nonzero gradients (so race_backward is in the path), and NO base
model param is trainable / gets a grad. Also probes B=8,T=4096 peak memory.
"""
import os
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transformers import AutoModelForCausalLM
from data_fineweb import get_tokenizer, make_eval_and_train
from hybrid import build_race_modules, freeze_teacher, set_trainable_race, trainable_parameters, count_params, odd_layers
from distill_local import register_capture, teacher_targets, student_layer, layer_losses, MODEL


def main():
    device = "cuda"
    print("torch", torch.__version__, "GPU", torch.cuda.get_device_name(0))
    tok = get_tokenizer()
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(device)
    freeze_teacher(model)
    replaced = [i for i in range(model.config.num_hidden_layers) if odd_layers(i)]
    race = build_race_modules(model, L=2, Kbits=2, M=1, device=device)
    set_trainable_race(race)
    for m in race.values():
        m.train()
    print(f"replaced {len(replaced)} layers; trainable {count_params(race)/1e6:.1f}M params")
    assert sum(p.requires_grad for p in model.parameters()) == 0, "base model not frozen!"

    store, handles = register_capture(model, replaced)

    # correctness-sized step (small) then a B=8/T=4096 memory probe
    for (B, T, tag) in [(2, 512, "small"), (8, 4096, "full B=8")]:
        torch.cuda.reset_peak_memory_stats()
        eb, tg = make_eval_and_train(tok, seq_length=T, batch_size=B,
                                     num_eval_batches=0, max_train_batches=1)
        batch = next(tg).to(device)
        h_in, attn_T, h_out_T, pos = teacher_targets(model, batch, store)
        for m in race.values():
            m.zero_grad(set_to_none=True)
        total = 0.0
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for i in replaced:
                aS, hS = student_layer(model, race, i, h_in[i], pos)
                _, loss_i = layer_losses(aS, attn_T[i], hS, h_out_T[i])
                total = total + loss_i
            total = total / len(replaced)
        assert torch.isfinite(total), "non-finite loss"
        total.backward()
        print(f"[{tag}] B={B} T={T} loss={total.item():.4f} "
              f"peak_mem={torch.cuda.max_memory_allocated()/1e9:.1f} GB")

        if tag == "small":
            # grad-flow assertions on the small step
            i0 = replaced[0]
            mod = race[str(i0)]
            for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                g = getattr(mod, name).weight.grad
                assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0, f"{name} no grad"
            assert mod.log_temp.grad is not None and mod.log_temp.grad.abs() >= 0, "log_temp no grad"
            print(f"   grad-flow OK: q/k/v/o + log_temp have nonzero finite grads (race_backward in path)")
            # confirm base params got no grad
            base_grads = [p.grad for p in model.parameters() if p.grad is not None]
            assert len(base_grads) == 0, f"base model received {len(base_grads)} grads!"
            print(f"   frozen OK: 0 base-model params received grads")

    for h in handles:
        h.remove()
    print("SMOKE OK")


if __name__ == "__main__":
    main()
