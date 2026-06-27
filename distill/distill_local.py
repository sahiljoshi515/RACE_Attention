"""Teacher-forced (local) distillation of Llama attention layers into RACE layers.

Frozen teacher = meta-llama/Llama-3.2-3B-Instruct. For each replaced (odd) layer
we feed the RACE module the TEACHER's input hidden state at that layer and match:
  * attention-output MSE   : MSE(student attn out, teacher attn out)
  * hidden-state MSE        : MSE(student layer out, teacher layer out)
Only the RACE q/k/v/o + log_temp train (the custom CUDA race_backward produces the
q/k/v gradients). Everything else is frozen. Logs per-layer attn/hidden MSE +
cosine, total loss, throughput, and peak memory; evals on a fixed held-out batch.
"""
import os
import sys
import json
import time
import math
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transformers import AutoModelForCausalLM
from data_fineweb import get_tokenizer, make_eval_and_train
from hybrid import build_race_modules, freeze_teacher, set_trainable_race, trainable_parameters, count_params, odd_layers

MODEL = "meta-llama/Llama-3.2-3B-Instruct"


def register_capture(model, replaced):
    """Hooks per replaced layer capture: the layer's RAW input (h_in) and RAW output
    (h_out, pre-final-norm) and the self_attn output; plus the rotary (cos,sin).
    We capture the raw decoder-layer output directly (NOT hidden_states[-1], which
    transformers overwrites with the post-final-RMSNorm tensor for the last layer)."""
    store = {"h_in": {}, "h_out": {}, "attn_out": {}, "rope": None}
    handles = []
    for i in replaced:
        def layer_hook(mod, inp, out, i=i):
            store["h_in"][i] = inp[0].detach()
            store["h_out"][i] = (out[0] if isinstance(out, tuple) else out).detach()
        handles.append(model.model.layers[i].register_forward_hook(layer_hook))

        def attn_hook(mod, inp, out, i=i):
            store["attn_out"][i] = out[0].detach()
        handles.append(model.model.layers[i].self_attn.register_forward_hook(attn_hook))

    def rope_hook(mod, inp, out):
        store["rope"] = (out[0].detach(), out[1].detach())
    handles.append(model.model.rotary_emb.register_forward_hook(rope_hook))
    return store, handles


def teacher_targets(model, input_ids, store):
    with torch.no_grad():
        model(input_ids=input_ids, use_cache=False)
    cos, sin = store["rope"]
    return dict(store["h_in"]), dict(store["attn_out"]), dict(store["h_out"]), (cos, sin)


def student_layer(model, race, i, h_in, pos_emb):
    """Compute student attn-output and layer-output for replaced layer i."""
    layer = model.model.layers[i]
    normed = layer.input_layernorm(h_in)
    attn_out_S, _ = race[str(i)](normed, position_embeddings=pos_emb)
    h1 = h_in + attn_out_S
    h_out_S = h1 + layer.mlp(layer.post_attention_layernorm(h1))
    return attn_out_S, h_out_S


def layer_losses(attn_S, attn_T, hout_S, hout_T):
    a = attn_S.float(); at = attn_T.float()
    h = hout_S.float(); ht = hout_T.float()
    attn_mse = F.mse_loss(a, at)
    hidden_mse = F.mse_loss(h, ht)
    # scale-invariant relative MSE (= ||S-T||^2 / ||T||^2) for interpretable trends
    rel_attn = (attn_mse / (at.pow(2).mean() + 1e-8)).item()
    rel_hidden = (hidden_mse / (ht.pow(2).mean() + 1e-8)).item()
    return {
        "attn_mse": attn_mse.item(),
        "hidden_mse": hidden_mse.item(),
        "rel_attn_mse": rel_attn,
        "rel_hidden_mse": rel_hidden,
        "attn_cos": F.cosine_similarity(a, at, dim=-1).mean().item(),
        "hidden_cos": F.cosine_similarity(h, ht, dim=-1).mean().item(),
    }, attn_mse + hidden_mse


def evaluate(model, race, replaced, eval_batch, store, device):
    model_was_train = any(m.training for m in race.values())
    for m in race.values():
        m.eval()
    with torch.no_grad():
        h_in, attn_T, h_out_T, pos = teacher_targets(model, eval_batch.to(device), store)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            per = {}
            for i in replaced:
                aS, hS = student_layer(model, race, i, h_in[i], pos)
                m, _ = layer_losses(aS, attn_T[i], hS, h_out_T[i])
                per[i] = m
    if model_was_train:
        for m in race.values():
            m.train()
    keys = ["attn_mse", "hidden_mse", "rel_attn_mse", "rel_hidden_mse", "attn_cos", "hidden_cos"]
    return {k: sum(per[i][k] for i in replaced) / len(replaced) for k in keys}, per


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--L", type=int, default=2)
    p.add_argument("--Kbits", type=int, default=2)
    p.add_argument("--M", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=4096)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--clip", type=float, default=1.0, help="grad-norm clip (0=off)")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    device = "cuda"
    torch.manual_seed(args.seed)
    S = args.L * (1 << args.Kbits)
    out_path = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "results", f"metrics_S{S}.jsonl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    print(f"GPU: {torch.cuda.get_device_name(0)} | config L={args.L} K={args.Kbits} M={args.M} S={S} "
          f"B={args.batch_size} T={args.seq_len} steps={args.steps}")

    tok = get_tokenizer()
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(device)
    freeze_teacher(model)
    n_layers = model.config.num_hidden_layers
    replaced = [i for i in range(n_layers) if odd_layers(i)]
    print(f"replacing {len(replaced)} layers: {replaced}")

    race = build_race_modules(model, L=args.L, Kbits=args.Kbits, M=args.M,
                              device=device, seed=args.seed)
    set_trainable_race(race)
    print(f"trainable RACE params: {count_params(race)/1e6:.1f}M across {len(race)} layers")

    # sanity: nothing in the base model is trainable
    n_base_train = sum(p.requires_grad for p in model.parameters())
    assert n_base_train == 0, f"base model has {n_base_train} trainable params!"

    opt = torch.optim.AdamW(trainable_parameters(race), lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=0.0)
    store, handles = register_capture(model, replaced)

    eval_batches, train_gen = make_eval_and_train(
        tok, seq_length=args.seq_len, batch_size=args.batch_size,
        num_eval_batches=1, max_train_batches=args.steps, seed=args.seed)
    eval_batch = eval_batches[0]
    print(f"eval batch {tuple(eval_batch.shape)}; starting training")

    logf = open(out_path, "w")
    tokens_done = 0
    for m in race.values():
        m.train()

    params = trainable_parameters(race)
    for step, batch in enumerate(train_gen):
        # linear warmup then constant
        lr = args.lr * min(1.0, (step + 1) / max(1, args.warmup))
        for g in opt.param_groups:
            g["lr"] = lr
        torch.cuda.synchronize(); t0 = time.perf_counter()
        h_in, attn_T, h_out_T, pos = teacher_targets(model, batch.to(device), store)

        opt.zero_grad(set_to_none=True)
        per = {}
        total = 0.0
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for i in replaced:
                aS, hS = student_layer(model, race, i, h_in[i], pos)
                metrics, loss_i = layer_losses(aS, attn_T[i], hS, h_out_T[i])
                per[i] = metrics
                total = total + loss_i
            total = total / len(replaced)
        total.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(params, args.clip) if args.clip > 0 else 0.0
        opt.step()
        torch.cuda.synchronize(); dt = time.perf_counter() - t0

        tokens_done += batch.numel()
        keys = ["attn_mse", "hidden_mse", "rel_attn_mse", "rel_hidden_mse", "attn_cos", "hidden_cos"]
        mean = {k: sum(per[i][k] for i in replaced) / len(replaced) for k in keys}
        rec = {"step": step, "total_loss": float(total.item()), "lr": lr,
               "grad_norm": float(gnorm), "mean": mean, "per_layer": per,
               "tokens": tokens_done, "tok_s": batch.numel() / dt,
               "mem_gb": torch.cuda.max_memory_allocated() / 1e9}
        logf.write(json.dumps(rec) + "\n"); logf.flush()
        print(f"step {step:3d} | loss {total.item():.4f} gnorm {float(gnorm):.2f} | "
              f"relAttn {mean['rel_attn_mse']:.3f} relHid {mean['rel_hidden_mse']:.3f} | "
              f"attnCos {mean['attn_cos']:.3f} hidCos {mean['hidden_cos']:.3f} | "
              f"{rec['tok_s']:.0f} tok/s | {rec['mem_gb']:.1f} GB")

        if step % args.eval_every == 0:
            ev, _ = evaluate(model, race, replaced, eval_batch, store, device)
            logf.write(json.dumps({"step": step, "eval": ev}) + "\n"); logf.flush()
            print(f"   [eval@{step}] relAttn {ev['rel_attn_mse']:.3f} relHid {ev['rel_hidden_mse']:.3f} "
                  f"attnMSE {ev['attn_mse']:.4f} hidMSE {ev['hidden_mse']:.4f} "
                  f"attnCos {ev['attn_cos']:.3f} hidCos {ev['hidden_cos']:.3f}")

    for h in handles:
        h.remove()
    logf.close()
    print(f"done; wrote {out_path}")


if __name__ == "__main__":
    main()
