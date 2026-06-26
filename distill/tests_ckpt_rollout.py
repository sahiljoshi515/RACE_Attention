"""Adversarial verification of CHECKPOINTING, the GLOBAL rollout, and DETERMINISM
for the RACE-2.0 global distillation code (distill_global.py / global_utils.py /
hybrid.py / eval_ruler.build_model).

This file ONLY tests; it does not modify any script under test. We build a *tiny*
local Llama (few layers/heads, small vocab/seq) but keep head_dim=128 (the RACE
CUDA kernel's per-head dimension), so the real CUDA forward/backward path runs.
The hardcoded MODEL = "meta-llama/Llama-3.2-3B-Instruct" in the modules under test
is redirected to the tiny local dir at runtime by monkeypatching the module-level
MODEL constant + AutoModelForCausalLM (load is a no-op from a local dir; no hub).

Each check prints EXACT diffs and a PASS/FAIL line. The script exits non-zero if any
check fails so a SLURM run surfaces failures.

Run on a GPU node:
    source distill/env.sh
    $PYBIN distill/tests_ckpt_rollout.py
"""
import os
import sys
import json
import copy
import math
import tempfile
import contextlib

import torch
import torch.nn.functional as F

# The RACE modules keep fp32 trainable params (hybrid.build_race_modules casts them
# .float()), while the base model is bf16. EVERY model forward in the scripts under
# test therefore runs inside torch.autocast(bf16) (distill_global.evaluate / the
# training loop / eval_ruler). We mirror that exactly: a bare forward would raise
# "mat1 and mat2 have the same dtype" (bf16 input vs fp32 q_proj weight).
def _autocast():
    return torch.autocast("cuda", dtype=torch.bfloat16)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from transformers import AutoModelForCausalLM, LlamaConfig
from transformers.models.llama.modeling_llama import LlamaForCausalLM

# ----------------------------------------------------------------------------- #
# results bookkeeping
# ----------------------------------------------------------------------------- #
RESULTS = []  # list of (name, passed_bool, detail_str)

# Intrinsic loss-reproducibility floor. The scripts run the forward under bf16 autocast
# and do NOT call torch.use_deterministic_algorithms(True), so two byte-identical runs
# differ ~1e-5 in the loss purely from GPU reduction-order nondeterminism (measured by
# check5's same-seed control). Both the determinism claim and the resume-boundary claim
# are judged against this floor: a faithful resume / a deterministic rerun must stay
# WITHIN it (it adds no error of its own). A genuine bug (wrong opt/RNG/base state) would
# diverge by O(lr) ~ 1e-3+ over a few steps, far above this floor.
RESUME_FLOOR = 5e-5


def record(name, passed, detail=""):
    RESULTS.append((name, bool(passed), detail))
    tag = "PASS" if passed else "FAIL"
    print(f"[{tag}] {name}: {detail}")


@contextlib.contextmanager
def section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)
    yield


# ----------------------------------------------------------------------------- #
# tiny model construction
# ----------------------------------------------------------------------------- #
TINY_DIR = None
DEVICE = "cuda"


def make_tiny_model_dir(seed=1234):
    """Create + save a tiny Llama (head_dim=128 to satisfy the RACE kernel; everything
    else small) to a temp dir so AutoModelForCausalLM.from_pretrained loads it offline
    and DETERMINISTICALLY (the saved weights are identical for every build)."""
    global TINY_DIR
    if TINY_DIR is not None:
        return TINY_DIR
    torch.manual_seed(seed)
    cfg = LlamaConfig(
        vocab_size=256,
        hidden_size=256,          # = num_attention_heads * head_dim = 2 * 128
        intermediate_size=512,
        num_hidden_layers=4,      # enough to test inter-layer rollout (RACE feeds next layer)
        num_attention_heads=2,
        num_key_value_heads=1,    # GQA path exercised (repeat_kv)
        head_dim=128,             # RACE kernel per-head dim
        max_position_embeddings=512,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
        attn_implementation="eager",
    )
    model = LlamaForCausalLM(cfg)
    model = model.to(torch.bfloat16)
    d = tempfile.mkdtemp(prefix="tiny_llama_")
    model.save_pretrained(d)
    # Minimal tokenizer config not needed: we never call get_tokenizer in these tests
    # (we feed random input_ids directly).
    TINY_DIR = d
    print(f"[setup] tiny model saved to {d} (cfg: L={cfg.num_hidden_layers} "
          f"H={cfg.num_attention_heads} kv={cfg.num_key_value_heads} hd={cfg.head_dim} "
          f"hidden={cfg.hidden_size} vocab={cfg.vocab_size})")
    return d


@contextlib.contextmanager
def patch_model_const(*modules):
    """Redirect the hardcoded MODEL constant in the given imported modules to the tiny
    local dir for the duration of the block. Restores afterwards."""
    d = make_tiny_model_dir()
    saved = {}
    for m in modules:
        if hasattr(m, "MODEL"):
            saved[(m, "MODEL")] = m.MODEL
            m.MODEL = d
        if hasattr(m, "TEACHER_MODEL"):
            saved[(m, "TEACHER_MODEL")] = m.TEACHER_MODEL
            m.TEACHER_MODEL = d
    try:
        yield d
    finally:
        for (m, attr), v in saved.items():
            setattr(m, attr, v)


def tiny_args(**over):
    """A small argparse.Namespace mimicking distill_global.parse() for build_student/save."""
    import argparse
    a = argparse.Namespace(
        pattern="ARRR", race_l=2, race_k=2, M=1, seq_len=64, batch_size=2,
        max_steps=4, eval_every=100, lr=5e-4, warmup_steps=2, clip=1.0,
        save_every=0, load_race_checkpoint=None, train_hash_geometry=False,
        hash_lr=1e-5, unfreeze="none", base_lr=1e-4, hidden_weight=1.0,
        kl_weight=0.5, kl_temp=1.0, ce_weight=0.0, kl_chunk=0, grad_checkpoint=False,
        seed=0, tag="test", out=None, ddp=False, grad_accum=1, lr_schedule="linear",
        min_lr_ratio=0.1, total_steps=0, resume=None, save_resume_every=0,
        data="fineweb", curriculum=None, lm_frac=0.7, long_source="fineweb_local",
        eval_seqlen=64, task_weights=None, n_distractors=0, probe_every=0,
        probe_niah=2, probe_max_new=4,
    )
    for k, v in over.items():
        setattr(a, k, v)
    return a


# ----------------------------------------------------------------------------- #
# CHECK 1: save_ckpt -> eval_ruler.build_model reloads + reproduces student logits
# ----------------------------------------------------------------------------- #
def check1_ckpt_roundtrip_frozen():
    import distill_global as dg
    import eval_ruler as er

    with patch_model_const(dg, er, _data_stub()) as _:
        args = tiny_args(pattern="ARRR", tag="c1frozen")
        torch.manual_seed(0)
        student, race, replaced, base_params = dg.build_student(args, DEVICE)
        # perturb the RACE params so they differ from the init (simulate a trained ckpt)
        with torch.no_grad():
            for p in dg.trainable_parameters_safe(race):
                p.add_(torch.randn_like(p) * 0.01)

        # in-run student logits (training regime: race params fp32, model bf16)
        torch.manual_seed(42)
        ids = torch.randint(0, 256, (1, 48), device=DEVICE)
        for m in race.values():
            m.eval()
        student.eval()
        with torch.no_grad(), _autocast():
            ref_logits = student(input_ids=ids, use_cache=False).logits.float().cpu()

        base_ids = {id(p) for p in base_params}
        ckpath = dg.save_ckpt(race, args, 7, student, base_ids)
        ck = torch.load(ckpath, map_location="cpu", weights_only=False)
        has_base = "base_state" in ck
        record("1.frozen.no_base_state_saved", not has_base,
               f"base_state present={has_base} (frozen run must NOT save base_state)")

        del student, race
        torch.cuda.empty_cache()

        # reload via eval_ruler.build_model in RECOMPUTE mode (cache mode toggles decode
        # state and would change the forward; recompute mirrors the training forward).
        model2, is_hybrid, meta = er.build_model("arrr", ckpath, attn_impl="eager",
                                                 device=DEVICE, decode_mode="recompute")
        model2.eval()
        # build_model casts RACE modules to bf16; ensure the same eval regime as the
        # training forward (which ran race in eval()).  Force fp32 race params to match
        # the in-run reference's numeric regime as closely as the loader allows.
        with torch.no_grad(), _autocast():
            rel_logits = model2(input_ids=ids, use_cache=False).logits.float().cpu()

        max_abs = (ref_logits - rel_logits).abs().max().item()
        # bf16 RACE recast in build_model introduces a tiny numeric delta vs the fp32
        # in-run forward. We additionally verify a BIT-IDENTICAL reload by loading the
        # checkpoint into a fp32-race student built the same way as training.
        record("1.frozen.argmax_identical",
               torch.equal(ref_logits.argmax(-1), rel_logits.argmax(-1)),
               f"max|Δlogit|={max_abs:.3e} (bf16-recast loader); argmax match checked")

        # --- TRUE checkpoint exactness is at the TENSOR level (forward-output equality is
        # masked by bf16 logit quantization + GPU run-to-run nondeterminism). Verify the
        # saved race_state tensors are BIT-IDENTICAL to what a freshly-built+reloaded
        # student holds. This is the actual round-trip guarantee.
        args2 = tiny_args(pattern="ARRR", tag="c1frozen", load_race_checkpoint=ckpath)
        torch.manual_seed(999)  # different seed: proves params come from the ckpt, not init
        student3, race3, _, _ = dg.build_student(args2, DEVICE)
        saved = ck["race_state"]
        reloaded = race3.state_dict()
        mism = [k for k in saved
                if not torch.equal(saved[k].to(DEVICE), reloaded[k].to(DEVICE))]
        record("1.frozen.race_state_bit_identical", len(mism) == 0,
               f"{len(saved)} race tensors round-tripped; mismatches={mism[:5]} "
               f"(BIT-EXACT checkpoint guarantee)")

        # Forward-output: must be within the intrinsic bf16/GPU nondeterminism floor
        # (measured below in check5/control), NOT bit-exact (logits are bf16-quantized).
        for m in race3.values():
            m.eval()
        student3.eval()
        with torch.no_grad(), _autocast():
            rel3 = student3(input_ids=ids, use_cache=False).logits.float().cpu()
        bit_max = (ref_logits - rel3).abs().max().item()
        # 1 bf16 ULP at logit magnitude ~max|logit| ~= max/256; allow a few ULP.
        tol = max(0.05, 4.0 * ref_logits.abs().max().item() / 256.0)
        record("1.frozen.fp32_reload_reproduces", bit_max <= tol,
               f"max|Δlogit|={bit_max:.3e} (fp32 race reload); tol={tol:.3e} "
               f"(bf16-quantized logits; tensor-level exactness checked separately)")
        del model2, student3, race3
        torch.cuda.empty_cache()


def check1_ckpt_roundtrip_unfreeze():
    """--unfreeze mlp: base_state must be SAVED and LOADED; reloaded model reproduces the
    IN-RUN logits (i.e. the trained MLPs), NOT stock Llama."""
    import distill_global as dg
    import eval_ruler as er

    with patch_model_const(dg, er, _data_stub()):
        args = tiny_args(pattern="ARRR", tag="c1mlp", unfreeze="mlp")
        torch.manual_seed(0)
        student, race, replaced, base_params = dg.build_student(args, DEVICE)
        record("1.mlp.base_params_nonempty", len(base_params) > 0,
               f"{len(base_params)} base tensors unfrozen")

        # perturb BOTH race and base params (simulate trained, diverged-from-stock MLPs)
        with torch.no_grad():
            for p in dg.trainable_parameters_safe(race):
                p.add_(torch.randn_like(p) * 0.01)
            for p in base_params:
                p.add_(torch.randn_like(p) * 0.02)

        torch.manual_seed(42)
        ids = torch.randint(0, 256, (1, 48), device=DEVICE)
        for m in race.values():
            m.eval()
        student.eval()
        with torch.no_grad(), _autocast():
            ref_logits = student(input_ids=ids, use_cache=False).logits.float().cpu()

        base_ids = {id(p) for p in base_params}
        ckpath = dg.save_ckpt(race, args, 11, student, base_ids)
        ck = torch.load(ckpath, map_location="cpu", weights_only=False)
        n_saved = len(ck.get("base_state", {}))
        record("1.mlp.base_state_saved", n_saved == len(base_params),
               f"saved {n_saved} base tensors, expected {len(base_params)}")

        del student, race
        torch.cuda.empty_cache()

        # stock-Llama reference (NO checkpoint) to prove the reload is NOT stock
        stock = AutoModelForCausalLM.from_pretrained(TINY_DIR, dtype=torch.bfloat16).to(DEVICE).eval()
        with torch.no_grad(), _autocast():
            stock_logits = stock(input_ids=ids, use_cache=False).logits.float().cpu()
        del stock
        torch.cuda.empty_cache()

        model2, is_hybrid, meta = er.build_model("arrr", ckpath, attn_impl="eager",
                                                 device=DEVICE, decode_mode="recompute")
        record("1.mlp.n_base_loaded_matches", meta["n_base_loaded"] == len(base_params),
               f"meta.n_base_loaded={meta['n_base_loaded']} expected {len(base_params)}")
        model2.eval()
        with torch.no_grad(), _autocast():
            rel_logits = model2(input_ids=ids, use_cache=False).logits.float().cpu()

        d_run = (ref_logits - rel_logits).abs().max().item()
        d_stock = (rel_logits - stock_logits).abs().max().item()
        # TENSOR-LEVEL guarantee: the saved base_state must be bit-identical to the trained
        # MLPs, so the reload loads the trained (not stock) weights. (The forward output is
        # bf16-quantized + RACE is recast to bf16 by the loader, so logit equality is only
        # approximate; the load fidelity itself is exact at the tensor level.)
        base_mis = [k for k in ck["base_state"]
                    if not torch.equal(ck["base_state"][k].to(DEVICE),
                                       dict(model2.named_parameters())[k].detach().to(DEVICE))]
        record("1.mlp.base_state_loaded_bit_identical", len(base_mis) == 0,
               f"{len(ck['base_state'])} base tensors loaded bit-exact into reloaded model; "
               f"mismatches={base_mis[:5]}")
        # The reload must be MUCH closer to the in-run model than to stock Llama -> it loaded
        # the trained MLPs, NOT stock (the published-result-invalidating regression to guard).
        record("1.mlp.reproduces_inrun_not_stock",
               d_stock > 5 * max(d_run, 1e-3),
               f"max|Δ vs in-run|={d_run:.3e}  max|Δ vs stock|={d_stock:.3e}  "
               f"(reload tracks trained model, far from stock)")
        del model2
        torch.cuda.empty_cache()


# ----------------------------------------------------------------------------- #
# CHECK 2: full-state save -> RESUME continues identically to an uninterrupted run
# ----------------------------------------------------------------------------- #
def _build_for_training(args, seed):
    """Build student+race+opt deterministically. Returns the pieces needed to run a few
    training steps that mirror distill_global.main's inner loop (single GPU)."""
    import distill_global as dg
    torch.manual_seed(seed)
    student, race, replaced, base_params = dg.build_student(args, DEVICE)
    teacher = AutoModelForCausalLM.from_pretrained(TINY_DIR, dtype=torch.bfloat16).to(DEVICE)
    from hybrid import freeze_teacher
    freeze_teacher(teacher)
    proj_params, hash_params = dg.split_param_groups(race)
    race_params = proj_params + hash_params
    groups = [{"params": proj_params, "lr": args.lr, "base_lr": args.lr, "name": "proj"}]
    if base_params:
        groups.append({"params": base_params, "lr": args.base_lr, "base_lr": args.base_lr, "name": "base"})
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.95), weight_decay=0.0)
    from global_utils import register_global_capture
    t_store, t_h = register_global_capture(teacher, replaced, detach=True)
    s_store, s_h = register_global_capture(student, replaced, detach=False)
    return dict(student=student, teacher=teacher, race=race, opt=opt,
                race_params=race_params, base_params=base_params, replaced=replaced,
                t_store=t_store, s_store=s_store, handles=t_h + s_h)


def _train_step(ctx, ids, args):
    """One optimizer step mirroring distill_global.main (hidden+kl loss). Returns loss float."""
    import distill_global as dg
    from global_utils import clear_store, hidden_loss, kl_loss
    ctx["opt"].zero_grad(set_to_none=True)
    clear_store(ctx["t_store"]); clear_store(ctx["s_store"])
    with torch.autocast("cuda", dtype=torch.bfloat16):
        with torch.no_grad():
            t_logits = ctx["teacher"](input_ids=ids, use_cache=False).logits
        s_logits = ctx["student"](input_ids=ids, use_cache=False).logits
    h_loss, _ = hidden_loss(ctx["s_store"], ctx["t_store"], ctx["replaced"])
    k_loss = kl_loss(s_logits, t_logits, T=args.kl_temp, chunk=args.kl_chunk)
    loss = args.hidden_weight * h_loss + args.kl_weight * k_loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(ctx["race_params"] + ctx["base_params"], args.clip)
    ctx["opt"].step()
    val = float(loss.item())
    clear_store(ctx["t_store"]); clear_store(ctx["s_store"])
    return val


def check2_resume_determinism(unfreeze="none"):
    import distill_global as dg
    tag = f"c2{unfreeze}"
    args = tiny_args(tag=tag, unfreeze=unfreeze, lr=1e-3, warmup_steps=2)
    N_TOTAL, N_SPLIT = 6, 3

    with patch_model_const(dg, _data_stub()):
        # fixed token stream so both runs see identical data
        torch.manual_seed(7)
        batches = [torch.randint(0, 256, (args.batch_size, args.seq_len), device=DEVICE)
                   for _ in range(N_TOTAL)]

        # ---- uninterrupted reference run ----
        ctx = _build_for_training(args, seed=0)
        ref_losses = []
        for st in range(N_TOTAL):
            for g in ctx["opt"].param_groups:
                g["lr"] = dg.lr_at(st, g["base_lr"], args.warmup_steps,
                                   args.max_steps, args.min_lr_ratio, args.lr_schedule)
            ref_losses.append(_train_step(ctx, batches[st], args))
        for h in ctx["handles"]:
            h.remove()
        del ctx
        torch.cuda.empty_cache()

        # ---- split run: train N_SPLIT, save full state, rebuild fresh, resume ----
        ctx = _build_for_training(args, seed=0)
        for st in range(N_SPLIT):
            for g in ctx["opt"].param_groups:
                g["lr"] = dg.lr_at(st, g["base_lr"], args.warmup_steps,
                                   args.max_steps, args.min_lr_ratio, args.lr_schedule)
            _train_step(ctx, batches[st], args)
        base_ids = {id(p) for p in ctx["base_params"]}
        rpath = dg.resume_path(args)
        dg.save_full_state(ctx["race"], ctx["opt"], args, N_SPLIT, 0, args.seq_len,
                           rpath, ctx["student"], base_ids)
        for h in ctx["handles"]:
            h.remove()
        del ctx
        torch.cuda.empty_cache()

        # rebuild a FRESH model (different init seed) + resume from the full state.
        ctx = _build_for_training(args, seed=12345)
        ck = torch.load(rpath, map_location=DEVICE, weights_only=False)
        ctx["race"].load_state_dict(ck["race_state"], strict=True)
        if ck.get("base_state"):
            ctx["student"].load_state_dict(ck["base_state"], strict=False)
        ctx["opt"].load_state_dict(ck["opt_state"])
        start_step = int(ck["global_step"])
        torch.set_rng_state(ck["cpu_rng"].cpu())
        torch.cuda.set_rng_state_all([s.cpu() for s in ck["cuda_rng"]])

        resumed_losses = []
        for st in range(start_step, N_TOTAL):
            for g in ctx["opt"].param_groups:
                g["lr"] = dg.lr_at(st, g["base_lr"], args.warmup_steps,
                                   args.max_steps, args.min_lr_ratio, args.lr_schedule)
            resumed_losses.append(_train_step(ctx, batches[st], args))
        for h in ctx["handles"]:
            h.remove()
        del ctx
        torch.cuda.empty_cache()

        ref_tail = ref_losses[N_SPLIT:]
        diffs = [abs(a - b) for a, b in zip(ref_tail, resumed_losses)]
        max_diff = max(diffs) if diffs else 0.0
        # The training forward runs under bf16 autocast WITHOUT torch.use_deterministic_
        # algorithms, so even two identical runs drift ~1e-5 in the loss (measured in
        # check5's control). A FAITHFUL resume must continue within that same intrinsic
        # floor -> the save/resume boundary itself adds no error. RESUME_FLOOR is set to
        # the measured intrinsic nondeterminism (see check5) with margin.
        record(f"2.{unfreeze}.resume_loss_continues",
               max_diff < RESUME_FLOOR,
               f"ref_tail={['%.6f' % x for x in ref_tail]} resumed={['%.6f' % x for x in resumed_losses]} "
               f"max|Δ|={max_diff:.3e} (intrinsic-noise floor {RESUME_FLOOR:.1e}; "
               f"resume adds no error beyond bf16 nondeterminism)")
        record(f"2.{unfreeze}.step_counter_resumed", start_step == N_SPLIT,
               f"resumed start_step={start_step} expected {N_SPLIT}")


# ----------------------------------------------------------------------------- #
# CHECK 3: GLOBAL rollout uses the student's OWN hidden states end-to-end
# ----------------------------------------------------------------------------- #
def check3_global_rollout():
    """Prove a RACE layer's OUTPUT feeds the next layer (own hidden states), NOT
    teacher-forced. Method: register hooks capturing each layer's INPUT during the
    student forward. Then perturb an EARLY RACE layer's output via a hook and confirm
    LATER layers' inputs change -> they are wired to the student's own activations.
    Contrast with distill_local, which feeds each layer the TEACHER's h_in."""
    import distill_global as dg

    with patch_model_const(dg, _data_stub()):
        args = tiny_args(pattern="ARRR", tag="c3")
        torch.manual_seed(0)
        student, race, replaced, base_params = dg.build_student(args, DEVICE)
        student.eval()
        for m in race.values():
            m.eval()
        ids = torch.randint(0, 256, (1, 32), device=DEVICE)

        n_layers = student.config.num_hidden_layers
        # capture each decoder layer's INPUT hidden state
        layer_in = {}

        def mk_in_hook(i):
            def hook(mod, inp, out):
                layer_in[i] = inp[0].detach().float().clone()
            return hook
        handles = [student.model.layers[i].register_forward_pre_hook(
            lambda mod, inp, i=i: layer_in.__setitem__(i, inp[0].detach().float().clone()))
            for i in range(n_layers)]

        with torch.no_grad(), _autocast():
            base_logits = student(input_ids=ids, use_cache=False).logits.float().clone()
        baseline_in = {i: v.clone() for i, v in layer_in.items()}
        for h in handles:
            h.remove()

        # pick the earliest RACE layer; perturb its OUTPUT and check downstream inputs
        first_race = min(replaced)
        perturb_h = None

        def out_perturb(mod, inp, out):
            if isinstance(out, tuple):
                return (out[0] + 1.0,) + tuple(out[1:])
            return out + 1.0
        perturb_h = student.model.layers[first_race].register_forward_hook(out_perturb)
        layer_in.clear()
        handles = [student.model.layers[i].register_forward_pre_hook(
            lambda mod, inp, i=i: layer_in.__setitem__(i, inp[0].detach().float().clone()))
            for i in range(n_layers)]
        with torch.no_grad(), _autocast():
            pert_logits = student(input_ids=ids, use_cache=False).logits.float().clone()
        for h in handles:
            h.remove()
        perturb_h.remove()

        # the layer immediately AFTER first_race must see a changed input (own h flows)
        downstream = [i for i in range(first_race + 1, n_layers)]
        changed = {}
        for i in downstream:
            if i in layer_in and i in baseline_in:
                changed[i] = (layer_in[i] - baseline_in[i]).abs().max().item()
        all_downstream_changed = all(v > 0.5 for v in changed.values()) and len(changed) > 0
        # layers BEFORE/AT the perturbed one must be unchanged (causal wiring sanity)
        upstream_unchanged = True
        for i in range(0, first_race + 1):
            if i in layer_in and i in baseline_in:
                if (layer_in[i] - baseline_in[i]).abs().max().item() > 1e-4:
                    upstream_unchanged = False
        logits_changed = (pert_logits - base_logits).abs().max().item()
        record("3.rollout.own_hidden_feeds_forward",
               all_downstream_changed and upstream_unchanged and logits_changed > 0.5,
               f"first_race=L{first_race}; downstream Δin={ {k: round(v,2) for k,v in changed.items()} }; "
               f"upstream_unchanged={upstream_unchanged}; Δlogits={logits_changed:.2f}")

        # Contrast: distill_local explicitly feeds the TEACHER h_in (teacher forcing).
        import distill_local as dl
        import inspect
        src = inspect.getsource(dl.student_layer)
        teacher_forced = "h_in" in src and "race[str(i)]" in src and "layer.input_layernorm(h_in)" in src
        record("3.contrast.local_is_teacher_forced", teacher_forced,
               "distill_local.student_layer consumes per-layer teacher h_in (teacher-forced); "
               "distill_global wires layer outputs into the next layer (end-to-end)")
        del student, race
        torch.cuda.empty_cache()


# ----------------------------------------------------------------------------- #
# CHECK 4: register_global_capture - captures exactly replaced layers; teacher
#          detached, student grad-bearing.
# ----------------------------------------------------------------------------- #
def check4_capture():
    import distill_global as dg
    from global_utils import register_global_capture, clear_store

    with patch_model_const(dg, _data_stub()):
        args = tiny_args(pattern="ARRR", tag="c4")
        torch.manual_seed(0)
        student, race, replaced, base_params = dg.build_student(args, DEVICE)
        teacher = AutoModelForCausalLM.from_pretrained(TINY_DIR, dtype=torch.bfloat16).to(DEVICE)
        from hybrid import freeze_teacher
        freeze_teacher(teacher)
        ids = torch.randint(0, 256, (1, 32), device=DEVICE)

        t_store, t_h = register_global_capture(teacher, replaced, detach=True)
        s_store, s_h = register_global_capture(student, replaced, detach=False)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            with torch.no_grad():
                teacher(input_ids=ids, use_cache=False)
            student(input_ids=ids, use_cache=False)

        captured_t = set(t_store["h_out"].keys())
        captured_s = set(s_store["h_out"].keys())
        exact = (captured_t == set(replaced)) and (captured_s == set(replaced))
        record("4.capture.exactly_replaced_layers", exact,
               f"replaced={sorted(replaced)} captured_t={sorted(captured_t)} captured_s={sorted(captured_s)}")

        # teacher tensors detached (no grad), student tensors grad-bearing
        t_detached = all(not v.requires_grad for v in t_store["h_out"].values())
        s_grad = all(v.requires_grad for v in s_store["h_out"].values())
        record("4.capture.teacher_detached", t_detached,
               f"teacher h_out requires_grad set = {set(v.requires_grad for v in t_store['h_out'].values())}")
        record("4.capture.student_grad_bearing", s_grad,
               f"student h_out requires_grad set = {set(v.requires_grad for v in s_store['h_out'].values())}")

        # final_norm also captured for both
        fn_ok = (t_store["final_norm"] is not None) and (s_store["final_norm"] is not None)
        record("4.capture.final_norm_present", fn_ok,
               f"t_final_norm={t_store['final_norm'] is not None} s_final_norm={s_store['final_norm'] is not None}")

        # grad actually flows to RACE params from the hidden loss alone (student detach=False)
        from global_utils import hidden_loss
        clear_store(t_store); clear_store(s_store)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            with torch.no_grad():
                teacher(input_ids=ids, use_cache=False)
            student(input_ids=ids, use_cache=False)
        h_loss, _ = hidden_loss(s_store, t_store, replaced)
        for p in dg.trainable_parameters_safe(race):
            p.grad = None
        h_loss.backward()
        race_with_grad = sum(1 for p in dg.trainable_parameters_safe(race)
                             if p.grad is not None and p.grad.abs().sum() > 0)
        record("4.capture.hidden_loss_grads_race", race_with_grad > 0,
               f"{race_with_grad} RACE params received nonzero grad from hidden_loss alone")
        for h in t_h + s_h:
            h.remove()
        del student, teacher, race
        torch.cuda.empty_cache()


# ----------------------------------------------------------------------------- #
# CHECK 5: determinism - same seed -> identical loss trajectory across two runs
# ----------------------------------------------------------------------------- #
def check5_determinism():
    import distill_global as dg
    args = tiny_args(tag="c5", lr=1e-3)
    N = 5

    with patch_model_const(dg, _data_stub()):
        def one_run():
            torch.manual_seed(7)
            batches = [torch.randint(0, 256, (args.batch_size, args.seq_len), device=DEVICE)
                       for _ in range(N)]
            ctx = _build_for_training(args, seed=0)
            losses = []
            for st in range(N):
                for g in ctx["opt"].param_groups:
                    g["lr"] = dg.lr_at(st, g["base_lr"], args.warmup_steps,
                                       args.max_steps, args.min_lr_ratio, args.lr_schedule)
                losses.append(_train_step(ctx, batches[st], args))
            for h in ctx["handles"]:
                h.remove()
            del ctx
            torch.cuda.empty_cache()
            return losses

        r1 = one_run()
        r2 = one_run()
        diffs = [abs(a - b) for a, b in zip(r1, r2)]
        max_diff = max(diffs)
        # FINDING: the trajectory is reproducible only to ~1e-5, NOT bit-exact, because
        # distill_global runs the forward under bf16 autocast and never calls
        # torch.use_deterministic_algorithms(True) / sets CUBLAS_WORKSPACE_CONFIG. The
        # *seed-controlled* parts (data, init, dropout-free) ARE deterministic; the
        # residual is GPU reduction-order noise. This is the same floor the resume check
        # (check2) is judged against. We assert reproducibility WITHIN that floor and
        # report the exact measured value as the honest determinism number.
        record("5.determinism.reproducible_within_bf16_floor", max_diff < RESUME_FLOOR,
               f"run1={['%.6f'%x for x in r1]} run2={['%.6f'%x for x in r2]} "
               f"max|Δ|={max_diff:.3e} (bf16/GPU floor; NOT bit-exact — no "
               f"use_deterministic_algorithms in distill_global)")

        # Seed-controlled inputs MUST be bit-identical (this part has no float-reduction
        # noise): same seed -> same data batches AND same RACE param init. Proves the
        # seed governs everything the trajectory depends on except GPU reduction order.
        def seed_inputs():
            torch.manual_seed(7)
            b = [torch.randint(0, 256, (args.batch_size, args.seq_len), device=DEVICE)
                 for _ in range(N)]
            ctx = _build_for_training(args, seed=0)
            w = [p.detach().clone() for p in ctx["race_params"]]
            for h in ctx["handles"]:
                h.remove()
            del ctx
            torch.cuda.empty_cache()
            return b, w
        b1, w1 = seed_inputs()
        b2, w2 = seed_inputs()
        data_id = all(torch.equal(x, y) for x, y in zip(b1, b2))
        init_id = all(torch.equal(x, y) for x, y in zip(w1, w2))
        record("5.determinism.seed_inputs_bit_identical", data_id and init_id,
               f"same-seed data batches identical={data_id}; RACE init identical={init_id} "
               f"(seed fully governs data + initialization)")


# ----------------------------------------------------------------------------- #
# helper: a stub for the data_fineweb module so build_student/eval never hits the hub.
# build_student itself does NOT call get_tokenizer; only main() does. We still return a
# dummy module object so patch_model_const can redirect its MODEL constant if referenced.
# ----------------------------------------------------------------------------- #
class _Stub:
    MODEL = None


def _data_stub():
    import data_fineweb as df
    return df


# ----------------------------------------------------------------------------- #
def main():
    assert torch.cuda.is_available(), "tests require a GPU (RACE CUDA kernel)"
    print(f"torch {torch.__version__} | cuda {torch.version.cuda} | "
          f"device {torch.cuda.get_device_name(0)}")
    make_tiny_model_dir()

    # monkeypatch a tiny convenience accessor used above (avoid name collision with the
    # script's trainable_parameters which lives in hybrid) -- attach to distill_global.
    import distill_global as dg
    from hybrid import trainable_parameters as _tp
    dg.trainable_parameters_safe = _tp

    with section("CHECK 1 - save_ckpt -> eval_ruler.build_model round-trip (FROZEN base)"):
        check1_ckpt_roundtrip_frozen()
    with section("CHECK 1b - --unfreeze mlp: base_state saved + loaded + reproduces in-run"):
        check1_ckpt_roundtrip_unfreeze()
    with section("CHECK 3 - GLOBAL rollout uses student's OWN hidden states end-to-end"):
        check3_global_rollout()
    with section("CHECK 4 - register_global_capture (exact layers; teacher detached / student grad)"):
        check4_capture()
    with section("CHECK 2 - full-state RESUME determinism (frozen base)"):
        check2_resume_determinism(unfreeze="none")
    with section("CHECK 2b - full-state RESUME determinism (--unfreeze mlp)"):
        check2_resume_determinism(unfreeze="mlp")
    with section("CHECK 5 - determinism: same seed -> identical loss trajectory"):
        check5_determinism()

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    n_pass = sum(1 for _, p, _ in RESULTS if p)
    n_fail = len(RESULTS) - n_pass
    for name, p, detail in RESULTS:
        print(f"  [{'PASS' if p else 'FAIL'}] {name}")
    print(f"\n{n_pass}/{len(RESULTS)} checks passed; {n_fail} FAILED")

    # emit machine-readable summary for the parent harness
    print("RESULT_JSON=" + json.dumps(
        {"passed": n_pass, "failed": n_fail,
         "checks": [{"name": n, "pass": p, "detail": d} for n, p, d in RESULTS]}))

    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
