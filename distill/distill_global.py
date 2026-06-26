"""GLOBAL hybrid-student distillation of Llama-3.2-3B into a RACE-attention hybrid.

Difference from the local pilot (distill_local.py): the student runs END-TO-END on
its OWN hidden states. Two model instances are loaded -- a frozen TEACHER and a
hybrid STUDENT whose attention layers at the pattern-selected indices are replaced
by RaceLlamaAttention. Each step:
  1. teacher(input_ids) under no_grad           -> teacher logits + per-layer h_out
  2. student(input_ids) under autocast bf16     -> student logits + per-layer h_out
  3. loss = hidden_weight * mean_layer MSE(student_h_out, teacher_h_out)
          + kl_weight     * T^2 * KL(softmax(teacher/T) || student)
          + ce_weight     * next-token CE                 (CE off by default)
Only the RACE q/k/v/o + log_temp train (custom CUDA race_backward). Everything else
is frozen. Tests whether a high-replacement (e.g. ARRR = 21/28) hybrid is globally
trainable. NO teacher forcing.

Run:
  python distill/distill_global.py --pattern ARRR --race-l 2 --race-k 2 \
      --seq-len 4096 --batch-size 2 --max-steps 200 --eval-every 10 \
      --lr 5e-5 --warmup-steps 20 --grad-checkpoint --kl-chunk 2048 --tag arrr
"""
import os
import sys
import json
import time
import math
import signal
import argparse
import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transformers import AutoModelForCausalLM                                 # noqa: E402
from data_fineweb import get_tokenizer, make_eval_and_train                   # noqa: E402
from hybrid import (build_race_modules, convert_to_hybrid, freeze_teacher,    # noqa: E402
                    set_trainable_race, set_trainable_base, trainable_parameters, count_params,
                    pattern_pred, make_replace_pred)
from global_utils import (register_global_capture, clear_store, hidden_loss,  # noqa: E402
                          kl_loss, ce_loss, eval_metrics, grad_health,
                          base_grad_count, fingerprint, snapshot_base, assert_base_unchanged)

MODEL = "meta-llama/Llama-3.2-3B-Instruct"
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")


def resolve_ckpt(path):
    """Resolve a checkpoint path tolerantly: as-given, else by basename under the
    distill/checkpoints dir, else relative to the distill dir. Lets the task's
    'checkpoints/<f>.pt' find files actually saved in distill/checkpoints/."""
    here = os.path.dirname(os.path.abspath(__file__))
    for c in (path, os.path.join(CKPT, os.path.basename(path)), os.path.join(here, path)):
        if os.path.exists(c):
            return c
    raise FileNotFoundError(f"checkpoint not found; tried: {[path, os.path.join(CKPT, os.path.basename(path)), os.path.join(here, path)]}")


def _sanitize(s):
    """Filename-safe pattern tag ('edges:4' -> 'edges4')."""
    return str(s).lower().replace(":", "").replace("/", "")


def _base_state(student, base_ids):
    """The trained (partially-unfrozen) base params, keyed by name, for the checkpoint.
    Empty dict when nothing is unfrozen (frozen-base runs stay byte-identical to before)."""
    if not student or not base_ids:
        return {}
    return {n: p.detach() for n, p in student.named_parameters() if id(p) in base_ids}


def save_ckpt(race, args, step_done, student=None, base_ids=None):
    """Save the trainable RACE params (race.state_dict() also carries the small fixed
    plane/proto buffers for exact reload). Filename includes pattern/L/K/tag/step. This is
    the EVAL-compatible weights+config checkpoint that eval_ruler.build_model reads. When the
    base is partially unfrozen, also saves `base_state` (the trained MLPs) so eval / the next
    stage don't silently revert them to stock Llama."""
    os.makedirs(CKPT, exist_ok=True)
    path = os.path.join(CKPT, f"{_sanitize(args.pattern)}_L{args.race_l}_K{args.race_k}_{args.tag}_step{step_done}.pt")
    payload = {"race_state": race.state_dict(), "step": step_done,
               "config": {"pattern": args.pattern, "L": args.race_l, "K": args.race_k,
                          "M": args.M, "tag": args.tag, "unfreeze": getattr(args, "unfreeze", "none")}}
    bs = _base_state(student, base_ids)
    if bs:
        payload["base_state"] = bs
    torch.save(payload, path)
    return path


def resume_path(args):
    """Fixed full-state checkpoint path for auto-resume across 24h job restarts (overwritten,
    not per-step). Distinct from the numbered eval checkpoints written by save_ckpt."""
    return os.path.join(CKPT, f"{_sanitize(args.pattern)}_{args.tag}_resume.pt")


def save_full_state(race, opt, args, global_step, tokens_done, curr_seqlen, path,
                    student=None, base_ids=None):
    """Atomically write the FULL training state needed to resume an interrupted run:
    RACE weights, AdamW moments, step, tokens, curriculum length, RNG, args, and (when the base
    is partially unfrozen) the trained base_state. Atomic via write-to-tmp + os.replace so a job
    killed mid-write never corrupts the resume file."""
    os.makedirs(CKPT, exist_ok=True)
    payload = {
        "race_state": race.state_dict(),
        "opt_state": opt.state_dict(),
        "global_step": global_step,
        "tokens_done": tokens_done,
        "curr_seqlen": curr_seqlen,
        "cpu_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all(),
        "args": vars(args),
        "config": {"pattern": args.pattern, "L": args.race_l, "K": args.race_k,
                   "M": args.M, "tag": args.tag},
    }
    bs = _base_state(student, base_ids)
    if bs:
        payload["base_state"] = bs
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)
    return path


def lr_at(step, base, warmup, total, min_ratio, schedule):
    """Per-step LR. 'linear' = warmup ramp then hold at base (legacy behaviour). 'cosine' =
    warmup then cosine decay to base*min_ratio over `total` steps."""
    if step < warmup:
        return base * (step + 1) / max(1, warmup)
    if schedule != "cosine" or total <= warmup:
        return base
    prog = min(1.0, (step - warmup) / max(1, total - warmup))
    min_lr = base * min_ratio
    return min_lr + 0.5 * (base - min_lr) * (1 + math.cos(math.pi * prog))


def parse_curriculum(spec):
    """'4096:0,8192:2000,16384:4000' (len:start_step) -> sorted [(start_step, seq_len)]."""
    stages = []
    for tok in spec.split(","):
        L, start = tok.split(":")
        stages.append((int(start), int(L)))
    return sorted(stages)


# ---- distributed + preemption ------------------------------------------------
_PREEMPTED = {"flag": False}


def _install_signal_handlers():
    """Trap SLURM's time-limit / preemption signals so the loop can flush a resume
    checkpoint before the 24h kill. SLURM sends SIGTERM (and SIGUSR1 with --signal)."""
    def handler(signum, frame):
        _PREEMPTED["flag"] = True
    for sig in (signal.SIGTERM, signal.SIGUSR1):
        try:
            signal.signal(sig, handler)
        except Exception:
            pass


def setup_distributed(args):
    """Init torch.distributed from torchrun env when --ddp and WORLD_SIZE>1. Returns
    (rank, local_rank, world_size, is_dist). Single-GPU path returns (0,0,1,False) and is
    byte-identical to the pre-DDP trainer."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if not args.ddp or world_size == 1:
        return 0, 0, 1, False
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size, True


def allreduce_grads(params, world_size):
    """Average gradients of the trainable RACE params across DDP ranks. Manual (not
    nn.parallel.DDP) because the RACE modules are invoked indirectly inside the student
    forward (convert_to_hybrid), so DDP's autograd hooks wouldn't fire correctly."""
    for p in params:
        if p.grad is not None:
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
            p.grad /= world_size


HASH_NAMES = ("planes_T", "protos_T")


def split_param_groups(race):
    """Return (proj_params, hash_params): RACE q/k/v/o + log_temp vs the hash-geometry
    params (planes_T/protos_T, only present/trainable after make_hash_trainable)."""
    proj, hashp = [], []
    for m in race.values():
        for n, p in m.named_parameters():
            if p.requires_grad:
                (hashp if n in HASH_NAMES else proj).append(p)
    return proj, hashp


def grad_norm(params):
    """L2 norm over a param list's grads (0.0 if empty / no grads)."""
    sq = 0.0
    for p in params:
        if p.grad is not None:
            sq += float(p.grad.detach().float().norm()) ** 2
    return sq ** 0.5


def build_student(args, device):
    """Second model instance -> hybrid student. Build order matters: freeze the
    whole model FIRST (which also disables the swapped-in RACE), THEN re-enable the
    RACE params; load a prior checkpoint (continue, not restart); then optionally
    convert the hash geometry to trainable Parameters."""
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(device)
    pred = make_replace_pred(args.pattern, model.config.num_hidden_layers)
    race = build_race_modules(model, L=args.race_l, Kbits=args.race_k, M=args.M,
                              device=device, replace_pred=pred, seed=args.seed)
    convert_to_hybrid(model, race)          # swap RACE into model.model.layers[i].self_attn
    freeze_teacher(model)                    # requires_grad_(False) on EVERYTHING + eval()
    set_trainable_race(race)                 # re-enable RACE params (must be AFTER freeze)
    replaced = sorted(int(k) for k in race.keys())

    if args.load_race_checkpoint:            # continue from prior trained RACE (planes still buffers here)
        ckpath = resolve_ckpt(args.load_race_checkpoint)
        ck = torch.load(ckpath, map_location=device, weights_only=False)
        race.load_state_dict(ck["race_state"], strict=True)
        if ck.get("base_state"):             # carry trained (unfrozen) base weights across stages
            model.load_state_dict(ck["base_state"], strict=False)
            print(f"loaded base_state ({len(ck['base_state'])} tensors) from checkpoint")
        print(f"loaded RACE checkpoint {ckpath} (saved step {ck.get('step')})")
    if args.train_hash_geometry:             # buffers -> trainable Parameters (AFTER load)
        for m in race.values():
            m.make_hash_trainable()
        print(f"hash geometry is TRAINABLE (planes_T/protos_T -> nn.Parameter), hash_lr={args.hash_lr}")

    race_ids = {id(p) for p in trainable_parameters(race)}
    base_params = set_trainable_base(model, args.unfreeze, race_ids)   # partial unfreeze (e.g. 'mlp')
    if base_params:
        n_base = sum(p.numel() for p in base_params)
        print(f"partial unfreeze '{args.unfreeze}': {len(base_params)} tensors / {n_base/1e6:.0f}M "
              f"base params trainable @ base_lr={args.base_lr}")
    allowed = race_ids | {id(p) for p in base_params}
    leaked = [n for n, p in model.named_parameters() if p.requires_grad and id(p) not in allowed]
    assert not leaked, f"unexpected trainable params: {leaked[:5]}"
    assert count_params(race) > 0, "no trainable RACE params!"

    model.train()        # train mode (Llama has no dropout); required for checkpointing to fire
    if args.grad_checkpoint:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model, race, replaced, base_params


def evaluate(teacher, student, race, replaced, eval_batch, t_store, s_store, args):
    for m in race.values():
        m.eval()
    clear_store(t_store); clear_store(s_store)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        t_logits = teacher(input_ids=eval_batch, use_cache=False).logits
        s_logits = student(input_ids=eval_batch, use_cache=False).logits
        ev = eval_metrics(t_logits, s_logits, t_store, s_store, replaced,
                          eval_batch, kl_temp=args.kl_temp)
        # eval_total uses the KL at the TRAINING temperature (objective-consistent for any
        # kl_temp); ev["kl"] stays the raw T=1 distributional metric for logging.
        kl_train = kl_loss(s_logits, t_logits, T=args.kl_temp, chunk=0).item()
        ev["eval_total_loss"] = (args.hidden_weight * ev["mean_hidden_mse"]
                                 + args.kl_weight * kl_train
                                 + args.ce_weight * ev["student_ce"])
    for m in race.values():
        m.train()
    clear_store(t_store); clear_store(s_store)
    del t_logits, s_logits
    return ev


def decollapse_probe(student, race, niah_samples, prompts, tok, max_new=16):
    """In-loop free-generation collapse + NIAH-4K retrieval probe (rank-0 only) — THE honest
    de-collapse signal. Teacher-forced ppl looked fine (37) while the model was fully collapsed;
    this measures what actually matters. Uses the use_cache=False RECOMPUTE decode path so it
    never enables the RACE decode-cache and never conflicts with grad-checkpointing. Strictly
    read-only w.r.t. training state: a finally-block restores RACE train() mode, forces
    decode-cache OFF + resets it, and re-enables grad-checkpointing — leaving the next training
    forward byte-identical (race_llama_attention requires decode-capture OFF for that)."""
    import ablation_score as _abl
    import eval_ruler as _er
    device = next(student.parameters()).device
    gc_on = bool(getattr(student, "is_gradient_checkpointing", False))
    if gc_on:
        student.gradient_checkpointing_disable()
    for m in race.values():
        m.eval()
    eos_ids = {tok.eos_token_id}
    for t in getattr(tok, "additional_special_tokens_ids", []) or []:
        eos_ids.add(t)
    try:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            collapsed = 0
            for p in prompts:
                msg = [{"role": "user", "content": p}]
                s = tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
                ids = tok(s, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
                gen = _abl.greedy_decode(student, ids, 64, eos_ids, use_cache=False)
                if not gen or _er._is_looping(gen):
                    collapsed += 1
            coll = collapsed / max(1, len(prompts))
            acc, _n = _abl.niah_accuracy(student, tok, niah_samples, use_cache=False)
        return {"collapsed_frac": round(coll, 3), "niah4k_acc": round(acc, 3)}
    finally:
        for m in race.values():
            if hasattr(m, "enable_decode_cache"):
                m.enable_decode_cache(False)
            if hasattr(m, "reset_decode_state"):
                m.reset_decode_state()
            m.train()
        if gc_on:
            student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--pattern", default="ARRR", help="A/R repeating pattern (A=softmax, R=RACE)")
    p.add_argument("--race-l", type=int, default=2)
    p.add_argument("--race-k", type=int, default=2)
    p.add_argument("--M", type=int, default=1)
    p.add_argument("--seq-len", type=int, default=4096)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--warmup-steps", type=int, default=20)
    p.add_argument("--grad-clip", "--clip", dest="clip", type=float, default=1.0, help="grad-norm clip (0=off)")
    p.add_argument("--save-every", type=int, default=0, help="checkpoint RACE params every N steps (0=off)")
    p.add_argument("--load-race-checkpoint", default=None,
                   help="continue from a prior RACE checkpoint (.pt with race_state); weights-only "
                        "(AdamW state/step NOT resumed -> warmup restarts); does NOT restart base")
    p.add_argument("--train-hash-geometry", action="store_true",
                   help="also train the RACE hash geometry (planes_T/protos_T -> nn.Parameter)")
    p.add_argument("--hash-lr", type=float, default=1e-5, help="lr for the hash-geometry param group")
    p.add_argument("--unfreeze", default="none", choices=["none", "mlp"],
                   help="partial base-unfreeze: 'mlp' trains the MLP blocks + norms (attention/embeds "
                        "stay frozen) to recover MMLU; 'none' = RACE-only (default, drop-in).")
    p.add_argument("--base-lr", type=float, default=1e-5, help="lr for the unfrozen base param group")
    p.add_argument("--hidden-weight", type=float, default=1.0)
    p.add_argument("--kl-weight", type=float, default=0.5)
    p.add_argument("--kl-temp", type=float, default=1.0)
    p.add_argument("--ce-weight", type=float, default=0.0, help="0 disables the CE term")
    p.add_argument("--kl-chunk", type=int, default=2048, help="token-chunk for fp32 KL (0=off)")
    p.add_argument("--grad-checkpoint", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="arrr")
    p.add_argument("--out", default=None)
    # --- DDP / long-run additions ---
    p.add_argument("--ddp", action="store_true", help="enable torch.distributed (torchrun); off=single-GPU byte-identical")
    p.add_argument("--grad-accum", type=int, default=1, help="micro-batches accumulated per optimizer step")
    p.add_argument("--lr-schedule", choices=["linear", "cosine"], default="linear")
    p.add_argument("--min-lr-ratio", type=float, default=0.1, help="cosine floor = lr*min_lr_ratio")
    p.add_argument("--total-steps", type=int, default=0, help="cosine horizon (0 -> max_steps)")
    p.add_argument("--resume", default=None,
                   help="full-state resume: a path, or 'auto' to load the fixed resume file if present")
    p.add_argument("--save-resume-every", type=int, default=200, help="full-state checkpoint every N opt-steps")
    p.add_argument("--data", choices=["fineweb", "mixed"], default="fineweb",
                   help="train data source (curriculum overrides this)")
    p.add_argument("--curriculum", default=None,
                   help="len:start_step list, e.g. '4096:0,8192:2000,16384:4000,32768:6000,65536:8000'")
    p.add_argument("--lm-frac", type=float, default=0.7, help="LM:retrieval mix fraction for mixed/curriculum data")
    p.add_argument("--long-source", default="fineweb_local",
                   help="long-doc / filler text source for mixed/curriculum data: 'fineweb_local' "
                        "(pre-staged parquet, network-free, default), 'fineweb_edu' (hub), or 'pg19' "
                        "(BROKEN under datasets 4.x: script-based, no longer loads).")
    p.add_argument("--eval-seqlen", type=int, default=4096, help="length of the held-out ppl probe (curriculum/mixed)")
    p.add_argument("--task-weights", default=None,
                   help="oversample hard synthetic tasks, e.g. "
                        "'niah_single:4,variable_tracking:3,common_words:2' (SYNTH_TASKS names).")
    p.add_argument("--n-distractors", type=int, default=0,
                   help="same-pattern decoys injected into niah/vt synthetic tasks (matches RULER's "
                        "hard distractor variants; e.g. 24).")
    # --- in-loop de-collapse probe (P2): the HONEST generation signal ---
    p.add_argument("--probe-every", type=int, default=0,
                   help="run the in-loop free-gen collapse + NIAH-4K retrieval probe every N steps "
                        "(0=off). Teacher-forced ppl is NOT a de-collapse signal (ppl 37 coexisted "
                        "with full collapse); this is. Adds a few s per eval on rank 0.")
    p.add_argument("--probe-niah", type=int, default=8, help="# synthetic NIAH-4K samples for the probe")
    p.add_argument("--probe-max-new", type=int, default=16, help="tokens decoded per NIAH probe sample")
    return p.parse_args()


def main():
    args = parse()
    rank, local_rank, world_size, is_dist = setup_distributed(args)
    is_main = (rank == 0)
    _install_signal_handlers()
    device = f"cuda:{local_rank}" if is_dist else "cuda"
    if is_dist:
        torch.cuda.set_device(local_rank)
    torch.manual_seed(args.seed + rank)        # per-rank seed -> disjoint data shuffles
    if is_main:
        os.makedirs(RESULTS, exist_ok=True)
    out_path = args.out or os.path.join(RESULTS, f"metrics_global_{args.tag}.jsonl")

    accum = max(1, args.grad_accum)
    total_steps = args.total_steps or args.max_steps
    S = args.race_l * (1 << args.race_k)
    if is_main:
        print(f"GPU: {torch.cuda.get_device_name(local_rank if is_dist else 0)} | GLOBAL distill | "
              f"pattern={args.pattern} S={S} (L={args.race_l},K={args.race_k}) B={args.batch_size} "
              f"T={args.seq_len} steps={args.max_steps} accum={accum} world={world_size} "
              f"eff_tok/step={args.batch_size*args.seq_len*accum*world_size} lr={args.lr} "
              f"sched={args.lr_schedule} kl_w={args.kl_weight} ce_w={args.ce_weight} "
              f"data={'curriculum' if args.curriculum else args.data} lm_frac={args.lm_frac} "
              f"train_hash={args.train_hash_geometry} resume={args.resume}")

    tok = get_tokenizer()
    teacher = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(device)
    freeze_teacher(teacher)
    student, race, replaced, base_params = build_student(args, device)
    n_layers = teacher.config.num_hidden_layers
    kept = [i for i in range(n_layers) if i not in replaced]
    if is_main:
        print(f"pattern={args.pattern}: {len(replaced)}/{n_layers} RACE layers {replaced}")
        print(f"            softmax kept at {kept}")

    proj_params, hash_params = split_param_groups(race)
    race_params = proj_params + hash_params
    race_ids = {id(p) for p in race_params}
    base_ids = {id(p) for p in base_params}
    if is_main:
        print(f"trainable: {sum(p.numel() for p in proj_params)/1e6:.1f}M proj + "
              f"{sum(p.numel() for p in hash_params)} hash + "
              f"{sum(p.numel() for p in base_params)/1e6:.0f}M base params across {len(race)} layers")
    assert sum(p.requires_grad for p in teacher.parameters()) == 0, "teacher not frozen!"

    groups = [{"params": proj_params, "lr": args.lr, "base_lr": args.lr, "name": "proj"}]
    if hash_params:
        groups.append({"params": hash_params, "lr": args.hash_lr, "base_lr": args.hash_lr, "name": "hash"})
    if base_params:
        groups.append({"params": base_params, "lr": args.base_lr, "base_lr": args.base_lr, "name": "base"})
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.95), weight_decay=0.0)
    hash_init = [p.detach().clone() for p in hash_params]      # for hash-geometry drift
    t_store, t_handles = register_global_capture(teacher, replaced, detach=True)
    s_store, s_handles = register_global_capture(student, replaced, detach=False)

    fp0_t = fingerprint(teacher)
    fp0_s = fingerprint(student, race_ids)
    snap_t = snapshot_base(teacher)           # exact frozen-base snapshots (hard guarantee)
    snap_s = snapshot_base(student, race_ids)
    w0 = race_params[0].detach().clone()      # to confirm the optimizer actually moves a RACE weight

    # ---- data source: curriculum (ramped length) > mixed > fineweb ----
    curriculum = None
    max_micro = args.max_steps * accum + 8
    tw = ({t.split(":")[0]: float(t.split(":")[1]) for t in args.task_weights.split(",")}
          if args.task_weights else None)   # hard-task oversampling for mixed/curriculum data
    if args.curriculum:
        from data_long import make_curriculum, eval_probe_batch
        sched = parse_curriculum(args.curriculum)
        bs_fn = lambda L: max(1, round(args.batch_size * args.seq_len / L))   # longer seq -> smaller B
        curriculum = make_curriculum(tok, schedule=sched, batch_size_fn=bs_fn,
                                     lm_frac=args.lm_frac, seed=args.seed + rank,
                                     source=args.long_source, task_weights=tw,
                                     n_distractors=args.n_distractors)
        eval_batch = eval_probe_batch(tok, seq_length=args.eval_seqlen, seed=args.seed,
                                      source=args.long_source).to(device)
        train_iter = None
        if is_main:
            print(f"curriculum {sched}; eval probe T={args.eval_seqlen}")
    elif args.data == "mixed":
        from data_long import mixed_batches, eval_probe_batch
        train_iter = mixed_batches(tok, seq_length=args.seq_len, batch_size=args.batch_size,
                                   lm_frac=args.lm_frac, seed=args.seed + rank, max_batches=max_micro,
                                   source=args.long_source, task_weights=tw,
                                   n_distractors=args.n_distractors)
        eval_batch = eval_probe_batch(tok, seq_length=args.seq_len, seed=args.seed,
                                      source=args.long_source).to(device)
    else:
        eval_batches, train_gen = make_eval_and_train(
            tok, seq_length=args.seq_len, batch_size=args.batch_size,
            num_eval_batches=1, max_train_batches=max_micro, seed=args.seed + rank)
        eval_batch = eval_batches[0][:1].to(device)
        train_iter = train_gen
    if is_main:
        print(f"eval batch {tuple(eval_batch.shape)}; training...")

    # ---- in-loop de-collapse probe samples (rank-0 only, built once) ----
    probe_niah, probe_prompts = None, None
    if is_main and args.probe_every > 0:
        import ablation_score as _abl
        filler = _abl.build_filler_pool(tok, seed=0)
        probe_niah = _abl.make_niah_samples(tok, filler, n=args.probe_niah, seed=0)
        probe_prompts = _abl.COLLAPSE_PROMPTS
        print(f"[probe] every {args.probe_every} steps | {len(probe_niah)} NIAH-4K samples | "
              f"{len(probe_prompts)} collapse prompts")

    # ---- full-state resume (optimizer + step + tokens + RNG) ----
    start_step, tokens_done = 0, 0
    rpath = resume_path(args)
    do_resume = args.resume and (args.resume != "auto" or os.path.exists(rpath))
    if do_resume:
        load_path = rpath if args.resume == "auto" else resolve_ckpt(args.resume)
        ck = torch.load(load_path, map_location=device, weights_only=False)
        race.load_state_dict(ck["race_state"], strict=True)
        if ck.get("base_state"):
            student.load_state_dict(ck["base_state"], strict=False)
        opt.load_state_dict(ck["opt_state"])
        start_step = int(ck.get("global_step", 0))
        tokens_done = int(ck.get("tokens_done", 0))
        try:
            torch.set_rng_state(ck["cpu_rng"].cpu())
            torch.cuda.set_rng_state_all([s.cpu() for s in ck["cuda_rng"]])
        except Exception as e:
            if is_main:
                print(f"  (rng restore skipped: {e})")
        if is_main:
            print(f"RESUMED from {load_path} at step {start_step}, tokens {tokens_done}")

    logf = open(out_path, "a" if start_step else "w") if is_main else None

    def get_micro(ostep):
        return curriculum.batch_for_step(ostep) if curriculum is not None else next(train_iter)

    # ---- training loop with gradient accumulation ----
    layers = sorted(replaced)
    opt.zero_grad(set_to_none=True)
    step = start_step
    last_seqlen = args.seq_len
    while step < args.max_steps:
        for g in opt.param_groups:
            g["lr"] = lr_at(step, g["base_lr"], args.warmup_steps, total_steps,
                            args.min_lr_ratio, args.lr_schedule)
        lr_proj = opt.param_groups[0]["lr"]
        lr_hash = opt.param_groups[1]["lr"] if len(opt.param_groups) > 1 else 0.0
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize(); t0 = time.perf_counter()

        win = {"total": 0.0, "h": 0.0, "kl": 0.0, "ce": 0.0}
        micro_tokens = 0
        per = None
        for _m in range(accum):
            batch = get_micro(step).to(device)
            last_seqlen = batch.shape[1]
            clear_store(t_store); clear_store(s_store)
            # CE-only stages (RADLADS S3 context-extension: hidden_weight==kl_weight==0) don't
            # need the teacher -> skip its forward entirely (~2x faster + lower memory at long
            # context, where the full-softmax teacher is the bottleneck).
            need_teacher = (args.hidden_weight > 0 or args.kl_weight > 0)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if need_teacher:
                    with torch.no_grad():
                        t_logits = teacher(input_ids=batch, use_cache=False).logits
                s_logits = student(input_ids=batch, use_cache=False).logits
            if need_teacher:
                h_loss, per = hidden_loss(s_store, t_store, replaced)
                k_loss = kl_loss(s_logits, t_logits, T=args.kl_temp, chunk=args.kl_chunk)
                del t_logits
            else:
                z = s_logits.new_zeros(())
                h_loss, k_loss = z, z
                per = {i: {"hidden_mse": 0.0, "rel_hidden_mse": 0.0, "hidden_cos": 0.0}
                       for i in replaced}
            if args.ce_weight > 0:
                c_loss = ce_loss(s_logits, batch)               # in-graph
            else:
                with torch.no_grad():
                    c_loss = ce_loss(s_logits, batch)           # logging only
            loss = (args.hidden_weight * h_loss + args.kl_weight * k_loss
                    + args.ce_weight * c_loss) / accum
            loss.backward()
            win["total"] += float(loss.item())                  # already /accum -> sums to mean
            win["h"] += float(h_loss.item()) / accum
            win["kl"] += float(k_loss.item()) / accum
            win["ce"] += float(c_loss.item()) / accum
            micro_tokens += batch.numel()
            clear_store(t_store); clear_store(s_store)
            del s_logits, loss, h_loss, k_loss, c_loss

        # ---- optimizer-step boundary ----
        if is_dist:
            allreduce_grads(race_params + base_params, world_size)   # sync RACE *and* unfrozen base across ranks
        proj_gn = grad_norm(proj_params)
        hash_gn = grad_norm(hash_params)
        gnorm = torch.nn.utils.clip_grad_norm_(race_params + base_params, args.clip) if args.clip > 0 else torch.zeros(())
        rgf, rfin = grad_health(race_params)
        bgc = base_grad_count(student, race_ids)
        opt.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize(); dt = time.perf_counter() - t0
        tokens_done += micro_tokens

        mean = {k: sum(per[i][k] for i in layers) / len(layers)
                for k in ("hidden_mse", "rel_hidden_mse", "hidden_cos")}
        if is_main:
            rec = {
                "step": step, "type": "train", "seqlen": last_seqlen,
                "total_loss": win["total"], "hidden_loss": win["h"],
                "kl_loss": win["kl"], "ce_loss": win["ce"],
                "train_ppl_est": float(math.exp(min(win["ce"], 20.0))),
                "lr": lr_proj, "lr_proj": lr_proj, "lr_hash": lr_hash, "grad_norm": float(gnorm),
                "proj_grad_norm": proj_gn, "hash_grad_norm": hash_gn,
                "mean": mean, "per_layer": {str(i): per[i] for i in layers},
                "race_grad_nonzero_frac": rgf, "race_grad_finite": bool(rfin), "base_grad_count": bgc,
                "tokens": tokens_done, "tokens_global": tokens_done * world_size,
                "tok_s": micro_tokens / dt * world_size, "mem_gb": torch.cuda.max_memory_allocated() / 1e9,
            }
            logf.write(json.dumps(rec) + "\n"); logf.flush()
            print(f"step {step:4d} | loss {win['total']:.4f} (h {win['h']:.4f} kl {win['kl']:.4f} "
                  f"ce {win['ce']:.4f}) T={last_seqlen} gn p/h {proj_gn:.1f}/{hash_gn:.2f} | "
                  f"relHid {mean['rel_hidden_mse']:.3f} hidCos {mean['hidden_cos']:.3f} | "
                  f"bGrad {bgc} | {rec['tok_s']:.0f} tok/s | {rec['mem_gb']:.1f} GB")

        assert math.isfinite(win["total"]), f"non-finite loss at step {step}"
        if not base_params:                  # frozen-base run: guarantee zero base grads (unchanged behavior)
            assert bgc == 0, f"frozen base received {bgc} grads at step {step}"
        assert rfin, f"non-finite RACE grad at step {step}"
        if args.train_hash_geometry:
            assert math.isfinite(hash_gn) and (step > 0 or hash_gn > 0), \
                f"hash geometry grad bad at step {step}: {hash_gn}"

        if is_main and step % args.eval_every == 0:
            ev = evaluate(teacher, student, race, replaced, eval_batch, t_store, s_store, args)
            ev["hash_drift"] = (float(sum((p.detach() - p0).float().norm() ** 2
                                          for p, p0 in zip(hash_params, hash_init)) ** 0.5)
                                if hash_params else 0.0)
            if args.probe_every > 0 and step % args.probe_every == 0:
                try:
                    pr = decollapse_probe(student, race, probe_niah, probe_prompts, tok,
                                          max_new=args.probe_max_new)
                    ev.update(pr)
                except Exception as e:
                    print(f"   [probe@{step}] FAILED: {type(e).__name__}: {e}")
            logf.write(json.dumps({"step": step, "type": "eval", "eval": ev}) + "\n"); logf.flush()
            g = ev["groups"]
            probe_str = (f" | collapsed {ev['collapsed_frac']:.2f} niah4k {ev['niah4k_acc']:.3f}"
                         if "collapsed_frac" in ev else "")
            print(f"   [eval@{step}] total {ev['eval_total_loss']:.3f} | hidCos {ev['mean_hidden_cos']:.3f} "
                  f"finalCos {ev['final_norm_cos']:.3f} KL {ev['kl']:.3f} top1 {ev['top1_agree']:.3f} "
                  f"ppl S/T {ev['student_ppl']:.1f}/{ev['teacher_ppl']:.1f} | "
                  f"minCos {ev['min_layer_cos']:.2f}@L{ev['min_layer']} | hashDrift {ev['hash_drift']:.3f}{probe_str}")

        if is_main and args.save_every > 0 and (step + 1) % args.save_every == 0:
            print(f"   [ckpt] saved {save_ckpt(race, args, step + 1, student, base_ids)}")

        step += 1

        # periodic + on-preemption full-state checkpoint (rank0), synced across ranks
        want_save = (args.save_resume_every > 0 and step % args.save_resume_every == 0)
        preempt = _PREEMPTED["flag"]
        if is_dist:
            flag = torch.tensor([1 if preempt else 0], device=device)
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            preempt = bool(flag.item())
        if is_main and (want_save or preempt):
            p = save_full_state(race, opt, args, step, tokens_done, last_seqlen, rpath, student, base_ids)
            print(f"   [resume-ckpt] {'PREEMPTED ' if preempt else ''}saved {p} @step {step}")
        if preempt:
            if is_dist:
                dist.barrier()
            break

    # final integrity checks (all ranks; cheap)
    fp1_t = fingerprint(teacher)
    fp1_s = fingerprint(student, race_ids)
    moved = (race_params[0].detach() - w0).abs().sum().item()
    assert_base_unchanged(teacher, snap_t)
    assert_base_unchanged(student, snap_s)
    if is_main:
        print(f"frozen-base unchanged (exact snapshot OK); abs drift t={abs(fp1_t-fp0_t):.3e} "
              f"s={abs(fp1_s-fp0_s):.3e}; RACE w0 moved {moved:.3e}")
    assert moved > 0 or do_resume, "RACE weights never moved -- training did nothing!"

    for h in t_handles + s_handles:
        h.remove()
    completed = step >= args.max_steps
    if is_main:
        logf.close()
        if completed:
            # sentinel for the self-resubmitting sbatch: present => no requeue needed.
            open(os.path.join(CKPT, f"{_sanitize(args.pattern)}_{args.tag}.done"), "w").close()
        print(f"done ({'completed' if completed else 'preempted'}); wrote {out_path}")
    if is_dist:
        dist.destroy_process_group()
    # Skip interpreter finalization: the RACE C++ extension's static destructors crash on a
    # non-Python thread at shutdown (PyGILState_Release fatal). All outputs are flushed +
    # checkpoints written atomically above, so a hard exit here is clean and gives rc 0 (so
    # torchrun doesn't report ChildFailedError and the self-resubmit logic isn't misled).
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
