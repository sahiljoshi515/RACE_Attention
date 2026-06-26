"""Hooks, losses, and metrics for GLOBAL hybrid-student distillation.

Unlike the local pilot (distill_local.py), the student runs END-TO-END on its own
hidden states (RACE outputs feed later layers). We capture each model's raw
per-decoder-layer outputs via forward hooks and match student -> frozen teacher.
NO teacher forcing anywhere in here.

Pure functions; no module-level state. Loss terms keep the student grad graph;
all logged metrics are detached.
"""
import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# capture (forward hooks)                                                      #
# --------------------------------------------------------------------------- #
def register_global_capture(model, replaced, detach):
    """Forward-hook each decoder layer in `replaced` to store its RAW output
    (out[0] if tuple else out). We hook the decoder layer's raw return value and
    never request output_hidden_states (whose last entry would be tied to the
    post-final-norm last_hidden_state), so this is the clean pre-norm hidden.
    Also hooks the final norm for the post-norm hidden.

    detach=True  -> frozen teacher (store detached tensors, cheap).
    detach=False -> student (store the GRAPH tensors so the hidden loss backprops).

    NOTE: under gradient checkpointing (use_reentrant=False) the layer forward is
    re-run during backward, so these hooks RE-FIRE and overwrite the store with
    recompute-time tensors. That is harmless here because the store is read to build
    the loss BEFORE backward and cleared after; captures are only valid pre-backward.

    Returns (store, handles). store = {"h_out": {i: tensor}, "final_norm": tensor}.
    """
    store = {"h_out": {}, "final_norm": None}
    handles = []
    for i in replaced:
        def layer_hook(mod, inp, out, i=i):
            h = out[0] if isinstance(out, tuple) else out
            store["h_out"][i] = h.detach() if detach else h
        handles.append(model.model.layers[i].register_forward_hook(layer_hook))

    def norm_hook(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        store["final_norm"] = h.detach() if detach else h
    handles.append(model.model.norm.register_forward_hook(norm_hook))
    return store, handles


def clear_store(store):
    """Drop references to captured tensors. Call at the START of each step (so a
    previous step's tensors can't be read) and AFTER backward (so the student's
    graph tensors are released and don't pin memory across steps)."""
    store["h_out"].clear()
    store["final_norm"] = None


# --------------------------------------------------------------------------- #
# losses (keep grad to the student)                                           #
# --------------------------------------------------------------------------- #
def hidden_loss(s_store, t_store, replaced):
    """mean over `replaced` of MSE(student_h_out_i, teacher_h_out_i) in fp32.
    Returns (loss_tensor, per_layer_metrics) where per-layer metrics are detached."""
    total = None
    per = {}
    for i in replaced:
        s = s_store["h_out"][i].float()
        t = t_store["h_out"][i].float()
        mse = F.mse_loss(s, t)
        total = mse if total is None else total + mse
        with torch.no_grad():
            per[i] = {
                "hidden_mse": mse.item(),
                "rel_hidden_mse": (mse / (t.pow(2).mean() + 1e-8)).item(),
                "hidden_cos": F.cosine_similarity(s.detach(), t, dim=-1).mean().item(),
            }
    return total / len(replaced), per


def kl_loss(student_logits, teacher_logits, T=1.0, chunk=0):
    """T^2 * KL(softmax(teacher/T) || student) with batchmean over (B*T) rows.

    Matches: F.kl_div(log_softmax(student/T), softmax(teacher/T), 'batchmean')*T^2.
    Teacher is detached; everything upcast to fp32. If chunk>0 the (rows,V) fp32
    softmax intermediates are computed in row-chunks of `chunk` (sum then /rows ==
    batchmean) to cap the logit-upcast memory spike. Grad flows to the student only.
    """
    V = student_logits.size(-1)
    s = student_logits.reshape(-1, V)
    t = teacher_logits.detach().reshape(-1, V)
    N = s.size(0)
    scale = T * T
    if chunk and chunk > 0 and N > chunk:
        total = s.new_zeros((), dtype=torch.float32)
        for a in range(0, N, chunk):
            b = min(a + chunk, N)
            log_s = F.log_softmax(s[a:b].float() / T, dim=-1)
            p_t = F.softmax(t[a:b].float() / T, dim=-1)
            total = total + F.kl_div(log_s, p_t, reduction="sum")
        return total / N * scale
    log_s = F.log_softmax(s.float() / T, dim=-1)
    p_t = F.softmax(t.float() / T, dim=-1)
    return F.kl_div(log_s, p_t, reduction="batchmean") * scale


def ce_loss(logits, input_ids):
    """Causal next-token cross-entropy (fp32). Used for the optional CE term and
    for perplexity (ppl = exp(ce))."""
    sl = logits[:, :-1].float().reshape(-1, logits.size(-1))
    lbl = input_ids[:, 1:].reshape(-1)
    return F.cross_entropy(sl, lbl)


# --------------------------------------------------------------------------- #
# metrics (all detached / no_grad)                                            #
# --------------------------------------------------------------------------- #
def layer_groups(replaced):
    """Split the replaced layer indices into 3 contiguous thirds (early/mid/late)."""
    layers = sorted(replaced)
    n = len(layers)
    a, b = n // 3, 2 * n // 3
    return {"early": layers[:a], "middle": layers[a:b], "late": layers[b:]}


@torch.no_grad()
def topk_agreement(student_logits, teacher_logits, k=5):
    """(top1 argmax agreement, fraction of student-top1 inside teacher-top-k)."""
    s1 = student_logits.argmax(-1)
    t1 = teacher_logits.argmax(-1)
    top1 = (s1 == t1).float().mean().item()
    tk = teacher_logits.topk(k, dim=-1).indices
    in_tk = (tk == s1.unsqueeze(-1)).any(-1).float().mean().item()
    return top1, in_tk


@torch.no_grad()
def eval_metrics(teacher_logits, student_logits, t_store, s_store, replaced,
                 input_ids, kl_temp=1.0, topk=5):
    """Full held-out eval metric bundle (all detached)."""
    per = {}
    for i in replaced:
        s = s_store["h_out"][i].float()
        t = t_store["h_out"][i].float()
        mse = F.mse_loss(s, t)
        per[i] = {
            "hidden_mse": mse.item(),
            "rel_hidden_mse": (mse / (t.pow(2).mean() + 1e-8)).item(),
            "hidden_cos": F.cosine_similarity(s, t, dim=-1).mean().item(),
        }
    layers = sorted(replaced)
    mean = {k: sum(per[i][k] for i in layers) / len(layers)
            for k in ("hidden_mse", "rel_hidden_mse", "hidden_cos")}

    fn_s, fn_t = s_store["final_norm"].float(), t_store["final_norm"].float()
    final_norm_cos = F.cosine_similarity(fn_s, fn_t, dim=-1).mean().item()

    # log the RAW distributional KL (T=1.0), independent of the training kl_temp
    kl = kl_loss(student_logits, teacher_logits, T=1.0, chunk=0).item()
    top1, top5 = topk_agreement(student_logits, teacher_logits, k=topk)
    s_ce = ce_loss(student_logits, input_ids).item()
    t_ce = ce_loss(teacher_logits, input_ids).item()

    groups = {}
    for g, idxs in layer_groups(replaced).items():
        if idxs:
            groups[g] = {
                "hidden_cos": sum(per[i]["hidden_cos"] for i in idxs) / len(idxs),
                "rel_hidden_mse": sum(per[i]["rel_hidden_mse"] for i in idxs) / len(idxs),
            }
    early3 = {str(i): per[i]["hidden_cos"] for i in (1, 2, 3) if i in per}
    worst = min(layers, key=lambda i: per[i]["hidden_cos"])

    return {
        "mean_hidden_mse": mean["hidden_mse"],
        "mean_rel_hidden_mse": mean["rel_hidden_mse"],
        "mean_hidden_cos": mean["hidden_cos"],
        "final_norm_cos": final_norm_cos,
        "kl": kl,
        "top1_agree": top1, "top5_agree": top5,
        "student_ppl": float(torch.exp(torch.tensor(s_ce))),
        "teacher_ppl": float(torch.exp(torch.tensor(t_ce))),
        "student_ce": s_ce, "teacher_ce": t_ce,
        "groups": groups,
        "early_layers_123": early3,
        "min_layer_cos": per[worst]["hidden_cos"], "min_layer": worst,
        "per_layer": {str(i): per[i] for i in layers},
    }


# --------------------------------------------------------------------------- #
# health / integrity checks                                                   #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def grad_health(race_params):
    """(fraction of RACE params with nonzero grad, all-finite bool)."""
    nz, finite = 0, True
    for p in race_params:
        if p.grad is not None:
            if p.grad.abs().sum() > 0:
                nz += 1
            if not torch.isfinite(p.grad).all():
                finite = False
    return nz / max(1, len(race_params)), finite


@torch.no_grad()
def base_grad_count(model, race_param_ids):
    """Number of NON-RACE params that received a grad (must be 0 -- base is frozen)."""
    return sum(1 for p in model.parameters()
               if id(p) not in race_param_ids and p.grad is not None
               and p.grad.abs().sum() > 0)


@torch.no_grad()
def fingerprint(model, race_param_ids=frozenset()):
    """Coarse global drift number: sum of abs over all FROZEN (non-RACE) params.
    For LOGGING only -- use snapshot_base/assert_base_unchanged for the hard
    guarantee (abs-sum can mask small per-param changes)."""
    total = 0.0
    for p in model.parameters():
        if id(p) not in race_param_ids and not p.requires_grad:
            total += p.detach().float().abs().sum().item()
    return total


@torch.no_grad()
def snapshot_base(model, race_param_ids=frozenset(), n=8):
    """Clone the first n FROZEN (non-RACE) param tensors (deterministic order) for an
    EXACT start-vs-end unchanged check -- stronger than the abs-sum fingerprint."""
    snap = {}
    for name, p in model.named_parameters():
        if id(p) not in race_param_ids and not p.requires_grad:
            snap[name] = p.detach().clone()
            if len(snap) >= n:
                break
    return snap


@torch.no_grad()
def assert_base_unchanged(model, snap):
    """Assert every snapshotted frozen param is bit-identical now (drift == 0)."""
    cur = dict(model.named_parameters())
    for name, p0 in snap.items():
        assert torch.equal(cur[name].detach(), p0), f"frozen base param changed: {name}"
