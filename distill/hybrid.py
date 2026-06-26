"""Layer-replacement + freezing utilities for the hybrid RACE/Llama model.

- build_race_modules: create RaceLlamaAttention for the replaced layers, copying
  q/k/v/o from the teacher (params cast to fp32 for stable AdamW; compute runs
  bf16 under autocast).
- convert_to_hybrid: swap them into the model in place (for inference / global mode).
- freeze_teacher / trainable_parameters: enforce the frozen/trainable partition.
"""
import torch
import torch.nn as nn

from race_llama_attention import RaceLlamaAttention


def odd_layers(i):
    """Alternating: RACE on every odd layer (1,3,..,27) -> 14/28 RACE, 14 softmax."""
    return i % 2 == 1


def srrr_layers(i):
    """(Softmax, RACE, RACE, RACE) repeating: softmax only every 4th layer (i%4==0),
    RACE on the other three. For 28 layers -> 7 softmax (0,4,..,24), 21 RACE."""
    return i % 4 != 0


def pattern_pred(pattern):
    """Predicate factory for a repeating A/R layer pattern. 'A' = keep the original
    Llama softmax attention, 'R' = replace with RACE. The string repeats over the
    layer index. Examples (28-layer Llama):
        'AR'   -> RACE on odd layers           (14/28, == odd_layers / 'alt')
        'AAR'  -> RACE every 3rd layer         (~9/28)
        'ARRR' -> (Softmax,RACE,RACE,RACE)x7   (21/28, A kept at 0,4,..,24)
    Returns f(i) -> bool, True == replace layer i with RACE."""
    pat = pattern.upper().strip()
    if not pat or (set(pat) - {"A", "R"}) or "R" not in pat:
        raise ValueError(f"pattern must be a non-empty 'A'/'R' string containing 'R', got {pattern!r}")
    return lambda i: pat[i % len(pat)] == "R"


def keep_edges_pred(n_layers, period=4, n_edge=2):
    """Predicate that keeps FULL softmax attention at the first/last `n_edge` layers and
    every `period`-th layer, replacing the rest with RACE. Concentrates exact attention
    where retrieval heads tend to live (early + late) for a better RULER shot. Returns
    f(i)->bool, True == replace layer i with RACE. Layer 0 is always kept softmax (the
    cache-mode decode in eval_ruler.py requires pred(0)==False)."""
    def pred(i):
        keep_full = (i < n_edge) or (i >= n_layers - n_edge) or (i % period == 0)
        return not keep_full
    assert not pred(0), "keep_edges_pred must keep layer 0 softmax (n_edge>=1)"
    return pred


def make_replace_pred(pattern, n_layers):
    """Resolve a pattern spec into a replace_pred(i)->bool, used by BOTH training
    (distill_global.build_student) and eval (eval_ruler.build_model) so the replaced-layer
    set is reconstructed identically from a checkpoint's config['pattern'].
      * 'edges[:P[:E]]' -> keep_edges_pred(n_layers, period=P (default 4), n_edge=E (default 2))
      * any A/R string  -> pattern_pred(pattern)
    Filenames sanitize ':' so a stored pattern like 'edges:4' round-trips."""
    pat = str(pattern).strip().lower()
    if pat.startswith("edges"):
        parts = pat.split(":")
        period = int(parts[1]) if len(parts) > 1 and parts[1] else 4
        n_edge = int(parts[2]) if len(parts) > 2 and parts[2] else 2
        return keep_edges_pred(n_layers, period=period, n_edge=n_edge)
    return pattern_pred(pattern)


def build_race_modules(model, L=2, Kbits=2, M=1, device="cuda",
                       replace_pred=odd_layers, seed=0):
    """Returns ModuleDict {str(layer_idx): RaceLlamaAttention} for replaced layers,
    initialized from the teacher's projections (fp32 params)."""
    cfg = model.config
    race = nn.ModuleDict()
    for i, layer in enumerate(model.model.layers):
        if replace_pred(i):
            mod = RaceLlamaAttention(cfg, i, L=L, Kbits=Kbits, M=M, device=device, seed=seed)
            mod.copy_projections_from(layer.self_attn)
            mod = mod.to(device).float()      # fp32 trainable params
            race[str(i)] = mod
    return race


def convert_to_hybrid(model, race_modules):
    """Swap RACE modules into the model in place (inference / global-mode forward)."""
    for k, mod in race_modules.items():
        model.model.layers[int(k)].self_attn = mod
    return model


def freeze_teacher(model):
    model.requires_grad_(False)
    model.eval()
    return model


def set_trainable_race(race_modules):
    """All RACE params (q/k/v/o + log_temp) trainable; planes/protos are buffers."""
    for mod in race_modules.values():
        mod.requires_grad_(True)
    return race_modules


def set_trainable_base(model, spec, race_param_ids):
    """Partial base-unfreeze for distillation. `spec='mlp'` re-enables requires_grad on the MLP
    blocks (gate/up/down) + the layer norms across ALL layers; embeddings, lm_head, and ALL
    attention q/k/v/o (the retrieval heads we preserve) stay frozen. `spec='none'` is a no-op.
    Never touches RACE params (race_param_ids). Returns the list of newly-trainable params
    (for a dedicated low-LR optimizer group + the checkpoint base_state)."""
    if not spec or spec == "none":
        return []
    if spec != "mlp":
        raise ValueError(f"unsupported --unfreeze spec {spec!r} (supported: 'none', 'mlp')")
    chosen = []
    for name, p in model.named_parameters():
        if id(p) in race_param_ids:
            continue
        if "self_attn" in name or "embed" in name or "lm_head" in name:
            continue                                   # keep attention + embeddings frozen
        if ".mlp." in name or "layernorm" in name or name.endswith("model.norm.weight"):
            p.requires_grad_(True)
            chosen.append(p)
    return chosen


def trainable_parameters(race_modules):
    params = []
    for mod in race_modules.values():
        params += [p for p in mod.parameters() if p.requires_grad]
    return params


def count_params(modules):
    return sum(p.numel() for m in modules.values() for p in m.parameters() if p.requires_grad)
