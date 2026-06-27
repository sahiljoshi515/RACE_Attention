"""Plots for the GLOBAL hybrid-student distillation run.

Reads results/metrics_global_<tag>.jsonl and writes 5 PNGs into results/ (filename
prefix defaults to global_<tag>, override with --out-prefix):
  <prefix>_eval_trends.png        eval MSE/cos/KL/agreement/ppl/CE vs step
  <prefix>_per_layer_cosine.png   final per-layer hidden cosine (+ 50% ALT pilot overlay)
  <prefix>_loss_breakdown.png     train total/hidden/kl/ce vs step
  <prefix>_layer_groups.png       early/mid/late group cosine + L1/2/3 vs step
  <prefix>_compare_previous.png   this run vs prior runs (--compare <tags...>)

Usage:
  python distill/plot_global.py --tag arrr_1k_ce --out-prefix arrr_1k_ce --compare arrr ar
"""
import os
import sys
import json
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# 50% alternating LOCAL pilot final per-layer hidden cosine (metrics_S8.jsonl, step 99).
ALT_PILOT = {1: 0.764, 3: 0.892, 5: 0.914, 7: 0.904, 9: 0.927, 11: 0.929, 13: 0.940,
             15: 0.966, 17: 0.975, 19: 0.970, 21: 0.977, 23: 0.980, 25: 0.971, 27: 0.963}


def load(tag):
    path = os.path.join(RES, f"metrics_global_{tag}.jsonl")
    trains, evals = [], []
    with open(path) as f:
        for ln in f:
            r = json.loads(ln)
            (evals if ("eval" in r) else trains).append(r)
    return trains, evals


def eval_trends(evals, prefix, label):
    steps = [e["step"] for e in evals]
    g = lambda k: [e["eval"].get(k, float("nan")) for e in evals]
    fig, ax = plt.subplots(2, 3, figsize=(18, 9))
    panels = [
        (ax[0, 0], "rel hidden MSE (mean)", [("rel_hidden_mse", g("mean_rel_hidden_mse"))], True),
        (ax[0, 1], "hidden cosine", [("mean layer", g("mean_hidden_cos")), ("final norm", g("final_norm_cos")),
                                     ("min layer", g("min_layer_cos"))], False),
        (ax[0, 2], "logits KL", [("kl", g("kl"))], True),
        (ax[1, 0], "top-k agreement", [("top1", g("top1_agree")), ("top5", g("top5_agree"))], False),
        (ax[1, 1], "perplexity", [("student", g("student_ppl")), ("teacher", g("teacher_ppl"))], True),
        (ax[1, 2], "eval CE / total loss", [("student CE", g("student_ce")), ("eval total", g("eval_total_loss"))], False),
    ]
    for a, title, series, logy in panels:
        for lbl, ys in series:
            a.plot(steps, ys, "o-", label=lbl, ms=3)
        a.set_title(title); a.set_xlabel("step"); a.grid(True, ls="--", lw=0.5)
        if logy:
            a.set_yscale("log")
        a.legend(fontsize=8)
    fig.suptitle(f"GLOBAL distill ({label}) — held-out eval trends")
    fig.tight_layout()
    out = os.path.join(RES, f"{prefix}_eval_trends.png")
    fig.savefig(out, dpi=150); plt.close(fig); print("wrote", out)


def per_layer_cosine(evals, prefix, label):
    last = evals[-1]["eval"]["per_layer"]
    layers = sorted(int(k) for k in last)
    cos = [last[str(l)]["hidden_cos"] for l in layers]
    fig, axx = plt.subplots(figsize=(11, 5.5))
    axx.plot(layers, cos, "o-", color="tab:blue", label=f"global {label} (final)")
    al = sorted(ALT_PILOT)
    axx.plot(al, [ALT_PILOT[l] for l in al], "s--", color="tab:gray", alpha=0.8,
             label="50% ALT local pilot (final)")
    axx.set_xlabel("layer index"); axx.set_ylabel("hidden-state cosine (student vs teacher)")
    axx.set_title(f"GLOBAL distill ({label}) — final per-layer hidden cosine")
    axx.grid(True, ls="--", lw=0.5); axx.legend()
    fig.tight_layout()
    out = os.path.join(RES, f"{prefix}_per_layer_cosine.png")
    fig.savefig(out, dpi=150); plt.close(fig); print("wrote", out)


def loss_breakdown(trains, prefix, label):
    steps = [t["step"] for t in trains]
    fig, axx = plt.subplots(figsize=(11, 5.5))
    for key, lbl in [("total_loss", "total"), ("hidden_loss", "hidden"),
                     ("kl_loss", "kl (raw)"), ("ce_loss", "ce")]:
        ys = [t.get(key, float("nan")) for t in trains]
        if key in ("total_loss", "hidden_loss", "kl_loss") or any(y == y and y != 0 for y in ys):
            axx.plot(steps, ys, "-", label=lbl, lw=1.0)
    axx.set_xlabel("step"); axx.set_ylabel("loss"); axx.set_yscale("log")
    axx.set_title(f"GLOBAL distill ({label}) — training loss breakdown")
    axx.grid(True, which="both", ls="--", lw=0.5); axx.legend()
    fig.tight_layout()
    out = os.path.join(RES, f"{prefix}_loss_breakdown.png")
    fig.savefig(out, dpi=150); plt.close(fig); print("wrote", out)


def layer_groups(evals, prefix, label):
    steps = [e["step"] for e in evals]
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))
    for grp, col in [("early", "tab:red"), ("middle", "tab:orange"), ("late", "tab:green")]:
        ys = [e["eval"].get("groups", {}).get(grp, {}).get("hidden_cos", float("nan")) for e in evals]
        ax[0].plot(steps, ys, "o-", color=col, label=grp, ms=3)
    ax[0].plot(steps, [e["eval"].get("min_layer_cos", float("nan")) for e in evals], "k:", label="min layer")
    ax[0].set_title("hidden cosine by layer group (early/mid/late) + min")
    ax[0].set_xlabel("step"); ax[0].set_ylabel("hidden cosine"); ax[0].grid(True, ls="--", lw=0.5); ax[0].legend()
    for li, col in [("1", "tab:purple"), ("2", "tab:brown"), ("3", "tab:cyan")]:
        ys = [e["eval"].get("early_layers_123", {}).get(li, float("nan")) for e in evals]
        ax[1].plot(steps, ys, "o-", color=col, label=f"layer {li}", ms=3)
    ax[1].set_title("earliest RACE layers 1/2/3 (compounding probe)")
    ax[1].set_xlabel("step"); ax[1].set_ylabel("hidden cosine"); ax[1].grid(True, ls="--", lw=0.5); ax[1].legend()
    fig.suptitle(f"GLOBAL distill ({label}) — layer-group diagnostics")
    fig.tight_layout()
    out = os.path.join(RES, f"{prefix}_layer_groups.png")
    fig.savefig(out, dpi=150); plt.close(fig); print("wrote", out)


def grad_norms(trains, evals, prefix, label):
    steps = [t["step"] for t in trains]
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))
    for key, lbl in [("grad_norm", "total"), ("proj_grad_norm", "proj"), ("hash_grad_norm", "hash")]:
        ys = [t.get(key, float("nan")) for t in trains]
        if any(y == y for y in ys):
            ax[0].plot(steps, ys, "-", lw=1.0, label=lbl)
    ax[0].set_title("grad norms (pre-clip)"); ax[0].set_xlabel("step"); ax[0].set_ylabel("grad L2")
    ax[0].set_yscale("log"); ax[0].grid(True, which="both", ls="--", lw=0.5); ax[0].legend()
    for key, lbl in [("lr_proj", "lr proj"), ("lr_hash", "lr hash")]:
        ys = [t.get(key, float("nan")) for t in trains]
        if any(y == y and y > 0 for y in ys):
            ax[1].plot(steps, ys, "-", lw=1.0, label=lbl)
    ax[1].set_yscale("log"); ax[1].set_xlabel("step"); ax[1].set_ylabel("learning rate")
    ax[1].grid(True, which="both", ls="--", lw=0.5)
    if evals:
        es = [e["step"] for e in evals]
        hd = [e["eval"].get("hash_drift", float("nan")) for e in evals]
        if any(y == y and y > 0 for y in hd):
            axd = ax[1].twinx(); axd.plot(es, hd, "k:", label="hash drift")
            axd.set_ylabel("hash drift (L2 from init)"); axd.legend(loc="lower right", fontsize=8)
    ax[1].set_title("learning rates + hash-geometry drift"); ax[1].legend(loc="center right", fontsize=8)
    fig.suptitle(f"GLOBAL distill ({label}) — grad norms & hash-geometry learning")
    fig.tight_layout()
    out = os.path.join(RES, f"{prefix}_grad_norms.png")
    fig.savefig(out, dpi=150); plt.close(fig); print("wrote", out)


def compare_previous(this_evals, this_label, prev_tags, prefix):
    series = [(this_label, this_evals)]
    for t in prev_tags:
        try:
            series.append((t, load(t)[1]))
        except FileNotFoundError:
            print(f"(compare: metrics_global_{t}.jsonl not found, skipping)")
    fig, ax = plt.subplots(2, 2, figsize=(14, 9))
    panels = [(ax[0, 0], "student perplexity", "student_ppl", True),
              (ax[0, 1], "logits KL", "kl", True),
              (ax[1, 0], "mean hidden cosine", "mean_hidden_cos", False),
              (ax[1, 1], "final-norm hidden cosine", "final_norm_cos", False)]
    for a, title, key, logy in panels:
        for lbl, ev in series:
            a.plot([e["step"] for e in ev], [e["eval"].get(key, float("nan")) for e in ev], "o-", ms=3, label=lbl)
        a.set_title(title); a.set_xlabel("step"); a.grid(True, ls="--", lw=0.5)
        if logy:
            a.set_yscale("log")
        a.legend(fontsize=8)
    fig.suptitle(f"Global distill comparison: {this_label} vs {', '.join(prev_tags)}")
    fig.tight_layout()
    out = os.path.join(RES, f"{prefix}_compare_previous.png")
    fig.savefig(out, dpi=150); plt.close(fig); print("wrote", out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="arrr")
    p.add_argument("--out-prefix", default=None, help="filename prefix (default global_<tag>)")
    p.add_argument("--compare", nargs="*", default=[], help="prior tags to overlay in the comparison plot")
    args = p.parse_args()
    prefix = args.out_prefix or f"global_{args.tag}"
    trains, evals = load(args.tag)
    if not evals:
        print("no eval records found; skipping eval-based plots")
    else:
        eval_trends(evals, prefix, args.tag)
        per_layer_cosine(evals, prefix, args.tag)
        layer_groups(evals, prefix, args.tag)
        if args.compare:
            compare_previous(evals, args.tag, args.compare, prefix)
    if trains:
        loss_breakdown(trains, prefix, args.tag)
        grad_norms(trains, evals, prefix, args.tag)


if __name__ == "__main__":
    main()
