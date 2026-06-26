"""Plot eval trends (held-out) and final per-layer cosine for the distill pilots."""
import os, json, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def load(path):
    evals, train = [], []
    for ln in open(path):
        d = json.loads(ln)
        (evals if "eval" in d else train).append(d)
    return evals, train


def main():
    configs = [(8, "S=8 (L=2,K=2)"), (24, "S=24 (L=3,K=3)")]
    # eval trends
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
    metr = [("rel_attn_mse", "rel attn MSE", True), ("rel_hidden_mse", "rel hidden MSE", True),
            ("attn_cos", "attn cosine", False), ("hidden_cos", "hidden cosine", False)]
    for S, label in configs:
        p = os.path.join(RES, f"metrics_S{S}.jsonl")
        if not os.path.exists(p):
            continue
        evals, _ = load(p)
        steps = [e["step"] for e in evals]
        for ax, (k, name, logy) in zip(axes, metr):
            ax.plot(steps, [e["eval"][k] for e in evals], "o-", label=label)
            ax.set_title(name); ax.set_xlabel("step"); ax.grid(True, ls="--", lw=0.5)
            if logy:
                ax.set_yscale("log")
            ax.legend()
    fig.suptitle("RACE distillation — held-out eval trends (teacher-forced local)")
    fig.tight_layout()
    out = os.path.join(RES, "eval_trends.png"); fig.savefig(out, dpi=150); plt.close(fig)
    print("wrote", out)

    # final per-layer cosine
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for S, label in configs:
        p = os.path.join(RES, f"metrics_S{S}.jsonl")
        if not os.path.exists(p):
            continue
        _, train = load(p)
        last = train[-1]["per_layer"]
        layers = sorted(int(k) for k in last)
        axes[0].plot(layers, [last[str(l)]["attn_cos"] for l in layers], "o-", label=label)
        axes[1].plot(layers, [last[str(l)]["hidden_cos"] for l in layers], "o-", label=label)
    axes[0].set_title("final attn cosine by layer"); axes[1].set_title("final hidden cosine by layer")
    for ax in axes:
        ax.set_xlabel("layer idx"); ax.grid(True, ls="--", lw=0.5); ax.legend()
    fig.tight_layout()
    out = os.path.join(RES, "final_per_layer_cosine.png"); fig.savefig(out, dpi=150); plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
