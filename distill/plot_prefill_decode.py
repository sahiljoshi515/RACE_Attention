"""Plot prefill + decode latency (full vs RACE-hybrid) from prefill_decode_latency.csv.
Left: prefill_ms vs T. Right: decode ms/token vs T. log-x. Also writes a markdown table."""
import os, csv, argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=os.path.join(RES, "prefill_decode_latency.csv"))
    p.add_argument("--out", default=os.path.join(RES, "prefill_decode_latency.png"))
    args = p.parse_args()
    rows = list(csv.DictReader(open(args.csv)))
    methods = []
    for r in rows:
        if r["method"] not in methods:
            methods.append(r["method"])
    Ts = sorted({int(r["T"]) for r in rows})
    colors = {m: c for m, c in zip(methods, plt.cm.tab10.colors)}

    def series(method, key):
        d = {int(r["T"]): float(r[key]) for r in rows if r["method"] == method and r[key] != "nan"}
        xs = [t for t in Ts if t in d]
        return xs, [d[t] for t in xs]

    fig, (axp, axd) = plt.subplots(1, 2, figsize=(13, 5))
    for m in methods:
        x, y = series(m, "prefill_ms")
        axp.plot([t / 1024 for t in x], y, "o-", label=m, color=colors[m])
        x, y = series(m, "decode_ms_per_tok")
        axd.plot([t / 1024 for t in x], y, "o-", label=m, color=colors[m])
    for ax, title, yl in [(axp, "Prefill latency (FA3 softmax + RACE)", "prefill (ms)"),
                          (axd, "Decode latency / token", "ms / token")]:
        ax.set_xscale("log", base=2); ax.set_xlabel("context length (K tokens)")
        ax.set_ylabel(yl); ax.set_title(title); ax.grid(True, alpha=0.3); ax.legend()
        ax.set_xticks([t / 1024 for t in Ts]); ax.set_xticklabels([f"{t//1024}K" for t in Ts])
    fig.suptitle("Full Llama-3.2-3B (all FA3) vs RACE-hybrid (FA3 softmax + RACE decode kernel), B=1 bf16, H200")
    fig.tight_layout(); fig.savefig(args.out, dpi=120); plt.close(fig)
    print(f"wrote {args.out}")

    # markdown table + speedups vs full
    full = {int(r["T"]): r for r in rows if r["method"] == "full"}
    lines = ["| method | T | prefill ms | decode ms/tok | mem GB | prefill× vs full | decode× vs full |",
             "|---|--:|--:|--:|--:|--:|--:|"]
    for m in methods:
        for r in [r for r in rows if r["method"] == m]:
            T = int(r["T"]); fp = full.get(T)
            psp = f"{float(fp['prefill_ms'])/float(r['prefill_ms']):.2f}×" if fp and r["prefill_ms"] != "nan" else "-"
            dsp = f"{float(fp['decode_ms_per_tok'])/float(r['decode_ms_per_tok']):.2f}×" if fp and r["decode_ms_per_tok"] != "nan" else "-"
            lines.append(f"| {m} | {T//1024}K | {r['prefill_ms']} | {r['decode_ms_per_tok']} | "
                         f"{r['mem_gb']} | {psp} | {dsp} |")
    md = "\n".join(lines)
    open(os.path.join(RES, "prefill_decode_latency.md"), "w").write(
        "# Prefill + decode latency: full (FA3) vs RACE-hybrid (FA3 + RACE kernel)\n\n" + md + "\n")
    print("\n" + md)


if __name__ == "__main__":
    main()
