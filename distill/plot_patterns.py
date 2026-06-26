"""Compare full FA3 vs the two RACE-hybrid layouts (alternating 14/28 and
(Softmax,RACE,RACE,RACE) 21/28) on forward latency, plus the speedup curves.

Usage: plot_patterns.py <alt_csv> <srrr_csv>   (defaults to the FA3 CSVs in results/).
Writes results/fwd_latency_fa3_compare.png and prints the speedup tables.
"""
import os, sys, csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def load(p):
    s = {}
    for r in csv.DictReader(open(p)):
        s.setdefault(r["method"], {})[int(r["T"])] = float(r["ms"])
    return s


def main():
    alt_csv = sys.argv[1] if len(sys.argv) > 1 else os.path.join(RES, "fwd_latency_fa3.csv")
    srrr_csv = sys.argv[2] if len(sys.argv) > 2 else os.path.join(RES, "fwd_latency_fa3_srrr.csv")
    alt, srrr = load(alt_csv), load(srrr_csv)
    Ts = sorted(set(alt["full"]) & set(srrr["full"]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    series = [
        ("full (28 FA3)", alt["full"], "s-", "black"),
        ("alt hybrid_S8 (14 RACE)", alt["hybrid_S8"], "o--", "tab:orange"),
        ("alt hybrid_S24 (14 RACE)", alt["hybrid_S24"], "o:", "tab:red"),
        ("srrr hybrid_S8 (21 RACE)", srrr["hybrid_S8"], "^-", "tab:blue"),
        ("srrr hybrid_S24 (21 RACE)", srrr["hybrid_S24"], "^--", "tab:green"),
    ]
    for label, d, st, col in series:
        ax1.plot(Ts, [d.get(T, float("nan")) for T in Ts], st, label=label, color=col)
    ax1.set_xscale("log", base=2); ax1.set_yscale("log")
    ax1.set_xlabel("Sequence length T"); ax1.set_ylabel("Forward latency (ms), decoder stack")
    ax1.set_title("Llama-3.2-3B forward latency (H200, bf16, B=1, genuine FA3)")
    ax1.grid(True, which="both", ls="--", lw=0.5); ax1.legend(fontsize=8)

    # speedup vs full FA3
    full = alt["full"]
    for label, d, st, col in series[1:]:
        ax2.plot(Ts, [full[T] / d[T] for T in Ts], st, label=label, color=col)
    ax2.axhline(1.0, color="black", lw=0.8, ls="-")
    ax2.axhline(2.0, color="tab:orange", lw=0.8, ls=":", label="alt ceiling 28/14=2x")
    ax2.axhline(4.0, color="tab:blue", lw=0.8, ls=":", label="srrr ceiling 28/7=4x")
    ax2.set_xscale("log", base=2)
    ax2.set_xlabel("Sequence length T"); ax2.set_ylabel("Speedup vs full FA3 (x)")
    ax2.set_title("Speedup vs full FA3  (>1 = faster)")
    ax2.grid(True, which="both", ls="--", lw=0.5); ax2.legend(fontsize=8)

    plt.tight_layout()
    out = os.path.join(RES, "fwd_latency_fa3_compare.png")
    plt.savefig(out, dpi=160); plt.close()
    print("wrote", out)

    # tables
    def row(name, d):
        return name + " | " + " | ".join(f"{full[T]/d[T]:5.2f}" for T in Ts)
    hdr = "speedup vs full | " + " | ".join(f"{T//1024}K" if T < 1<<20 else "1M" for T in Ts)
    print("\n" + hdr)
    print(row("alt  S8 ", alt["hybrid_S8"]))
    print(row("srrr S8 ", srrr["hybrid_S8"]))
    print(row("alt  S24", alt["hybrid_S24"]))
    print(row("srrr S24", srrr["hybrid_S24"]))
    print("\nsrrr-vs-alt ratio (how much faster srrr is than alt):")
    for s in ("hybrid_S8", "hybrid_S24"):
        print(f"  {s}: " + " | ".join(f"{alt[s][T]/srrr[s][T]:4.2f}" for T in Ts))


if __name__ == "__main__":
    main()
