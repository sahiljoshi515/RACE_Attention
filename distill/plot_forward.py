"""Plot full vs hybrid forward latency and report the crossover T."""
import os, csv, sys, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def load(paths):
    series = {}   # method -> {T: ms}
    for p in paths:
        if not os.path.exists(p):
            continue
        for r in csv.DictReader(open(p)):
            series.setdefault(r["method"], {})[int(r["T"])] = float(r["ms"])
    return series


def main():
    paths = sys.argv[1:] or [os.path.join(RES, "fwd_latency.csv")]
    series = load(paths)
    Ts = sorted({T for m in series.values() for T in m})
    plt.figure(figsize=(9.5, 6))
    order = ["full"] + sorted(m for m in series if m != "full")
    styles = {"full": ("s-", "black")}
    for m in order:
        ys = [series[m].get(T, float("nan")) for T in Ts]
        st, col = styles.get(m, ("o-", None))
        plt.plot(Ts, ys, st, label=m, color=col)

    # crossover: smallest T where each hybrid < full
    cross = {}
    full = series.get("full", {})
    for m in series:
        if m == "full":
            continue
        for T in Ts:
            a, b = series[m].get(T, float("nan")), full.get(T, float("nan"))
            if a == a and b == b and a < b:
                cross[m] = T
                break
        else:
            cross[m] = None
        if cross[m]:
            plt.axvline(cross[m], ls=":", lw=0.8, alpha=0.5)

    plt.xscale("log", base=2); plt.yscale("log")
    plt.xlabel("Sequence length T"); plt.ylabel("Forward latency (ms) — decoder stack")
    plt.title("Llama-3.2-3B: full vs hybrid (14 RACE layers) forward latency (H200, bf16, B=1)")
    plt.grid(True, which="both", ls="--", lw=0.5); plt.legend()
    plt.tight_layout()
    out = os.path.join(RES, "fwd_latency.png"); plt.savefig(out, dpi=160); plt.close()
    print("wrote", out)
    print("crossover (hybrid < full):", json.dumps(cross))
    # speedup at the largest common finite T
    for m in cross:
        finiteT = [T for T in Ts if series[m].get(T, float("nan")) == series[m].get(T, float("nan"))
                   and full.get(T, float("nan")) == full.get(T, float("nan"))]
        if finiteT:
            T = finiteT[-1]
            print(f"  {m}: at T={T}, full={full[T]:.1f}ms hybrid={series[m][T]:.1f}ms "
                  f"speedup={full[T]/series[m][T]:.2f}x")


if __name__ == "__main__":
    main()
