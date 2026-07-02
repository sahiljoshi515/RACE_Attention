"""Build/load the CPU RACE kernels (OpenMP), Linux + macOS.

Two independent extensions, each JIT-compiled once and cached:

  race_pref   -- RACE causal prefix-mean building block
      race_prefix_mean_flat(probsK[NS,T], V[NS,T,D], eps)        -> E[NS,T,D]
      race_prefix_mean_flat_bw(probsK, V, gradE, eps)            -> [gW, gV]

  linear_pref -- Katharopoulos causal linear attention (fast-transformers)
      causal_dot_product(Q,K,V,out)                              # [N,H,L,E/M]
      causal_dot_backward(Q,K,V,grad_out,gQ,gK,gV)

Use lazily:
    from race_ext import load_race, load_linear
    race = load_race(); linear = load_linear()
"""
import os
import platform
from torch.utils.cpp_extension import load

_HERE = os.path.dirname(os.path.abspath(__file__))


def _omp_flags():
    cflags = ["-O3"]
    ldflags = []
    if platform.system() == "Darwin":
        omp = os.environ.get("LIBOMP_PREFIX")
        if not omp:
            for p in ("/opt/homebrew/opt/libomp", "/usr/local/opt/libomp"):
                if os.path.isdir(p):
                    omp = p
                    break
        if not omp:
            raise SystemExit("libomp not found. `brew install libomp` or set LIBOMP_PREFIX")
        cflags += ["-Xpreprocessor", "-fopenmp", f"-I{omp}/include"]
        ldflags += [f"-L{omp}/lib", "-lomp", f"-Wl,-rpath,{omp}/lib"]
    else:
        cflags += ["-fopenmp"]
        ldflags += ["-fopenmp"]
    return cflags, ldflags


_race = None
_linear = None


def load_race(verbose=False):
    global _race
    if _race is None:
        cf, lf = _omp_flags()
        _race = load(name="race_pref_cpu",
                     sources=[os.path.join(_HERE, "race_pref.cpp")],
                     extra_cflags=cf, extra_ldflags=lf, verbose=verbose)
    return _race


def load_linear(verbose=False):
    global _linear
    if _linear is None:
        cf, lf = _omp_flags()
        _linear = load(name="linear_pref_cpu",
                       sources=[os.path.join(_HERE, "linear_pref.cpp")],
                       extra_cflags=cf, extra_ldflags=lf, verbose=verbose)
    return _linear


if __name__ == "__main__":
    print("building race_pref ...")
    load_race(verbose=True)
    print("building linear_pref ...")
    load_linear(verbose=True)
    print("OK: both CPU extensions built.")
