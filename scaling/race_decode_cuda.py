"""Fused single-token RACE decode backed by the standalone CUDA kernel.

Thin Python wrapper around the ``race_decode`` extension (decode_kernel.cu). One
fused decode step per call: soft-hash of q/k, running-prefix state update of A,B
(IN PLACE), and the readout into out. Separate from scaling/race_causal_cuda.py
(the chunked training scan).
"""
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "kernels", "gpu"))
from race_decode_build import load_ext  # noqa: E402

_EXT = None


def _ext():
    global _EXT
    if _EXT is None:
        _EXT = load_ext(verbose=True)
    return _EXT


def race_decode_step(q, k, v, planes_T, protos_T, A, B, out, scale, eps, L, Kbits):
    """One fused decode step. q,k,v [N,hd] fp32; planes_T [hd,L*Kbits]; protos_T [Kbits,R];
    A [N,S] and B [N,S,hd] updated IN PLACE; out [N,hd] written. scale = sqrt(hd)*exp(log_temp)
    computed host-side. Returns out."""
    ext = _ext()
    ext.race_decode_step(q.contiguous(), k.contiguous(), v.contiguous(),
                         planes_T.contiguous(), protos_T.contiguous(),
                         A, B, out, float(scale), float(eps), int(L), int(Kbits))
    return out
