"""FlashAttention-3 (causal) adapter with SDPA fallbacks.

Unified [B,H,T,D] -> [B,H,T,D] interface so the benchmark can swap baselines:
  --fa3-impl fa3         genuine FlashAttention-3 (flash_attn_interface), Hopper
  --fa3-impl sdpa-cudnn  PyTorch SDPA cuDNN fused attention (Hopper, FA3-class)
  --fa3-impl sdpa-flash  PyTorch SDPA flash backend
All paths use the causal mask.
"""
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

_fa3 = None


def _load_fa3():
    global _fa3
    if _fa3 is None:
        import flash_attn_interface as fa3
        _fa3 = fa3
    return _fa3


def fa3_available():
    try:
        _load_fa3()
        return True
    except Exception:
        return False


def fa3_causal(Q, K, V, impl="fa3"):
    """Q,K,V: [B,H,T,D] (fp16/bf16). Returns [B,H,T,D]."""
    if impl == "fa3":
        fa3 = _load_fa3()
        # FA3 expects [B,T,H,D]
        q = Q.transpose(1, 2).contiguous()
        k = K.transpose(1, 2).contiguous()
        v = V.transpose(1, 2).contiguous()
        out = fa3.flash_attn_func(q, k, v, causal=True)
        if isinstance(out, (tuple, list)):
            out = out[0]
        return out.transpose(1, 2).contiguous()
    elif impl == "sdpa-cudnn":
        with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
            return F.scaled_dot_product_attention(Q, K, V, is_causal=True)
    elif impl == "sdpa-flash":
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            return F.scaled_dot_product_attention(Q, K, V, is_causal=True)
    else:
        raise ValueError(f"unknown fa3 impl: {impl}")
