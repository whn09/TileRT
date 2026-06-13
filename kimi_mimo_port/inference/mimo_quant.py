"""Dequantization helpers for MiMo-V2.5-Pro weights.

MiMo uses mixed precision:
  - MoE expert weights: MXFP4 (FP4 e2m1 packed 2-per-byte) + E8M0 block scales
    (block_size=32). Stored as uint8 `.weight` [out, in/2] and uint8
    `.weight_scale` [out, in/32].
  - Everything else (qkv_proj, dense MLP): FP8 e4m3 + fp32 block scales
    `weight_scale_inv` (block [128,128]). o_proj is unquantized bf16.

These helpers dequantize to bf16. For a 1T model this is memory-heavy, so
expert dequant is meant to be done per-expert on demand, not all at once.
"""

import torch

# FP4 e2m1: 4-bit sign-exponent(2)-mantissa(1). The 16 representable values.
_FP4_E2M1_LUT = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


def dequant_mxfp4(weight_u8: torch.Tensor, scale_u8: torch.Tensor,
                  block_size: int = 32, out_dtype=torch.bfloat16) -> torch.Tensor:
    """Dequantize an MXFP4 weight to ``out_dtype``.

    Args:
        weight_u8: uint8 tensor [out, in/2] — two FP4 nibbles packed per byte
            (low nibble = first element, high nibble = second).
        scale_u8:  uint8 tensor [out, in/block_size] — E8M0 exponents; the
            multiplicative scale for a block is 2^(scale - 127).
        block_size: elements sharing one scale (32 for MiMo).

    Returns:
        bf16 weight [out, in].
    """
    dev = weight_u8.device
    lut = _FP4_E2M1_LUT.to(dev)
    out, half = weight_u8.shape
    in_dim = half * 2

    # unpack nibbles -> [out, in] fp4 codes
    lo = (weight_u8 & 0x0F).to(torch.long)
    hi = (weight_u8 >> 4).to(torch.long)
    codes = torch.empty(out, in_dim, dtype=torch.long, device=dev)
    codes[:, 0::2] = lo
    codes[:, 1::2] = hi
    vals = lut[codes]  # [out, in] fp32

    # per-block E8M0 scale: 2^(scale - 127)
    scale = scale_u8.to(torch.float32) - 127.0
    factor = torch.exp2(scale)  # [out, in/block]
    factor = factor.repeat_interleave(block_size, dim=1)[:, :in_dim]
    return (vals * factor).to(out_dtype)


def dequant_fp8_blockwise(weight_fp8: torch.Tensor, scale_inv: torch.Tensor,
                          block: int = 128, out_dtype=torch.bfloat16) -> torch.Tensor:
    """Dequantize a block-wise FP8 (e4m3) weight to ``out_dtype``.

    Args:
        weight_fp8: float8_e4m3fn tensor [out, in].
        scale_inv:  fp32 tensor [ceil(out/block), ceil(in/block)] block scales.
        block: block size (128 for MiMo / DeepSeek FP8).
    """
    out, in_dim = weight_fp8.shape
    w = weight_fp8.to(torch.float32)
    sout, sin = scale_inv.shape
    s = scale_inv.repeat_interleave(block, dim=0)[:out]
    s = s.repeat_interleave(block, dim=1)[:, :in_dim]
    return (w * s).to(out_dtype)
