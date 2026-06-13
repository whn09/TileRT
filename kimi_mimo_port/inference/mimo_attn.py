"""MiMo-V2.5-Pro attention: GQA + hybrid full/sliding-window + partial RoPE + sink.

This is the core of the MiMo port and the main thing that differs from the
DeepSeek/Kimi MLA scaffold. Implemented in plain PyTorch (SDPA-style matmul)
for correctness first; kernel acceleration (TileOPs gqa_sliding_window) is a
later optimization.

MiMo architecture (from config.json):
  - GQA: 128 query heads, 8 KV heads, head_dim 192, v_head_dim 128
  - fused qkv_proj weight [27136, 6144]  (Q 128*192 + K 8*192 + V 8*128)
  - partial RoPE: only first head_dim*partial_rotary_factor (~64) dims rotated
  - hybrid layers: full attention (rope_theta 5e6) vs sliding-window
    (window 128, rope_theta 1e4, + per-head attention_sink_bias)
  - attention_value_scale 0.612 applied to V
  - o_proj [6144, 16384] unquantized bf16
"""

import math
import torch
import torch.nn.functional as F
from torch import nn


def precompute_rope(head_dim: int, partial_factor: float, theta: float,
                    max_seq: int, device="cuda"):
    """Precompute cos/sin for partial RoPE.

    Only the first ``rot_dim = round(head_dim * partial_factor)`` dims (rounded
    to even) are rotated; the rest pass through unchanged.
    """
    rot_dim = int(round(head_dim * partial_factor))
    rot_dim -= rot_dim % 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, rot_dim, 2, device=device).float() / rot_dim))
    t = torch.arange(max_seq, device=device).float()
    freqs = torch.outer(t, inv_freq)  # [max_seq, rot_dim/2]
    return torch.cos(freqs), torch.sin(freqs), rot_dim


def apply_partial_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                       rot_dim: int) -> torch.Tensor:
    """Apply RoPE to the first ``rot_dim`` dims of x [b, s, h, d]; pass rest through."""
    x_rot, x_pass = x[..., :rot_dim], x[..., rot_dim:]
    # interleaved (real, imag) pairs
    x1 = x_rot[..., 0::2]
    x2 = x_rot[..., 1::2]
    c = cos[None, :, None, :]  # [1, s, 1, rot/2]
    s = sin[None, :, None, :]
    rx1 = x1 * c - x2 * s
    rx2 = x1 * s + x2 * c
    x_rot_out = torch.stack([rx1, rx2], dim=-1).flatten(-2)
    return torch.cat([x_rot_out, x_pass], dim=-1)


class MiMoAttention(nn.Module):
    """GQA attention with optional sliding window + attention sink (per layer)."""

    def __init__(self, args, layer_id: int):
        super().__init__()
        self.layer_id = layer_id
        # hybrid_layer_pattern: 0 = full attention, 1 = sliding window
        self.is_swa = bool(args.hybrid_layer_pattern[layer_id]) if args.hybrid_layer_pattern else False

        self.n_heads = args.swa_num_attention_heads if self.is_swa else args.num_attention_heads
        self.n_kv = args.swa_num_key_value_heads if self.is_swa else args.num_key_value_heads
        self.head_dim = args.swa_head_dim if self.is_swa else args.head_dim
        self.v_head_dim = args.swa_v_head_dim if self.is_swa else args.v_head_dim
        self.theta = args.swa_rope_theta if self.is_swa else args.rope_theta
        self.window = args.sliding_window if self.is_swa else None
        self.value_scale = args.attention_value_scale or 1.0
        self.softmax_scale = self.head_dim ** -0.5

        world = args.world_size
        self.n_local_heads = self.n_heads // world
        # GQA: replicate KV heads if fewer than world (here n_kv=8, world=8 -> 1/dev)
        self.n_local_kv = max(1, self.n_kv // world)

        qsz = self.n_heads * self.head_dim
        ksz = self.n_kv * self.head_dim
        vsz = self.n_kv * self.v_head_dim
        # fused qkv (col-parallel); o_proj row-parallel, unquantized
        from model import ColumnParallelLinear, RowParallelLinear
        self.qkv_proj = ColumnParallelLinear(args.dim, qsz + ksz + vsz)
        self.o_proj = RowParallelLinear(self.n_heads * self.v_head_dim, args.dim)
        self._qkv_split = (qsz, ksz, vsz)

        self.has_sink = (args.add_swa_attention_sink_bias if self.is_swa
                         else args.add_full_attention_sink_bias)
        if self.has_sink:
            self.attention_sink_bias = nn.Parameter(
                torch.zeros(self.n_local_heads, dtype=torch.bfloat16))
        else:
            self.attention_sink_bias = None

        # rope cache (set in from_pretrained / lazily)
        self._rope = None
        self.kv_cache = None
        self.v_cache = None

    def _ensure_rope(self, max_seq, device):
        if self._rope is None:
            cos, sin, rot = precompute_rope(self.head_dim, self.partial_factor,
                                            self.theta, max_seq, device)
            self._rope = (cos, sin, rot)

    def forward(self, x, start_pos, mask, max_seq, partial_factor):
        self.partial_factor = partial_factor
        bsz, seqlen, _ = x.size()
        end = start_pos + seqlen
        self._ensure_rope(max_seq, x.device)
        cos, sin, rot = self._rope
        cs = cos[start_pos:end]; sn = sin[start_pos:end]

        qkv = self.qkv_proj(x)
        qsz, ksz, vsz = self._qkv_split
        # account for tensor-parallel sharding of the fused dim
        qsz_l = qsz // self.qkv_proj.world_size if hasattr(self.qkv_proj, "world_size") else qsz
        q, k, v = torch.split(qkv, [self.n_local_heads * self.head_dim,
                                    self.n_local_kv * self.head_dim,
                                    self.n_local_kv * self.v_head_dim], dim=-1)
        q = q.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        k = k.view(bsz, seqlen, self.n_local_kv, self.head_dim)
        v = v.view(bsz, seqlen, self.n_local_kv, self.v_head_dim)

        q = apply_partial_rope(q, cs, sn, rot)
        k = apply_partial_rope(k, cs, sn, rot)

        # KV cache
        if self.kv_cache is None:
            self.kv_cache = torch.zeros(bsz, max_seq, self.n_local_kv, self.head_dim,
                                        dtype=k.dtype, device=k.device)
            self.v_cache = torch.zeros(bsz, max_seq, self.n_local_kv, self.v_head_dim,
                                       dtype=v.dtype, device=v.device)
        self.kv_cache[:bsz, start_pos:end] = k
        self.v_cache[:bsz, start_pos:end] = v
        k_all = self.kv_cache[:bsz, :end]
        v_all = self.v_cache[:bsz, :end]

        # GQA: repeat KV to match query heads
        rep = self.n_local_heads // self.n_local_kv
        k_all = k_all.repeat_interleave(rep, dim=2)
        v_all = v_all.repeat_interleave(rep, dim=2)

        # scores [b, h, s, t]
        scores = torch.einsum("bshd,bthd->bhst", q.float(), k_all.float()) * self.softmax_scale

        # causal + (optional) sliding-window mask
        qpos = torch.arange(start_pos, end, device=x.device)[:, None]
        kpos = torch.arange(0, end, device=x.device)[None, :]
        causal = kpos <= qpos
        if self.window is not None:
            causal = causal & (kpos > qpos - self.window)
        attn_mask = torch.where(causal, 0.0, float("-inf")).to(scores.dtype)
        scores = scores + attn_mask[None, None]

        if self.attention_sink_bias is not None:
            # attention sink: an extra logit per head that absorbs probability
            sink = self.attention_sink_bias.float().view(1, -1, 1, 1)
            sink = sink.expand(bsz, -1, seqlen, 1)
            scores = torch.cat([scores, sink], dim=-1)
            probs = scores.softmax(dim=-1)
            probs = probs[..., :-1]  # drop the sink column after normalization
        else:
            probs = scores.softmax(dim=-1)

        out = torch.einsum("bhst,bthd->bshd", probs.to(v_all.dtype), v_all)
        out = out * self.value_scale
        out = out.reshape(bsz, seqlen, self.n_local_heads * self.v_head_dim)
        return self.o_proj(out)
