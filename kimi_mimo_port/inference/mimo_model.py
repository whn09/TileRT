"""MiMo-V2.5-Pro top-level model (example-level, correctness-first).

Reuses DeepSeek scaffold primitives (ParallelEmbedding / Linear /
ColumnParallelLinear / RowParallelLinear / RMSNorm) but swaps in:
  - MiMoAttention (GQA + hybrid full/SWA + partial RoPE + sink)  [mimo_attn.py]
  - MiMoMoE (MXFP4 routed experts, no shared expert)             [mimo_moe.py]
RoPE is handled *inside* MiMoAttention (partial + per-layer theta), so this
file does not precompute global freqs_cis.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.distributed as dist
from torch import nn

import model as ds  # deepseek scaffold (Linear, norms, parallel layers)
from mimo_attn import MiMoAttention
from mimo_moe import MiMoMoE


@dataclass
class MiMoArgs:
    max_batch_size: int = 1
    max_seq_len: int = 4096
    dtype: str = "fp8"
    scale_fmt: Optional[str] = None

    vocab_size: int = 152576
    dim: int = 6144                 # hidden_size
    inter_dim: int = 16384          # dense MLP
    moe_inter_dim: int = 2048
    n_layers: int = 70
    n_dense_layers: int = 1         # layer 0 dense; 1..69 MoE

    # GQA full-attention
    num_attention_heads: int = 128
    num_key_value_heads: int = 8
    head_dim: int = 192
    v_head_dim: int = 128
    attention_value_scale: float = 0.612
    rope_theta: float = 5_000_000.0
    partial_rotary_factor: float = 0.334
    add_full_attention_sink_bias: bool = False

    # SWA
    swa_num_attention_heads: int = 128
    swa_num_key_value_heads: int = 8
    swa_head_dim: int = 192
    swa_v_head_dim: int = 128
    swa_rope_theta: float = 10000.0
    sliding_window: int = 128
    add_swa_attention_sink_bias: bool = True
    hybrid_layer_pattern: List[int] = field(default_factory=list)

    # MoE routing
    n_routed_experts: int = 384
    num_experts_per_tok: int = 8
    n_shared_experts: int = 0
    score_func: str = "sigmoid"
    n_expert_groups: int = 1       # n_group
    n_limited_groups: int = 1      # topk_group
    n_activated_experts: int = 8
    route_scale: float = 1.0       # MiMo routed_scaling_factor is None -> 1.0
    norm_topk_prob: bool = True
    mxfp4_block_size: int = 32

    eps: float = 1e-5
    world_size: int = 1


class MiMoBlock(nn.Module):
    def __init__(self, layer_id: int, args: MiMoArgs):
        super().__init__()
        self.attn = MiMoAttention(args, layer_id)
        if layer_id < args.n_dense_layers:
            self.ffn = ds.MLP(args.dim, args.inter_dim)
        else:
            self.ffn = MiMoMoE(args, layer_id)
        self.attn_norm = ds.RMSNorm(args.dim, args.eps)
        self.ffn_norm = ds.RMSNorm(args.dim, args.eps)
        self.max_seq_len = args.max_seq_len
        self.partial = args.partial_rotary_factor

    def forward(self, x, residual, start_pos, mask):
        if residual is None:
            x, residual = self.attn_norm(x), x
        else:
            x, residual = self.attn_norm(x, residual)
        x = self.attn(x, start_pos, mask, self.max_seq_len, self.partial)
        x, residual = self.ffn_norm(x, residual)
        x = self.ffn(x)
        return x, residual


class MiMoTransformer(nn.Module):
    def __init__(self, args: MiMoArgs):
        super().__init__()
        global_ws = dist.get_world_size() if dist.is_initialized() else 1
        args.world_size = global_ws
        ds.world_size = global_ws
        ds.rank = dist.get_rank() if dist.is_initialized() else 0
        ds.Linear.dtype = torch.float8_e4m3fn if args.dtype == "fp8" else torch.bfloat16
        ds.Linear.scale_fmt = args.scale_fmt

        self.args = args
        self.max_seq_len = args.max_seq_len
        self.embed = ds.ParallelEmbedding(args.vocab_size, args.dim)
        self.layers = nn.ModuleList([MiMoBlock(i, args) for i in range(args.n_layers)])
        self.norm = ds.RMSNorm(args.dim, args.eps)
        self.head = ds.ColumnParallelLinear(args.dim, args.vocab_size, dtype=torch.float32)

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int = 0):
        # MiMoAttention builds its own causal/SWA mask from positions, so we
        # only pass a sentinel (None) here; the per-layer attn handles masking.
        h, residual = self.embed(tokens), None
        for layer in self.layers:
            h, residual = layer(h, residual, start_pos, None)
        h, _ = self.norm(h, residual)
        logits = self.head(h[:, -1].float())
        if dist.is_initialized() and dist.get_world_size() > 1:
            alll = [torch.empty_like(logits) for _ in range(dist.get_world_size())]
            dist.all_gather(alll, logits)
            logits = torch.cat(alll, dim=-1)
        return logits
