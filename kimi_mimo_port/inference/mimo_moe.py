"""MiMo MoE with MXFP4 experts (correctness-first, dequant-on-use).

MiMo: 384 routed experts, top-8, sigmoid + noaux_tc routing, norm_topk_prob,
NO shared experts. Expert gate/up/down weights are MXFP4 (uint8-packed FP4 +
E8M0 block scales, block 32); dense layer-0 MLP is FP8.

For correctness we keep expert weights quantized in memory (uint8, 4-bit ->
~570GB across 8 GPUs is feasible) and dequantize each *selected* expert's
weights to bf16 on the fly per token-batch. This is slow (example-level) but
memory-safe; a fused MXFP4 grouped-GEMM (TileOPs) is the perf path later.
"""

import torch
import torch.nn.functional as F
from torch import nn

from mimo_quant import dequant_mxfp4


class MXFP4Expert(nn.Module):
    """One routed expert: gate/up/down, weights held as MXFP4 (uint8)."""

    def __init__(self, dim: int, inter_dim: int, block: int = 32):
        super().__init__()
        self.dim = dim
        self.inter_dim = inter_dim
        self.block = block
        # packed FP4: [out, in/2] uint8 ; scale: [out, in/block] uint8
        self.gate_w = nn.Parameter(torch.zeros(inter_dim, dim // 2, dtype=torch.uint8), False)
        self.gate_s = nn.Parameter(torch.zeros(inter_dim, dim // block, dtype=torch.uint8), False)
        self.up_w = nn.Parameter(torch.zeros(inter_dim, dim // 2, dtype=torch.uint8), False)
        self.up_s = nn.Parameter(torch.zeros(inter_dim, dim // block, dtype=torch.uint8), False)
        self.down_w = nn.Parameter(torch.zeros(dim, inter_dim // 2, dtype=torch.uint8), False)
        self.down_s = nn.Parameter(torch.zeros(dim, inter_dim // block, dtype=torch.uint8), False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [n_tokens_routed_here, dim]
        gw = dequant_mxfp4(self.gate_w, self.gate_s, self.block, x.dtype)
        uw = dequant_mxfp4(self.up_w, self.up_s, self.block, x.dtype)
        dw = dequant_mxfp4(self.down_w, self.down_s, self.block, x.dtype)
        h = F.silu(F.linear(x, gw)) * F.linear(x, uw)
        return F.linear(h, dw)


class MiMoMoE(nn.Module):
    """Routed-only MoE (no shared experts), MXFP4 experts."""

    def __init__(self, args, layer_id: int):
        super().__init__()
        from model import Gate  # reuse DeepSeek gate (sigmoid + bias + topk)
        self.n_experts = args.n_routed_experts
        self.topk = args.num_experts_per_tok
        self.norm_topk = args.norm_topk_prob
        self.gate = Gate(args)
        world = args.world_size
        self.n_local = self.n_experts // world
        self.expert_start = (torch.distributed.get_rank() * self.n_local
                             if torch.distributed.is_initialized() else 0)
        self.experts = nn.ModuleList([
            MXFP4Expert(args.dim, args.moe_inter_dim, args.mxfp4_block_size)
            for _ in range(self.n_local)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.size()
        x = x.view(-1, shape[-1])
        weights, indices = self.gate(x)  # [n_tok, topk]
        y = torch.zeros_like(x)
        counts = torch.bincount(indices.flatten(), minlength=self.n_experts).tolist()
        for i in range(self.n_local):
            gid = self.expert_start + i
            if counts[gid] == 0:
                continue
            idx, top = torch.where(indices == gid)
            y[idx] += self.experts[i](x[idx]) * weights[idx, top, None]
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(y)
        return y.view(shape)
