"""Model arguments for Kimi-K2.6 (text backbone).

Transcribed from ``moonshotai/Kimi-K2.6`` ``config.json`` -> ``text_config``
(``model_type == "kimi_k2"``). The backbone is an MLA MoE, structurally close to
DeepSeek-V3.2, so this dataclass deliberately mirrors
``deepseek_v3_2.model_args.ModelArgs`` field-for-field, overriding only the
values that differ (vocab, experts, routing, rope_theta, seq len).

Two published checkpoints differ only in MoE-expert quantization; everything
else (attention, shared experts, dense MLP, lm_head, layer 0, vision) stays at
higher precision in both:
  - ``moonshotai/Kimi-K2.6``            -> INT4 (compressed-tensors, group 32)
  - ``nvidia/Kimi-K2.6-NVFP4``          -> NVFP4 (float, num_bits 4, group 16)

The NVFP4 build is the one to target on B200 (native FP4 tensor cores). Notably
NVIDIA exports it with ``text_config.model_type == "deepseek_v3"`` — i.e. they
treat Kimi's backbone as DeepSeek-V3 — which is strong corroboration that the
DeepSeek-V3.2 MLA kernels can drive Kimi. ``moe_quant`` selects the format.
"""

from dataclasses import dataclass
from typing import Literal

__all__ = [
    "ModelArgsKimiK2",
]


@dataclass
class ModelArgsKimiK2:
    """Hyperparameters for Kimi-K2.6 text backbone (MLA MoE)."""

    arch_name: str = "kimi_k2"

    max_batch_size: int = 1
    # text_config.max_position_embeddings = 262_144
    max_seq_len: int = 256 * 1024
    dtype: Literal["bf16", "fp8"] = "fp8"
    scale_fmt: str | None = None

    # ---- MoE-expert quantization ----
    # Default to NVFP4 (nvidia/Kimi-K2.6-NVFP4): float, num_bits 4, group 16 —
    # the build meant for B200 native FP4 tensor cores. Switch to "int4"
    # (group 32) for the moonshotai/Kimi-K2.6 compressed-tensors checkpoint.
    # Only routed experts are quantized; self_attn / shared_experts / lm_head /
    # layer 0 stay at higher precision.
    moe_quant: Literal["bf16", "fp8", "int4", "nvfp4"] = "nvfp4"
    # NVFP4 group size is 16; INT4 checkpoint uses 32. Kept in sync with moe_quant.
    moe_quant_block_size: int = 16

    # ---- dimensions ----
    vocab_size: int = 163840
    dim: int = 7168  # hidden_size
    inter_dim: int = 18432  # dense MLP intermediate_size
    moe_inter_dim: int = 2048  # moe_intermediate_size
    n_layers: int = 61  # num_hidden_layers
    n_dense_layers: int = 1  # first_k_dense_replace
    n_heads: int = 64  # num_attention_heads

    # ---- MoE routing ----
    n_routed_experts: int = 384
    n_shared_experts: int = 1
    n_activated_experts: int = 8  # num_experts_per_tok
    n_expert_groups: int = 1  # n_group
    n_limited_groups: int = 1  # topk_group
    score_func: Literal["softmax", "sigmoid"] = "sigmoid"
    topk_method: str = "noaux_tc"
    route_scale: float = 2.827  # routed_scaling_factor

    # ---- MLA attention (same shape family as DeepSeek-V3.2) ----
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128

    # ---- RoPE ----
    original_seq_len: int | None = None
    rope_theta: float = 50000.0
    rope_factor: float | None = None
    beta_fast: int | None = None
    beta_slow: int | None = None
    mscale: float = 1.0

    block_size: int = 128
    eps: float = 1e-5  # rms_norm_eps
    tie_word_embeddings: bool = False

    # num_nextn_predict_layers = 0 in config -> no native MTP layer shipped.
    num_nextn_predict_layers: int = 0
