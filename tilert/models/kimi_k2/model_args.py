"""Model arguments for Kimi-K2.6 (text backbone).

Transcribed from ``moonshotai/Kimi-K2.6`` ``config.json`` -> ``text_config``
(``model_type == "kimi_k2"``). The backbone is an MLA MoE, structurally close to
DeepSeek-V3.2, so this dataclass deliberately mirrors
``deepseek_v3_2.model_args.ModelArgs`` field-for-field, overriding only the
values that differ (vocab, experts, routing, rope_theta, seq len).

Quantization note: the released checkpoint quantizes MoE experts to **INT4**
(group_size 32, compressed-tensors), keeping attention / shared-experts / dense
MLP / lm_head / vision in higher precision. ``moe_quant`` is configurable so the
same args can describe an INT4 or a (hypothetical) NVFP4 build.
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
    # Open-source checkpoint: INT4 (compressed-tensors, group 32). Set to
    # "nvfp4" if/when an NVFP4 build is produced for B200.
    moe_quant: Literal["bf16", "fp8", "int4", "nvfp4"] = "int4"
    moe_quant_block_size: int = 32

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
