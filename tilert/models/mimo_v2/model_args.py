"""Model arguments and hyperparameters for MiMo-V2.5-Pro.

Values are transcribed from the official HuggingFace checkpoint
``XiaomiMiMo/MiMo-V2.5-Pro-FP4-DFlash`` (``config.json``, ``text`` portion).
The ``quantization_config`` there is mixed precision:

  - MoE experts:        MXFP4 (block size 32)
  - everything else:    FP8 (e4m3, weight_block_size [128, 128])
  - every ``self_attn.o_proj``: excluded from FP4 (kept at higher precision)

so the dataclass carries an explicit ``moe_quant`` / ``moe_quant_block_size`` to
describe the FP4 MoE path, distinct from the FP8 ``dtype`` used elsewhere.
"""

from dataclasses import dataclass, field
from typing import Literal

__all__ = [
    "ModelArgsMiMoV2",
]


@dataclass
class ModelArgsMiMoV2:
    """Hyperparameters for MiMo-V2.5-Pro (1T MoE, GQA + SWA hybrid attention).

    Unlike DeepSeek-V3.2 / GLM-5 (MLA + global sparse attention), MiMo uses
    grouped-query attention with a hybrid full / sliding-window layer pattern.
    The fields below mirror the names used by the existing TileRT model-arg
    dataclasses where they correspond, and add MiMo-specific fields
    (``hybrid_layer_pattern``, ``sliding_window``, ``moe_quant`` ...).
    """

    arch_name: str = "mimo_v2"

    max_batch_size: int = 1
    # config.json: max_position_embeddings = 1_048_576 (1M context).
    max_seq_len: int = 1024 * 1024

    # Non-expert compute precision (attention, dense MLP, norms, lm_head).
    dtype: Literal["bf16", "fp8"] = "fp8"
    scale_fmt: str | None = None

    # ---- MoE-expert quantization (the FP4 path that makes UltraSpeed fast) ----
    # MXFP4 with a 32-element block; only routed-expert GEMM weights use it.
    moe_quant: Literal["bf16", "fp8", "mxfp4"] = "mxfp4"
    moe_quant_block_size: int = 32
    # o_proj of every attention layer is explicitly NOT quantized to FP4.
    exclude_o_proj_from_quant: bool = True

    # ---- dimensions ----
    vocab_size: int = 152576
    dim: int = 6144  # hidden_size
    inter_dim: int = 16384  # dense MLP intermediate_size
    moe_inter_dim: int = 2048  # moe_intermediate_size
    n_layers: int = 70  # num_hidden_layers

    # ---- attention (GQA, NOT MLA) ----
    n_heads: int = 128  # num_attention_heads
    n_kv_heads: int = 8  # num_key_value_heads (GQA)
    head_dim: int = 192
    v_head_dim: int = 128
    attention_value_scale: float = 0.612

    # ---- hybrid full / sliding-window attention ----
    # config.json `hybrid_layer_pattern`: 0 = full attention, 1 = SWA.
    sliding_window: int = 128
    attention_chunk_size: int = 128
    add_swa_attention_sink_bias: bool = True
    add_full_attention_sink_bias: bool = False
    # Per-layer 0/1 pattern (0=full, 1=SWA). Populated from config at load time;
    # left empty here so the converter can fill it from the checkpoint.
    hybrid_layer_pattern: list[int] = field(default_factory=list)

    # ---- MoE routing ----
    n_routed_experts: int = 384
    n_shared_experts: int = 0  # config: n_shared_experts = None
    n_activated_experts: int = 8  # num_experts_per_tok
    n_expert_groups: int = 1  # n_group
    n_limited_groups: int = 1  # topk_group
    score_func: Literal["softmax", "sigmoid"] = "sigmoid"
    route_scale: float | None = None  # routed_scaling_factor (None in config)
    norm_topk_prob: bool = True
    # Index of the first MoE layer (dense layers come first). Derived from
    # `moe_layer_freq` in config; filled at load time.
    n_dense_layers: int = 1

    # ---- RoPE ----
    rope_theta: float = 5_000_000.0
    partial_rotary_factor: float = 1.0
    original_seq_len: int | None = None
    rope_factor: float | None = None

    # ---- misc ----
    block_size: int = 128
    eps: float = 1e-5  # layernorm_epsilon
    tie_word_embeddings: bool = False

    # ---- DFlash speculative decoding (block-diffusion drafter) ----
    # The open-sourced checkpoint ships a separate BF16 DFlash drafter that
    # fills a whole masked block per forward pass. Block size 8 per the blog.
    dflash_block_size: int = 8
    dflash_dtype: Literal["bf16"] = "bf16"
