# MiMo-V2.5-Pro 架构蓝图（移植依据）

> 来源：`XiaomiMiMo/MiMo-V2.5-Pro-FP4-DFlash` 的 `config.json` + `configuration_mimo_v2.py`。
> **注意：repo 没有 `modeling_mimo_v2.py`（官方只给了 config，模型实现未公开）**，
> 所以 MiMo 的 modeling 必须按 config 自己实现，不能照搬。

## 核心参数

| 项 | 值 | 说明 |
|---|---|---|
| hidden_size | 6144 | |
| num_hidden_layers | 70 | |
| **attention** | **GQA**（非 MLA！） | q_proj/k_proj/v_proj/o_proj，fused_qkv 布局 |
| num_attention_heads | 128 | query heads |
| num_key_value_heads | 8 | KV heads（GQA 16:1）|
| head_dim | 192 | QK head dim |
| **v_head_dim** | **128** | V head dim ≠ QK dim！|
| attention_value_scale | 0.612 | V 缩放 |
| **partial_rotary_factor** | **0.334** | 只 ~64/192 维用 RoPE（partial RoPE）|
| rope_theta | 5,000,000 | 全注意力层 |

## 混合注意力（hybrid）

`hybrid_layer_pattern`：`0` = 全注意力，`1` = 滑窗（SWA）。
前20层模式：`[0,1,1,1,1,1,1, 0,1,1,1,1,1,1,1, 0,1,1,1,1, ...]`
→ 每 7-8 层插一个全注意力层，其余都是 SWA。

SWA 层参数（独立于全注意力）：
- swa_num_attention_heads 128 / swa_num_key_value_heads 8 / swa_head_dim 192 / swa_v_head_dim 128
- **sliding_window = 128**
- **swa_rope_theta = 10000**（≠ 全注意力的 5e6！）
- **add_swa_attention_sink_bias = True**（SWA 层有 attention sink）
- add_full_attention_sink_bias = False（全注意力层无 sink）

## MoE

- n_routed_experts 384 / num_experts_per_tok 8 / **n_shared_experts None（无共享专家）**
- scoring_func sigmoid + topk_method noaux_tc + norm_topk_prob True
- n_group 1 / topk_group 1（无分组路由）
- moe_layer_freq：layer 0 是 dense MLP，layer 1-69 是 MoE
- moe_intermediate_size 2048 / dense intermediate_size 16384

## 量化

- store_dtype mxfp4，mxfp4_block_size 32 → **MoE 专家 = MXFP4（block 32）**
- 其余 = FP8 e4m3（weight_block_size [128,128]）
- ignored_layers：每层 `self_attn.o_proj` 不量化

## 移植难点（vs Kimi 的 MLA）

| 组件 | Kimi（已跑通） | MiMo |
|---|---|---|
| 注意力 | 标准 MLA，绕过 indexer 即可 | **GQA + 混合全/滑窗 + partial RoPE + sink + V≠QK dim**，需从头写 |
| RoPE | 标准 | **partial（仅 1/3 维）+ 全/滑窗两套 theta** |
| MoE 量化 | FP8（example 现成） | **MXFP4 block32**（需 TileOPs mxfp4 dequant GEMM）|
| 共享专家 | 有 | 无 |

## 实现策略（example 级）

1. 用 deepseek inference scaffold 的外壳（embed/norm/MoE 路由/generate 循环）
2. **重写 attention 层**：GQA + 按 hybrid_layer_pattern 切换 full/SWA + partial RoPE + SWA sink bias
   - attention kernel 用 TileOPs 的 `gqa_decode` / `gqa_sliding_window_fwd`，或先用纯 PyTorch SDPA 保正确性
3. **MoE 改 MXFP4**：dequant 专家权重（mxfp4→bf16）后走普通 grouped MLP，或用 tilelang mxfp4 GEMM
4. dense layer 0 + MoE layer 1-69
5. fused_qkv 权重切分

先求正确性（可先用纯 PyTorch attention，不追 kernel 性能），跑通后再换 tilelang kernel。
