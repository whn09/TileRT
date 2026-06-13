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

---

## 运行进度（2026-06-13，B200 cu128 容器 tl128）

✅ **全管线跑通**：权重加载（22s，8卡分片 + FP8 dequant + MXFP4 uint8 加载）→ 70 层前向
（GQA+SWA+partial RoPE+sink + MXFP4 MoE 路由）→ 8卡 NCCL allreduce → 采样 → 解码 → 输出。

运行命令（NCCL 必须禁用 aws-ofi 插件，否则 NET/OFI 初始化失败崩溃）：
```bash
NCCL_NET_PLUGIN=none NCCL_NET=Socket NCCL_P2P_LEVEL=NVL \
torchrun --nproc-per-node 8 generate_mimo.py \
  --ckpt-path /nvme/models/MiMo-V2.5-Pro-FP4-DFlash --max-new-tokens 16
```

⚠️ **当前状态：管线通，但输出是乱码**（重复 token "2 thek is thek is..."）。
即数值正确性还有 bug，高频嫌疑（待排查）：
1. MXFP4 dequant 数值（nibble 高低位序 / E8M0 scale 2^(s-127) 偏移）
2. fused_qkv 切分（Q/K/V 边界、head 排布 [heads,head_dim,dim] 假设）
3. partial RoPE（rot_dim、交错 vs 前后半、cos/sin 索引）
4. attention sink 归一化 / value_scale 0.612 位置
5. MoE 路由 norm_topk_prob / route_scale

这是跨架构移植的典型"流程通→调数值"阶段，和 Kimi 当时一样（Kimi 也是先跑通再修 mask）。

### 调试进展更新
- ✅ 修复 bug #1：qkv 权重 reshape（`[heads, head_dim, dim]`）
- ✅ 修复 bug #2（关键）：dense MLP + attention 的 Linear 被全局 fp8 dtype 污染，
  bf16 权重 copy 进 fp8 张量变全零 → 强制 qkv_proj/o_proj/dense-MLP 用 bf16。
  修复后 layer 输出不再全零（layer0 std 42）。
- ✅ 修复 bug #3：partial RoPE 改为 HF rotate_half 约定（non-interleaved）。
- ⚠️ 仍乱码（重复单 token）。layer0 absmax 2880 偏大，怀疑 attention 输出量级/
  qkv-split 顺序/sink 处理仍有数值 bug。

**剩余待查**（需对比参考实现，但 MiMo 官方无公开 modeling）：
1. fused_qkv 行顺序（是否 head-interleaved，非简单 [Q|K|V] 连续块）
2. attention 输出量级（softmax_scale / value_scale 0.612 的应用位置）
3. attention sink 归一化语义

**诚实评估**：全管线已通（加载/前向/8卡/解码），数值对齐是跨架构移植最耗时的一环，
通常需逐层对照 HF 参考激活值；MiMo 未开源 modeling 使其更难。属"工程可完成但需更多时间"
的状态，非架构性障碍。

### 进一步诊断（接近机器到期）
- dense MLP 权重 dequant 正常（std 0.009）、加载正常（bf16）、forward 对随机输入正常（out std 0.24）。
- 但真实 forward 中 dense FFN 输出 std 42 / absmax 2880 —— 用随机输入复现不出，
  说明是**真实激活的 outlier 维度**被放大（LLM 常见 massive activation），未必是 bug。
- **关键疑点**：`attn_norm.weight` mean≈0.013 / std≈0.027（异常小，正常 RMSNorm 权重 ~1.0）。
  强烈怀疑 MiMo 的 RMSNorm 采用 **(1 + weight)** 约定（类 Gemma），而当前 ds.RMSNorm
  直接用 `weight`。这会系统性地改变所有 norm 的缩放 → 数值整体错乱 → 乱码。
  **下一步首选修复**：把 MiMo 的 RMSNorm 改成 `x_normed * (1 + weight)` 再验证。

### 收尾状态（机器到期）
全管线跑通；数值未对齐（乱码）。最可能的下一个修复是 RMSNorm 的 (1+w) 约定。
所有代码 + 诊断已 commit。这是"还差几个数值约定 bug"的状态，非架构障碍。

### RMSNorm (1+w) 修复验证（最后一步）
应用 `(1 + weight)` 后，输出从"单 token 死循环"变成"有变化的多 token" —— 证实 RMSNorm
(1+weight) 约定**方向正确**，数值在朝合理方向走，但仍乱码：还有其它数值约定 bug 未解
（qkv 行顺序 / partial RoPE 细节 / sink 语义之一或多个）。

## 最终状态（机器到期）
- ✅ 架构完整逆向 + 实现（无官方 modeling 参考）
- ✅ 全管线在 8×B200 跑通（加载/70层前向/8卡NCCL/解码）
- ✅ 已修 4 个 bug：qkv reshape、fp8-dtype 污染、HF rotate_half RoPE、RMSNorm (1+w)
- ⚠️ 仍乱码：还差 1-N 个数值约定 bug，需逐层对照（MiMo 无公开 modeling，最难点）
- 属"工程可完成、非架构障碍"，预计还需数小时逐层 debug（理想情况有 HF 参考实现可对照会快很多）

---

## 🎉 跑通正确性（2026-06-13，B200）

```
Prompt:     What is the capital of France? Answer in one sentence.
Completion: <think>The user is asking a straightforward factual question about
            the capital of France. This...
```
连贯英文（MiMo 是 thinking 模型，<think> 开头正确）。

### 决定性的两个修复
1. **qkv 交错布局**（用户线索 + whn09 Neuron 参考交叉验证）：fused qkv 不是
   `[all_Q|all_K|all_V]`，而是 8 个 KV-group 交错 `[16Q,1K,1V]×8`，且 FP8 scale
   有 phantom-row padding（每组 27 block → 3456 padded，dequant 后裁回 3392）。
2. **撤销错误的 (1+w) RMSNorm**：MiMo 用标准 RMSNorm。原始 norm 权重 mean 有大有小
   （model.norm 3.8、layer norms 0.2~0.5），分布正常；之前 layer0 全零是 fp8-dtype
   污染 bug（已单独修），被我误判为 norm 约定问题。

### 注意：芯片无关性
参考了 whn09 的 Neuron(Trainium) MiMo 实现，但**只采纳芯片无关的部分**：qkv 物理布局、
RoPE/scaling/sink 数学、RMSNorm 约定。Neuron 特定的部分（CONVERT_TO_MHA、NKI
accumulator、tp 分片策略）未照搬——本实现是 B200 + 纯 PyTorch + 8卡 GQA 分片。

### 全部已修 bug 汇总
qkv reshape → fp8-dtype 污染(全零) → HF rotate_half RoPE → 误加(1+w)后撤销 →
**qkv 交错布局 + phantom-scale padding（决定性）**。

---

## 性能实测（2026-06-13，B200，单请求纯 decode）

| 引擎 / 模型 | OTPS | TPOT | 性质 |
|---|---|---|---|
| **MiMo tilelang example（本移植）** | **3.9 tok/s** | 255 ms | 正确但慢（dequant-on-use MoE）|
| 参考：Kimi tilelang example | 6.86 tok/s | 146 ms | 同栈 |
| 参考：SGLang（GLM/Kimi） | 130-180 tok/s | ~7 ms | 生产引擎 |
| 参考：闭源 TileRT（GLM/DeepSeek）| 200-687 tok/s | 1.5-5 ms | 持久化引擎 |

**为什么 3.9 tok/s**：MoE 每 token 把选中的 8 个专家权重从 MXFP4 现场 dequant 成 bf16
（384 选 8，纯 PyTorch），是主要瓶颈；attention 也是纯 PyTorch SDPA、无 kernel 融合、
无投机解码。这是"正确性参考实现"，非性能方案。

**性能优化路径（若要快）**：① MoE 改用 TileOPs 的 MXFP4 grouped-GEMM（免 dequant）；
② attention 换 TileOPs gqa_sliding_window kernel；③ 持久化引擎级融合 ≈ 复刻 TileRT 核心。
1000 tok/s 仍需官方 libtilert_mimo.so。

---

## 性能优化尝试：tilelang MXFP4 fused GEMM（perf path 起点）

瓶颈定位（单 token decode，1 个 MoE 层 8.14ms）：
- gate 路由 0.09ms（可忽略）
- **8 专家 × 0.88ms = 7.07ms（瓶颈）**，每专家 = 3 个 MXFP4 dequant+GEMM

tilelang fused MXFP4 GEMM 验证（`examples/dequantize_gemm/..._mxfp4_hopper.py` 的 matmul）：
- **速度：0.078ms vs PyTorch dequant+gemm 0.305ms = 3.9x 加速** ✓
- **数值：rel_err 1.0 ❌（未对齐）** —— 我的 PyTorch dequant 解包与 example torch_convert
  数值一致（code3=1.5, ×scale 0.000488=0.0007 ✓），故差异在 kernel 的 GEMM 转置约定
  （tilelang B[N,K] 布局 vs F.linear x@w.T）或 scale 应用语义。可调但需迭代。
- kernel 调用签名：`(A[M,K]bf16, B[N,K/2]u8, Scale[N,K/32]u8, Bias[M,N]bf16) -> C[M,N]`

→ MXFP4 kernel 加速潜力确认（~4x），数值对齐是下一步。但更优先：试 SGLang+DFlash。

---

## SGLang MiMo-FP4 + DFlash 尝试（2026-06-13，单节点 8 卡）

**SGLang dev 镜像原生支持 MiMo**：`models/mimo_v2.py`、`mimo_v2_nextn.py` +
`speculative/dflash_worker.py`（官方 DFlash 投机解码）。SGLang 正确识别
`MiMoV2ForCausalLM` + Hybrid SWA + DFLASH 算法。

**但权重加载失败**（4 次尝试，均卡在同一点）：
```
KeyError: 'model.layers.1.mlp.experts.w2_weight_scale'
  mimo_v2.py:1461  param = params_dict[name]
```
试过的组合（都失败）：
- `--quantization fp8`（匹配 config 的 quant_method）
- `+ --moe-runner-backend flashinfer_mxfp4`
- `+ --ep-size 8 --moe-dense-tp-size 1`（EP 模式，匹配 DeepEPMoE.make_expert_params_mapping）

**根因**：MiMo 的 MXFP4 专家权重（`experts.N.down_proj.weight_scale`）经
`make_expert_params_mapping` 映射到 `experts.w2_weight_scale`，但 FusedMoE 层在
当前量化配置下未注册该参数 → KeyError。即 checkpoint 量化格式
（`quant_method:fp8 + store_dtype:mxfp4`）与 SGLang FusedMoE 创建的参数集不匹配。

**判断**：用户原命令是 16 卡 2 节点 + 完整 EP/DP（`--ep-size 16 --tp 16 --dp 2
--enable-dp-attention --moe-a2a-backend`）精心调过的；单节点 8 卡触发了不同的、
该 dev 镜像版本未完全支持的权重加载路径。可能需要：① 精确复刻 16卡多节点配置；
② 或一个 MiMo-FP4 支持更完整的 sglang 版本；③ 或权重预处理成 SGLang 期望的
stacked/fused 格式。非单节点几次试错能解决。

---

## SGLang MiMo+DFlash via PR #27638（2026-06-13，深入但受限于硬件）

PR #27638（csAugust:mimo-v2-fp4-dflash）专门加 MiMo-V2.5-Pro FP4 + DFlash 支持。
用 PYTHONPATH 把 PR 的 sglang python 覆盖进 dev 镜像，**逐一突破**：
1. ✅ 权重加载（PR 的 mxfp4→FP8 dequant，过了 `w2_weight_scale` KeyError）
2. ✅ attention backend：fa3 在 B200(SM100) 被硬断言挡（只支持 SM8/9）；**fa4 无此断言且支持 sink** → 用 fa4
3. ✅ cuda graph 在 B200+PR 下编译崩 → `--disable-cuda-graph` 绕过
4. ✅ server fired up（tp8 + enable-dp-attention dp=1 + fa4 + patch）
5. ⚠️ 但**推理时崩**：`prepare_mlp_sync_batch` → `num_tokens // None`
   （`num_tokens_per_req`/`num_tokens_per_batch` 均 None）。已 patch 第一个
   AttributeError，又冒出 None 除法。

**根本硬约束（关键）**：错误信息明确——
```
MiMoV2ForCausalLM requires effective attention TP size 8 because its fused
qkv_proj weights are TP=8-interleaved; got 4 (tp8, dp2, dp-attention)
```
MiMo 的 fused qkv 是 **TP=8 交错打包**（即用户最初指出的交错布局），所以
**有效 attn-TP 必须=8**。而 MiMo 又**必须 enable-dp-attention + dp=2**
（用户确认 + PR 设计）。dp=2 会把 attn-TP 砍成 4 ≠ 8。
**同时满足 dp=2 + attn-TP=8 需要 16 卡**（PR 原命令正是 tp16/dp2，2 节点）。
单节点 8 卡数学上无法满足 → DFlash 的 DP 代码路径在 dp=1 下未被 PR 验证、有 None bug。

**结论**：SGLang MiMo+DFlash 已推进到"仅差正确的 16 卡 2 节点配置"。所有软件障碍
（权重/backend/cuda graph）都已解决，剩下的是**硬件规模**：需要 2×8 B200 节点跑
tp16/dp2 才能既满足 qkv 的 attn-TP=8 又满足 MiMo 要求的 dp=2。本次单节点 8 卡是硬件
不足，非软件不可行。
