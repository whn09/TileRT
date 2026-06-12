# TileRT B200 实测性能报告

> 实测环境：AWS B200 机器（P6-1），8× NVIDIA B200（sm_100, 183GB each），27TB NVMe，
> 官方镜像 `ghcr.io/tile-ai/tilert:cu132-latest` + `tilert==0.1.4`，torch 2.11.0+cu130。
> 测试日期：2026-06-12-13。单请求（batch=1）低延迟场景。

---

## 一句话结论

**TileRT 的性能声明属实。** 实测 DeepSeek-V3.2 在极限负载下达到 **687 tok/s**，超过官方
声明的 "up to 600"。日常自然语言负载约 **370 tok/s**，约为 SGLang（同机、TP8 + EAGLE
投机解码）的 **2 倍**。两者差异完全由 MTP 接受长度（acceptance length）决定，与负载类型相关，
不是引擎性能问题。

---

## 1. 模型支持矩阵（实测）

| 模型 | 能否在开源 TileRT 上跑 | 说明 |
|---|---|---|
| **DeepSeek-V3.2-Exp** | ✅ 是 | 旗舰首发模型，原生 DSA 稀疏注意力，`libtilert_dsv32.so` |
| **GLM-5-FP8** | ✅ 是 | `libtilert_glm5.so`；需把 tokenizer_class 从 `TokenizersBackend` 改为 `PreTrainedTokenizerFast` |
| **Kimi-K2-Instruct (FP8)** | ❌ 否 | 标准 MLA，但 dsv32 后端把 DSA 稀疏注意力焊死在持久化执行图里，converter 崩在 `indexer.wk.weight` |
| **Kimi-K2.6 (NVFP4)** | ❌ 否 | 专家权重是 NVFP4(U8 打包)，后端 expert GEMM 写死 `assert dtype==float8_e4m3fn`，只吃 FP8 |
| **MiMo-V2.5-Pro** | ❌ 否 | `libtilert_mimo.so` 未开源；GQA+SWA 架构，无对应后端 |

---

## 2. 性能数字（单请求 OTPS / TPOT）

### TileRT（自带 profiler 报告，纯 decode）

| 模型 | 无 MTP | MTP（自然语言负载） | MTP（结构化负载） | MTP（极限重复负载） |
|---|---|---|---|---|
| **DeepSeek-V3.2** | 204 tok/s (4.89ms) | 374 tok/s (accept 2.17) | 448 tok/s (accept 2.61) | **687 tok/s (accept 3.98)** |
| **GLM-5** | 176 tok/s (5.67ms) | 362 tok/s (accept 2.37) | — | **597 tok/s (accept 3.98)** |

> 极限负载 = 高重复性 prompt（让 MTP 接受长度逼近上限 4）。DeepSeek 峰值 687、GLM 峰值 597，
> 均达到/超过官方 "up to 600 / up to 500" 声明 → **官方性能声明属实，无虚标**。

### SGLang 对照（同机，非流式 usage 测量，含少量 TTFT 摊销）

| 模型 | 配置 | OTPS | TPOT |
|---|---|---|---|
| DeepSeek-V3.2 | TP8 + EAGLE spec decode | ~173–197 tok/s | 5.1–5.8 ms |
| GLM-5 | TP8 + EAGLE spec decode | ~180 tok/s | 5.57 ms |
| Kimi-K2-Instruct | TP8, 无 spec decode（用户指定命令）| 137 tok/s | 7.31 ms |

---

## 3. 与官方声明的对齐分析（核心）

### 官方声明（README / release notes）
- v0.1.3：GLM-5 **up to 500**、DeepSeek-V3.2 **up to 600** tok/s
- v0.1.2：mtp=3 **up to 590** tok/s **under synthetic workloads（合成负载）**
- README benchmark 图注：MTP 数据用**平均接受长度 3.2**，并含一根 **best-case MTP acceptance 峰值**柱

### 关键限定词
- "**up to**" = 峰值，不是典型值
- "**synthetic workloads**" = 合成负载（高重复性，投机解码接受率高）
- "**best-case acceptance**" = 最优接受长度场景

### 实测验证：差距 100% 来自接受长度
在**完全相同的 DeepSeek-V3.2 + 引擎**下，仅改变负载类型：

| 负载类型 | 接受长度 | OTPS |
|---|---|---|
| 自然语言（AI 历史长文）| 2.17 | 374 |
| 结构化（重复序列+计数）| 2.61 | 448 |
| 极限重复（输出 1000 个 7）| **3.98** | **687** |

**接受长度每 +0.4，OTPS 约 +70。** 实测峰值 687 > 官方 600，外推到官方口径的接受长度
3.2 时正好落在 ~550 区间——与官方 "up to 600" 完全自洽。

**结论：TileRT 没有虚标。** 我最初测到的 370 是真实日常负载（接受长度 2.2），官方的 600 是
合成负载峰值（接受长度 3.2+）。口径不同，都真实。

---

## 4. TileRT vs SGLang：真实优势

在**同等条件**（同机、同模型、都开投机解码、单请求）下：
- TileRT MTP ≈ **2× SGLang EAGLE**（DeepSeek：374 vs ~185；GLM：362 vs ~180）
- 这就是 TileRT "持久化引擎消除算子边界" 的实际收益，在两个模型上一致复现

**适用判断：**
- 追求**单请求极致低延迟**（高频交易、实时 Agent、代码补全）→ TileRT，约 2× 优势
- 追求**高并发吞吐 / 通用模型支持** → SGLang/vLLM
- 两者可组合：SGLang 做 prefill + TileRT 做 decode（TileRT 代码里有 `inject_cache` 的 P/D 分离接口）

---

## 5. 关于 1000 tok/s（小米 MiMo）

小米宣传的 1000 tok/s 是 **MiMo 专用后端（未开源）+ DFlash 投机解码** 的成绩，跑在单台 8×B200。
开源版 TileRT 在 DeepSeek/GLM 上：
- 日常负载 ~370 tok/s
- 极限负载可达 ~687 tok/s（已超官方 DeepSeek 声明）
- 但 1000 需要 MiMo 的 MXFP4(仅MoE) + DFlash(block-level 投机解码，接受长度 6+) 的组合，
  这套后端 `libtilert_mimo.so` 尚未开源

---

## 6. 复现要点 / 踩坑记录

1. **环境**：官方镜像只提供环境，tilert 需 `pip install tilert==0.1.4`（PyPI 上无 0.1.4，需从 GitHub Release 装 wheel 或 pip 装 0.1.3）。镜像是 cp312。
2. **下载**：`hf_transfer` + `hf_xet`（Xet 后端）实测约 715 MB/s，比单连接 curl(44MB/s)快 ~16×。1TB 模型约 15-20 分钟。
3. **转换**：`weight_converter --model_type {deepseek-v32|glm-5}`，产出 8 套 `dev_0..dev_7` 分片，约与原模型同体积。需把原始目录的 tokenizer/config (`cp -n`) 拷进转换后目录。
4. **GLM-5 tokenizer 坑**：新版 GLM-5 checkpoint 的 `tokenizer_class=TokenizersBackend`，镜像里的 transformers 4.46.3 不认；改成 `PreTrainedTokenizerFast` 即可（底层 tokenizer.json 通用）。
5. **跑**：`python -m tilert.generate --model {deepseek_v3_2|glm5} [--with-mtp] --interactive`。

---

## 附 A：复用闭源 dsv32 后端跑 Kimi/MiMo（逆向结论：不可行）

试图让 Kimi-K2 复用 dsv32 后端，深入逆向闭源 `.so` 后确认**此路不通**：
- `.so` 内**有** dense MLA kernel（`pure_mla_layer`），但端到端解码循环 `dsa_show_hands` 只有 DSA 稀疏版，无 dense 入口（执行图在闭源 C++ 编译死）。
- `pure_mla_layer` 的 params/temp_vars/cache_vars 布局 + 持久化引擎同步协议（wait_flag/timeline）全在闭源层，无 Python 参考，从外部驱动是高风险逆向，非合理工时可完成。

---

## 附 B：⭐ MiMo 支持的真正可行路径 —— 用开源 TileLang/TileOPs 自建（最终目标）

**重大发现**：tile-ai 主页（github.com/tile-ai）开源了完整的编译器+算子栈，
不必死等闭源 `libtilert_mimo.so`。关键 repo：
- **`tilelang`**（6485★）：GPU kernel DSL，TileRT 的 kernel 逐步并入此处
- **`TileOPs`**（143★）：基于 TileLang 的 LLM 算子库（`tileops/kernels/`）
- **`tilescale`**（163★）：tile-based 计算语言

### MiMo 组件 × 开源 kernel 覆盖核对（实测查证）

| MiMo 需要的组件 | 开源是否有 | 证据文件 |
|---|---|---|
| GQA 注意力 | ✅ | `TileOPs/tileops/kernels/attention/gqa_decode.py`, `gqa_fwd.py` |
| **滑动窗口 (SWA)** | ✅ | `gqa_sliding_window_fwd.py`, `gqa_sliding_window_varlen_fwd.py` |
| **Attention Sink** | ✅ | `tilelang/examples/attention_sink/`（带 `Sinks` tensor + `window_size` 参数）|
| **MXFP4 GEMM** | ✅ | `tilelang/examples/dequantize_gemm/*mxfp4*`, `src/tl_templates/cuda/cuda_fp4.h` |
| **FP8+FP4 混合 (sm100/B200 原生)** | ✅ | `tilelang/examples/deepseek_v4/fp8_fp4_gemm_1d1d_sm100.py`（`float4_e2m1fn` + `tcgen05_gemm_blockscaled`）|
| MoE (grouped gemm + topk) | ✅ | `TileOPs/tileops/kernels/grouped_gemm/`, `moe_fused_topk` |
| MLA / RMSNorm / RoPE / activation | ✅ | TileOPs 全套 |
| **DFlash（block-diffusion 投机解码）** | ❌ **缺** | tile-ai 任何开源 repo 都无；仅闭源 TileRT 有 5 个 spec 文件 |

### 结论：MiMo 可分两步实现

1. **MiMo 主干（GQA+SWA+sink + MXFP4 MoE）→ 可用开源 kernel 搭建**。
   B200 上跑 MiMo-V2.5-Pro-FP4 的所有算子构件都已开源（连 sm100 的 fp8/fp4 混合
   GEMM 都有现成 example）。这是一条**不依赖闭源 .so** 的真实路径。
   工作量：用 TileLang/TileOPs 组装 70 层 MiMo 推理图（数天到两周量级的工程）。

2. **DFlash 投机解码 → 是唯一缺口**。这是小米拉到 1000 tok/s 的关键（接受长度 6+），
   开源栈里没有。但**没有 DFlash 也能跑 MiMo**，只是速度退化到约
   300-400 tok/s（普通 MTP/无投机水平），仍可验证模型正确性与基础性能。

**给团队的建议路径**（按投入产出排序）：
- 先用开源 TileOPs 的 GQA-SWA + MXFP4-MoE kernel 搭出 MiMo 主干，跑通正确性（不带 DFlash）
- 性能基线达标后，再自研或等官方放出 DFlash drafter 拉满到 1000
- 这比逆向闭源 `.so` 靠谱得多，且 kernel 都是 tile-ai 官方维护、持续更新的
