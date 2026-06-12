# Kimi-K2 / MiMo 移植到开源 TileLang 推理栈

> 目标：在**不依赖闭源 `libtilert_*.so`** 的前提下，让 Kimi-K2（及最终 MiMo）
> 在 B200 上跑出正确输出。基于 tile-ai 开源的
> `tilelang/examples/deepseek_v32/inference/`（完整 PyTorch + tilelang kernel 推理栈）改造。

## 为什么这条路可行

- 闭源 TileRT 引擎（`dsa_show_hands`）把 DeepSeek-V3.2 的 DSA 稀疏注意力编译死，
  无法跑标准 MLA 的 Kimi（详见仓库根 `B200_PERFORMANCE_REPORT.md` 附 A）。
- 但 tilelang 开源了一个**可改的 PyTorch 推理 example**，Kimi-K2 是 DeepSeek-V3 同族
  （标准 MLA，无 DSA/indexer），只需关掉 indexer + 改超参即可。

## 改动清单（`inference/` 目录）

相对 tilelang 原版 `examples/deepseek_v32/inference/`：

1. **`model.py`**
   - `ModelArgs` 新增 `use_indexer: bool = True`
   - `MLA.__init__`：`use_indexer=False` 时不构造 `Indexer`
   - `MLA.forward`：`use_indexer=False` 时跳过 DSA indexer，退化为标准 full-causal MLA
     （prefill 只加因果 mask；decode attend 全部历史位置）。数学上 full attention 是
     sparse 的超集，正确。
2. **`convert.py`**
   - 未知权重 key 从 `assert` 改为"跳过并打印"，容忍 Kimi 多出/缺失的 key
3. **`config_kimi_k2.json`**（新增）
   - Kimi-K2-Instruct 超参：vocab 163840 / 61层 / 1 dense层 / 64 头 / 384专家 /
     n_group=1 / route_scale 2.827 / sigmoid / rope_theta 50000 + YaRN(factor32) /
     `use_indexer: false`

## 跑法（在 P6-1 B200，干净容器 `tlexp` = sglang:dev-cu13 + tilelang）

```bash
# 1. 转换 Kimi-K2 权重 (mp=8)
cd inference
export EXPERTS=384
python convert.py --hf-ckpt-path /nvme/models/Kimi-K2-Instruct \
  --save-path /nvme/weights/Kimi-K2-tilelang --n-experts $EXPERTS --model-parallel 8

# 2. 跑生成
torchrun --nproc-per-node 8 generate.py \
  --ckpt-path /nvme/weights/Kimi-K2-tilelang \
  --config config_kimi_k2.json --interactive
```

## 状态

- [x] 代码改造完成（model.py / convert.py / config）
- [ ] 转换 Kimi 权重
- [ ] 跑通正确性
- [ ] 性能基线

## 注意

- 这是 **example 级 PyTorch 推理（几十 tok/s）**，不是 TileRT 闭源引擎的高性能（200-600 tok/s）。
  目标是"正确跑通 + 验证开源栈支持新模型可行"，为 MiMo 铺路。
- MiMo 阶段还需 TileOPs 的 `gqa_sliding_window_fwd` + MXFP4 kernel（DeepSeek MLA example 里没有），
  那是下一步。
