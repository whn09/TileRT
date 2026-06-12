# MiMo-V2.5-Pro & Kimi-K2.6 — TileRT backend skeletons

> Branch: `feat/mimo-kimi-backend`. **Skeleton only — does not run end-to-end yet.**

## TL;DR

TileRT's actual compute lives in closed-source backend libraries
(`libtilert_dsv32.so`, `libtilert_glm5.so`). The Python in this repo is a thin
orchestration shell over `torch.ops.tilert.*` kernels. The MiMo `.so`
(`libtilert_mimo.so`) — the one that powers the >1000 tok/s
MiMo-V2.5-Pro-UltraSpeed demo — is **not open-sourced**. So no amount of Python
makes MiMo or Kimi decode correctly today.

This branch adds everything *around* that missing `.so`, so the moment it ships
(or is handed to us), wiring it up is a small, well-marked task:

| Layer | MiMo | Kimi-K2.6 | Status |
|---|---|---|---|
| `ModelArgs` (from real HF config) | `mimo_v2/model_args.py` | `kimi_k2/model_args.py` | ✅ done |
| Generator (CLI/benchmark parity) | `mimo_v2/generator.py` | `kimi_k2/generator.py` | ✅ done |
| Decode layer (kernel boundary) | `mimo_v2/modules/end2end.py` | reuses DeepSeek path | ⛔ guarded |
| Op surface (documented) | `mimo_v2/ops/__init__.py` | — | ✅ sketched |
| Backend registry | `tilert/__init__.py` | `tilert/__init__.py` | ✅ done |
| CLI `--model` | `generate.py` | `generate.py` | ✅ done |
| Weight converter entry | `weight_converter.py` | `weight_converter.py` | MiMo ⛔ / Kimi ✅* |
| Preflight check | `scripts/preflight_mimo_kimi.py` | same | ✅ done |

\* Kimi reuses the DeepSeek MLA converter (same attention family); MiMo needs a
dedicated GQA+MXFP4 converter (raises `NotImplementedError` with the spec).

## Why MiMo ≠ Kimi in difficulty

Pulled from the published HF `config.json`s:

- **MiMo-V2.5-Pro** — **GQA + sliding-window (SWA) hybrid** attention
  (128/8 heads, window 128, attention-sink bias), 70 layers, 384 experts,
  **MoE→MXFP4 (block 32)** with every `o_proj` excluded, plus a **BF16 DFlash**
  block-diffusion drafter. None of the existing MLA/DSA kernels apply — MiMo
  needs all-new attention + MXFP4-MoE kernels in its `.so`.
- **Kimi-K2.6** — text backbone is **MLA** (q_lora 1536 / kv_lora 512), 61
  layers, 384 experts, sigmoid `noaux_tc` routing. This is the *same family as
  DeepSeek-V3.2*, so the existing `libtilert_dsv32.so` may run it directly.
  Note: the open checkpoint quantizes experts to **INT4** (compressed-tensors,
  group 32), **not NVFP4** — `moe_quant` is configurable for either.

## Next-week on-device checklist

1. `python scripts/preflight_mimo_kimi.py --model-weights-dir <dir>` — confirms
   8×B200, backend registration, and that the CLI path fails *only* at the
   missing-kernel boundary (no import/attr errors).
2. **Kimi path A (cheapest test):** convert Kimi weights
   (`weight_converter --model_type kimi-k2.6`), then
   `TILERT_KIMI_REUSE_DSV32=1 python -m tilert.generate --model kimi_k2 ...`.
   If the DeepSeek MLA kernels accept Kimi's shapes, you get real numbers today.
   **Verify output correctness before trusting throughput.**
3. **MiMo:** blocked on `libtilert_mimo.so`. Once obtained: drop it in `tilert/`,
   implement the kernel calls in `mimo_v2/modules/end2end.py` against the ops it
   registers (see `mimo_v2/ops/__init__.py`), and write the MiMo branch of
   `WeightConverter`.
