"""Kimi-K2.6 model family for TileRT.

Wires Moonshot AI's **Kimi-K2.6** into TileRT's backend/generator/converter
plumbing. The text backbone (HF ``text_config.model_type == "kimi_k2"``) is an
**MLA** MoE — the *same architecture family* as DeepSeek-V3.2 — with q_lora 1536
/ kv_lora 512, 384 routed experts, sigmoid routing (noaux_tc). This means the
existing ``deepseek_v3_2`` MLA/DSA kernels are the natural starting point for a
Kimi backend, unlike MiMo (GQA+SWA), which needs all-new attention kernels.

STATUS — skeleton only (``feat/mimo-kimi-backend`` branch).
Two things are still required before this runs on real hardware:
  1. A compiled backend for Kimi. It may turn out that the existing
     ``libtilert_dsv32.so`` can serve Kimi directly (shared MLA kernels), in
     which case the generator just needs the right ModelArgs + weight layout;
     otherwise a ``libtilert_kimi.so`` is needed. To be confirmed on-device.
  2. Quantization: the open-source Kimi-K2.6 checkpoint is **INT4**
     (compressed-tensors, group 32) on the MoE experts — NOT NVFP4. If an
     NVFP4 variant is the target, the weight converter must emit NVFP4 blocks
     and the backend must expose an NVFP4 expert GEMM. See ``model_args`` /
     ``weight_converter`` for the configurable ``moe_quant`` field.
"""

from tilert.models.kimi_k2.model_args import ModelArgsKimiK2

__all__ = [
    "ModelArgsKimiK2",
]
