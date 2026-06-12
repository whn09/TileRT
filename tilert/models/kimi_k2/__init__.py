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
  2. Quantization: two checkpoints exist, differing only in MoE-expert format —
       - ``nvidia/Kimi-K2.6-NVFP4``  -> NVFP4 (float4, group 16)  [B200 target]
       - ``moonshotai/Kimi-K2.6``    -> INT4 (compressed-tensors, group 32)
     Both keep self_attn / shared_experts / lm_head / layer 0 in higher
     precision. The default ``moe_quant`` is NVFP4; the backend must expose a
     matching NVFP4 expert GEMM. NVIDIA exports the NVFP4 build with
     ``model_type == "deepseek_v3"``, reinforcing the DeepSeek-backend reuse path.
"""

from tilert.models.kimi_k2.model_args import ModelArgsKimiK2

__all__ = [
    "ModelArgsKimiK2",
]
