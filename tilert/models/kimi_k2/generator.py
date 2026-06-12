"""Generator for Kimi-K2.6 (text backbone).

Because Kimi-K2.6's backbone is MLA — the same family as DeepSeek-V3.2 — the
intent is to reuse the DeepSeek decode path as much as possible. This skeleton
therefore subclasses behavior from the DeepSeek generator where it can, and
isolates the one hard dependency (a working backend ``.so`` + matching weight
layout) behind a clear guard.

Two on-device outcomes are possible and the code is structured for both:
  (A) ``libtilert_dsv32.so`` can run Kimi directly given Kimi ModelArgs and
      DeepSeek-compatible sharded weights -> this generator just delegates to
      ``DSAv32Generator`` with ``ModelArgsKimiK2``.
  (B) Kimi needs its own ``libtilert_kimi.so`` -> implement a dedicated decode
      layer like MiMo's. The guard below makes the missing piece explicit.
"""

from __future__ import annotations

from tilert import logger
from tilert.models.kimi_k2.model_args import ModelArgsKimiK2

__all__ = [
    "KimiK2Generator",
    "build_kimi_generator",
]


class KimiBackendNotWiredError(RuntimeError):
    """Raised until the Kimi backend mapping is confirmed on real hardware."""


_GUIDANCE = (
    "Kimi-K2.6 backend is not yet wired. Kimi's text backbone is MLA (same family "
    "as DeepSeek-V3.2), so the most likely path is to reuse the DeepSeek-V3.2 "
    "kernels:\n"
    "  1. Convert Kimi weights with `weight_converter --model_type kimi-k2.6` "
    "(emits the 8-shard DeepSeek-compatible layout; MoE experts stay INT4/NVFP4 "
    "per ModelArgsKimiK2.moe_quant).\n"
    "  2. On-device, try loading the deepseek_v3_2 backend with ModelArgsKimiK2:\n"
    "       tilert.load_backend('deepseek_v3_2')\n"
    "       gen = DSAv32Generator(model_args=ModelArgsKimiK2(), ...)\n"
    "     If the MLA kernels accept Kimi's shapes (q_lora=1536, kv_lora=512, "
    "384 experts), generation works as-is.\n"
    "  3. If shapes/quant are rejected (e.g. INT4/NVFP4 expert GEMM missing), a "
    "dedicated libtilert_kimi.so is required; implement a decode layer mirroring "
    "tilert/models/mimo_v2/modules/end2end.py.\n"
    "Set TILERT_KIMI_REUSE_DSV32=1 to attempt path (A) automatically."
)


def build_kimi_generator(
    model_args: ModelArgsKimiK2,
    *,
    max_new_tokens: int,
    temperature: float,
    model_weights_dir: str,
    with_mtp: bool,
    top_p: float,
    top_k: int,
    enable_thinking: bool,
    sampling_seed: int,
):  # noqa: ANN201
    """Construct a generator for Kimi-K2.6.

    Attempts the DeepSeek-V3.2 reuse path (A) when ``TILERT_KIMI_REUSE_DSV32=1``;
    otherwise raises with actionable guidance so the on-device step is explicit.
    """
    import os

    if os.environ.get("TILERT_KIMI_REUSE_DSV32") == "1":
        # Path (A): drive Kimi through the DeepSeek MLA backend.
        # NB: the caller must have loaded the deepseek_v3_2 backend already and
        # converted Kimi weights into the DeepSeek-compatible 8-shard layout.
        from tilert.models.deepseek_v3_2.generator import DSAv32Generator

        logger.warning(
            "Kimi-K2.6: attempting to reuse the DeepSeek-V3.2 MLA backend "
            "(TILERT_KIMI_REUSE_DSV32=1). This is EXPERIMENTAL and must be "
            "validated on-device — verify output correctness before trusting "
            "any throughput numbers."
        )
        return DSAv32Generator(
            model_args=model_args,  # type: ignore[arg-type]
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            model_weights_dir=model_weights_dir,
            with_mtp=with_mtp,
            top_p=top_p,
            top_k=top_k,
            use_topp=top_p < 1.0,
            sampling_seed=sampling_seed,
            enable_thinking=enable_thinking,
        )

    raise KimiBackendNotWiredError(_GUIDANCE)


# Alias kept so `from ... import KimiK2Generator` works symmetrically with the
# other model families. Resolves to the DeepSeek generator under path (A).
KimiK2Generator = build_kimi_generator
