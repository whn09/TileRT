"""MiMo-V2.5-Pro model family for TileRT.

This package wires Xiaomi's **MiMo-V2.5-Pro-UltraSpeed** (the 1T-parameter,
MoE-FP4 + DFlash model that powers the >1000 tok/s demo) into TileRT's
backend-selection / generator / weight-conversion plumbing.

STATUS — skeleton only (added on the ``feat/mimo-kimi-backend`` branch).
The compute kernels for MiMo live in a backend shared library
(``libtilert_mimo.so``) that Xiaomi/TileRT have **not yet open-sourced**.
Until that ``.so`` ships, the Python layer here is complete and self-consistent
(model args, CLI wiring, weight-converter entry, benchmark hooks) but the decode
layer raises a clear error at the kernel boundary instead of pretending to run.

Architecture note: MiMo uses **GQA + sliding-window (SWA) hybrid attention**,
which is structurally different from the MLA/DSA attention used by the existing
``deepseek_v3_2`` and ``glm_5`` backends — so the ``modules`` / ``ops`` layer
cannot simply be copied from those families; it must match the kernels that the
official ``.so`` exposes.
"""

from tilert.models.mimo_v2.model_args import ModelArgsMiMoV2

__all__ = [
    "ModelArgsMiMoV2",
]
