"""End-to-end decode layer for MiMo-V2.5-Pro.

This is the boundary between the (open-source) Python orchestration and the
(not-yet-released) ``libtilert_mimo.so`` compute kernels. The class shape mirrors
``glm_5._dsa_v32.modules.end2end.ShowHandsDSALayer`` so the generator and
benchmark code are model-agnostic, but every method that would invoke a backend
kernel raises :class:`BackendNotAvailableError` with an actionable message
instead of silently producing wrong results.

When ``libtilert_mimo.so`` becomes available, implement the bodies below against
the ops it registers (GQA + SWA attention, MXFP4 MoE GEMM, DFlash verify) and
delete the guards. The expected op surface is sketched in
``tilert/models/mimo_v2/ops/__init__.py``.
"""

from __future__ import annotations

import torch

from tilert import logger
from tilert.models.mimo_v2.model_args import ModelArgsMiMoV2


class BackendNotAvailableError(RuntimeError):
    """Raised when a MiMo backend kernel is invoked but the .so is missing."""


_MISSING_BACKEND_MSG = (
    "MiMo-V2.5-Pro requires the compiled backend 'libtilert_mimo.so', which is "
    "NOT bundled in this open-source build. The Python orchestration "
    "(model args, generator, CLI, weight converter, benchmark hooks) is complete "
    "and ready, but the decode kernels (GQA+SWA attention, MXFP4 MoE GEMM, DFlash "
    "verify) live in that shared library.\n"
    "  -> Once Xiaomi/TileRT publish libtilert_mimo.so, drop it into the tilert/ "
    "package dir, register it in tilert/__init__.py:_BACKENDS, and implement the "
    "kernel calls in tilert/models/mimo_v2/modules/end2end.py.\n"
    "  -> Until then this path intentionally fails fast rather than returning "
    "incorrect output."
)


class ShowHandsMiMoLayer:
    """Persistent decode layer for MiMo. Skeleton — kernels not yet available.

    The constructor records shapes/config and pre-computes the per-device shard
    layout (TP=8, one shard per B200) so weight loading is ready, but it does
    not touch any backend op. Methods that need kernels raise immediately.
    """

    def __init__(
        self,
        model_args: ModelArgsMiMoV2,
        model_path: str,
        with_mtp: bool = False,
        top_p: float = 0.9,
        top_k: int = 256,
        use_topp: bool = False,
        num_devices: int = 8,
    ):
        self.config = model_args
        self.model_path = model_path
        self.with_mtp = with_mtp
        self.top_p = top_p
        self.top_k = top_k
        self.use_topp = use_topp
        self.num_devices = num_devices
        self._dsa_objects: list[object | None] = [None] * num_devices
        logger.warning(
            "ShowHandsMiMoLayer constructed in SKELETON mode "
            "(num_devices=%d, with_mtp=%s). Backend kernels are unavailable; "
            "forward()/from_pretrained() will raise until libtilert_mimo.so ships.",
            num_devices,
            with_mtp,
        )

    # ---- weight loading -------------------------------------------------
    def from_pretrained(self, model_weights_dir: str) -> None:
        raise BackendNotAvailableError(_MISSING_BACKEND_MSG)

    def init_random_weights(self) -> None:
        raise BackendNotAvailableError(_MISSING_BACKEND_MSG)

    # ---- decode ---------------------------------------------------------
    def forward(self, tokens: torch.Tensor, with_mtp: bool = False):  # noqa: ANN201
        raise BackendNotAvailableError(_MISSING_BACKEND_MSG)

    def prefill(self, tokens: torch.Tensor, block_size: int) -> None:
        raise BackendNotAvailableError(_MISSING_BACKEND_MSG)

    def extract_next_token(self, results) -> torch.Tensor:  # noqa: ANN001
        raise BackendNotAvailableError(_MISSING_BACKEND_MSG)

    # ---- DFlash speculative decoding -----------------------------------
    def get_next_draft_tokens(self, device_id: int) -> torch.Tensor:
        raise BackendNotAvailableError(_MISSING_BACKEND_MSG)

    def get_num_accepted(self, device_id: int) -> int:
        raise BackendNotAvailableError(_MISSING_BACKEND_MSG)

    def get_predicted_tokens(self, device_id: int) -> torch.Tensor:
        raise BackendNotAvailableError(_MISSING_BACKEND_MSG)

    # ---- runtime state (safe no-ops where possible) --------------------
    def set_cur_pos(self, cur_pos: int) -> None:
        # No backend state to set in skeleton mode; record for parity.
        self._cur_pos = cur_pos

    def set_sampling_seed(self, seed: int, with_mtp: bool = False) -> None:
        self._sampling_seed = seed

    def update_sampling_config(
        self, temperature: float, top_p: float, top_k: int, use_topp: bool
    ) -> None:
        self.top_p, self.top_k, self.use_topp = top_p, top_k, use_topp

    def reset_sequence(self) -> None:
        pass

    def cleanup(self) -> None:
        pass
