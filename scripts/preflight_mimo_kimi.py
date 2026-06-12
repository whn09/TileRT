#!/usr/bin/env python3
"""On-device preflight for the MiMo / Kimi skeleton backends.

Run this FIRST on the B200 box next week. It checks everything that does NOT
require the unreleased compute kernels, so the only thing left to verify once a
backend ``.so`` is dropped in is the actual decode path.

What it validates:
  1. Hardware: 8 GPUs visible, each reporting as B200.
  2. Backend registry: mimo_v2 / kimi_k2 are registered in tilert._BACKENDS.
  3. CLI/generator wiring: get_generator(...) reaches the right code path and
     fails with a CLEAR, expected error (missing .so / not-wired) rather than an
     import or attribute error.
  4. ModelArgs match the published HF configs.

Usage:
    python scripts/preflight_mimo_kimi.py --model-weights-dir /path/to/weights
    python scripts/preflight_mimo_kimi.py            # skips weight-dependent checks
"""

from __future__ import annotations

import argparse
import sys


def _check_hardware() -> bool:
    try:
        import torch
    except ImportError:
        print("  [SKIP] torch not importable (not on the target box?)")
        return False
    if not torch.cuda.is_available():
        print("  [FAIL] CUDA not available")
        return False
    n = torch.cuda.device_count()
    print(f"  GPUs visible: {n}")
    ok = n == 8
    if not ok:
        print(f"  [WARN] expected 8 GPUs for the 1000 tok/s config, found {n}")
    for i in range(n):
        name = torch.cuda.get_device_name(i)
        flag = "OK" if "B200" in name else "??"
        print(f"    cuda:{i}: {name} [{flag}]")
        if "B200" not in name:
            ok = False
    return ok


def _check_registry() -> bool:
    import tilert

    backends = getattr(tilert, "_BACKENDS", {})
    ok = True
    for mt, so in [("mimo_v2", "libtilert_mimo.so"), ("kimi_k2", "libtilert_kimi.so")]:
        present = backends.get(mt) == so
        print(f"  registry[{mt}] -> {backends.get(mt)} {'OK' if present else 'MISSING'}")
        ok = ok and present
    return ok


def _check_model_args() -> bool:
    from tilert.models.kimi_k2.model_args import ModelArgsKimiK2
    from tilert.models.mimo_v2.model_args import ModelArgsMiMoV2

    m, k = ModelArgsMiMoV2(), ModelArgsKimiK2()
    checks = [
        ("MiMo n_layers", m.n_layers, 70),
        ("MiMo n_routed_experts", m.n_routed_experts, 384),
        ("MiMo moe_quant", m.moe_quant, "mxfp4"),
        ("MiMo n_kv_heads (GQA)", m.n_kv_heads, 8),
        ("Kimi n_layers", k.n_layers, 61),
        ("Kimi q_lora_rank (MLA)", k.q_lora_rank, 1536),
        ("Kimi n_routed_experts", k.n_routed_experts, 384),
        ("Kimi moe_quant (NVFP4 build)", k.moe_quant, "nvfp4"),
        ("Kimi moe_quant_block_size", k.moe_quant_block_size, 16),
    ]
    ok = True
    for name, got, want in checks:
        good = got == want
        print(f"  {name}: {got} {'OK' if good else f'(expected {want})'}")
        ok = ok and good
    return ok


def _check_generator_path(model: str, weights_dir: str | None) -> bool:
    """Confirm the CLI path reaches the backend and fails with the EXPECTED error."""
    if not weights_dir:
        print(f"  [SKIP] {model}: no --model-weights-dir (tokenizer load needed)")
        return True
    from tilert.generate import get_generator

    try:
        get_generator(
            model_type=model,
            max_new_tokens=8,
            temperature=1.0,
            model_weights_dir=weights_dir,
            with_mtp=False,
        )
        print(f"  [UNEXPECTED] {model}: generator built without a backend .so?!")
        return False
    except RuntimeError as e:
        # Expected: missing .so (load_backend) or skeleton guard.
        msg = str(e).splitlines()[0]
        print(f"  {model}: failed as expected at backend boundary -> {msg[:80]}")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  [FAIL] {model}: unexpected error type {type(e).__name__}: {e}")
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-weights-dir", default=None)
    args = ap.parse_args()

    results: dict[str, bool] = {}
    print("== 1. Hardware ==")
    results["hardware"] = _check_hardware()
    print("== 2. Backend registry ==")
    results["registry"] = _check_registry()
    print("== 3. Model args vs HF config ==")
    results["model_args"] = _check_model_args()
    print("== 4. Generator wiring ==")
    results["mimo_path"] = _check_generator_path("mimo_v2", args.model_weights_dir)
    results["kimi_path"] = _check_generator_path("kimi_k2", args.model_weights_dir)

    print("\n== Summary ==")
    for name, ok in results.items():
        print(f"  {name:12} {'PASS' if ok else 'CHECK'}")
    # Hardware/weights may legitimately be skipped off-box; only wiring must pass.
    critical = ["registry", "model_args", "mimo_path", "kimi_path"]
    return 0 if all(results[c] for c in critical) else 1


if __name__ == "__main__":
    sys.exit(main())
