"""Verify tilelang MXFP4 dequant-GEMM against the PyTorch reference on a real
MiMo expert weight, and measure speedup. Foundation for replacing the slow
dequant-on-use MoE with a fused MXFP4 kernel.

Run inside the cu128 tilelang container (tl128):
    python verify_mxfp4_kernel.py
"""
import json, os, sys, time
import torch
from safetensors import safe_open

sys.path.insert(0, "/nvme/tilelang_examples")  # where we copy the example
from mimo_quant import dequant_mxfp4

CKPT = "/nvme/models/MiMo-V2.5-Pro-FP4-DFlash"


def get(idx, key):
    with safe_open(os.path.join(CKPT, idx[key]), framework="pt") as f:
        return f.get_tensor(key)


def main():
    idx = json.load(open(os.path.join(CKPT, "model.safetensors.index.json")))["weight_map"]
    # gate_proj of layer1 expert0: packed uint8 [2048, 3072] (=> N=2048, K=6144),
    # scale uint8 [2048, 192] (192 = 6144/32)
    w = get(idx, "model.layers.1.mlp.experts.0.gate_proj.weight").cuda()    # [N, K/2] u8
    s = get(idx, "model.layers.1.mlp.experts.0.gate_proj.weight_scale").cuda()  # [N, K/32] u8
    N, Khalf = w.shape
    K = Khalf * 2
    print(f"expert gate_proj: packed weight {tuple(w.shape)} (N={N}, K={K}), scale {tuple(s.shape)}")

    # reference: PyTorch dequant -> bf16 weight, then matmul
    x = torch.randn(8, K, device="cuda", dtype=torch.bfloat16)  # 8 tokens
    w_bf16 = dequant_mxfp4(w, s, block_size=32, out_dtype=torch.bfloat16)  # [N, K]
    print(f"dequant weight: {tuple(w_bf16.shape)} std {float(w_bf16.float().std()):.5f}")
    torch.cuda.synchronize(); t = time.time()
    for _ in range(50):
        ref = torch.nn.functional.linear(x, w_bf16)  # [8, N]
    torch.cuda.synchronize()
    ref_ms = (time.time() - t) / 50 * 1000
    # but dequant itself is the real cost in the current MoE — time it too
    torch.cuda.synchronize(); t = time.time()
    for _ in range(50):
        wd = dequant_mxfp4(w, s, 32, torch.bfloat16)
        _ = torch.nn.functional.linear(x, wd)
    torch.cuda.synchronize()
    dequant_plus_gemm_ms = (time.time() - t) / 50 * 1000
    print(f"PyTorch: gemm-only {ref_ms:.3f}ms | dequant+gemm {dequant_plus_gemm_ms:.3f}ms (this is the MoE bottleneck)")

    # tilelang fused mxfp4 matmul
    try:
        import tilelang
        import tilelang.language as T
        from example_dequant_gemm_bf16_mxfp4_hopper import matmul
        M = 8
        kernel = matmul(M, N, K, T.bfloat16, T.bfloat16, T.float32, num_bits=4,
                        scale_size=32, block_M=64, block_N=128, block_K=128,
                        num_stages=2, threads=256, split=1, fast_dequant=False)
        # kernel signature is (A, B, Scale, Bias, ->C); Bias required even when unused
        bias = torch.zeros(M, N, device="cuda", dtype=torch.bfloat16)  # Bias_shape=(M,N)
        out = kernel(x, w, s.to(torch.uint8), bias)
        torch.cuda.synchronize(); t = time.time()
        for _ in range(50):
            out = kernel(x, w, s.to(torch.uint8), bias)
        torch.cuda.synchronize()
        tl_ms = (time.time() - t) / 50 * 1000
        diff = (out.float() - ref.float()).abs()
        rel = diff.mean() / ref.float().abs().mean()
        print(f"tilelang fused mxfp4 gemm: {tl_ms:.3f}ms | rel_err {float(rel):.4f}")
        print(f"SPEEDUP vs dequant+gemm: {dequant_plus_gemm_ms/tl_ms:.1f}x")
    except Exception as e:
        import traceback; traceback.print_exc()
        print("tilelang kernel path failed:", str(e)[:200])


if __name__ == "__main__":
    main()
