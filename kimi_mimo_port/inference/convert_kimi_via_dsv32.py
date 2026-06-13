"""Convert Kimi-K2-Instruct using the DeepSeek-V3.2 (dsv32) TileRT converter,
by injecting DUMMY indexer weights so the DSA pipeline accepts it.

Rationale: Kimi-K2 is standard MLA (DeepSeek-V3 family) WITHOUT the DSA sparse
indexer. DeepSeek-V3.2's DSA selects top-`index_topk` (2048) positions; when the
sequence length <= 2048, top-k selects ALL positions == full attention. The
indexer's topk_indices are only used to build an attention mask
(scatter_(-1, topk_indices, 0)), NOT for weighting — so dummy indexer weights
are harmless for short sequences (<=2048 total tokens), yielding correct
standard-MLA output while reusing the FAST closed-source dsv32 .so engine.

Indexer tensors injected per layer (shapes from DeepSeek-V3.2, dim=7168,
q_lora=1536 — identical to Kimi):
  indexer.k_norm.{weight,bias}            f32  [128]
  indexer.weights_proj.weight             bf16 [64, 7168]
  indexer.wk.weight                       fp8  [128, 7168]
  indexer.wk.weight_scale_inv             f32  [1, 56]
  indexer.wq_b.weight                     fp8  [8192, 1536]
  indexer.wq_b.weight_scale_inv           f32  [64, 12]

Usage (inside tilert cu13 container, where the dsv32 .so lives):
  python convert_kimi_via_dsv32.py --model_dir /nvme/models/Kimi-K2-Instruct \
      --save_dir /nvme/weights/Kimi-K2-dsv32 [--test_mode]
"""
import argparse
import torch

from tilert.models.preprocess import weight_converter as wc
from tilert.models.deepseek_v3_2.model_args import ModelArgs as DSAv32ModelArgs


_INDEXER_SPEC = {
    "self_attn.indexer.k_norm.weight": ((128,), torch.float32, "ones"),
    "self_attn.indexer.k_norm.bias": ((128,), torch.float32, "zeros"),
    "self_attn.indexer.weights_proj.weight": ((64, 7168), torch.bfloat16, "zeros"),
    "self_attn.indexer.wk.weight": ((128, 7168), torch.float8_e4m3fn, "zeros"),
    "self_attn.indexer.wk.weight_scale_inv": ((1, 56), torch.float32, "ones"),
    "self_attn.indexer.wq_b.weight": ((8192, 1536), torch.float8_e4m3fn, "zeros"),
    "self_attn.indexer.wq_b.weight_scale_inv": ((64, 12), torch.float32, "ones"),
}


def _inject_dummy_indexer(weights_dict, layer_idx, device):
    prefix = f"model.layers.{layer_idx}."
    for suf, (shape, dtype, fill) in _INDEXER_SPEC.items():
        key = prefix + suf
        if key in weights_dict:
            continue
        if fill == "ones":
            t = torch.ones(shape, dtype=dtype, device=device)
        else:
            t = torch.zeros(shape, dtype=dtype, device=device)
        weights_dict[key] = t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--save_dir", required=True)
    ap.add_argument("--test_mode", action="store_true")
    args = ap.parse_args()

    model_args = DSAv32ModelArgs()  # Kimi-K2 shares DeepSeek-V3.2 MLA dims
    # Kimi-K2 has 64 attention heads (DeepSeek-V3.2 has 128); all other MLA dims
    # match (qk_nope 128, qk_rope 64, v 128, q_lora 1536, kv_lora 512).
    model_args.n_heads = 64
    model_args.vocab_size = 163840  # Kimi vocab
    model_args.n_routed_experts = 384  # Kimi has 384 experts (DeepSeek 256)
    model_args.n_activated_experts = 8
    model_args.n_dense_layers = 1  # Kimi first_k_dense_replace=1 (DeepSeek=3)
    print(f"Kimi ModelArgs: n_heads={model_args.n_heads}, experts={model_args.n_routed_experts}, vocab={model_args.vocab_size}")

    converter = wc.WeightConverter(model_args, 8, args.model_dir, args.save_dir, args.test_mode)
    # Kimi-K2-Instruct has NO MTP/nextn layer (DeepSeek-V3.2 has layer 61 MTP).
    # Drop it so the converter doesn't try to load a non-existent layer.
    converter.num_mtp_layers = 0
    converter.num_dense_layers = 1  # Kimi first_k_dense_replace=1
    converter.num_moe_layers = model_args.n_layers - converter.num_dense_layers
    converter.total_layers = converter.num_dense_layers + converter.num_moe_layers
    if args.test_mode:
        converter.target_layers = [0, converter.num_dense_layers, converter.total_layers - 1]
    else:
        converter.target_layers = list(range(converter.total_layers))
    print(f"total_layers={converter.total_layers} (MTP dropped), targets={converter.target_layers if args.test_mode else 'all'}")

    # monkeypatch convert_a_layer to inject dummy indexer right after load
    orig = converter.convert_a_layer.__func__

    def patched(self, layer_idx):
        # replicate the original load, then inject, then call the rest.
        # Simplest: wrap load_file path by pre-populating via transform hooks is
        # hard; instead re-implement the load+inject+transform sequence.
        import os
        from safetensors.torch import load_file
        key = f"layer_{layer_idx}"
        weights_dict = {}
        for fn in self.files_by_layers[key]:
            weights_dict.update(load_file(os.path.join(self.model_dir, fn), device=self.default_device))
        _inject_dummy_indexer(weights_dict, layer_idx, self.default_device)
        mla = self.transform_mla(weights_dict, layer_idx)
        if layer_idx < self.num_dense_layers:
            mlp = self.transform_mlp(weights_dict, layer_idx)
        else:
            mlp = self.transform_moe(weights_dict, layer_idx)
        mtp = {f"dev_{d}": {} for d in range(self.num_devices)}
        if layer_idx >= self.num_dense_layers + self.num_moe_layers:
            mtp = self.transform_mtp(weights_dict, layer_idx)
        return mla, mlp, mtp

    import types
    converter.convert_a_layer = types.MethodType(patched, converter)
    converter.to_tilert_weights()
    print("Kimi-K2 -> dsv32 TileRT conversion done (dummy indexer injected).")


if __name__ == "__main__":
    main()
