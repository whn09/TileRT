"""MiMo-V2.5-Pro generation (example-level, correctness-first).

Loads HF weights directly (no separate convert step): FP8 weights go to the
DeepSeek-scaffold Linear (.weight + .scale), MXFP4 expert weights go to the
MXFP4Expert uint8 buffers, and experts are sharded across ranks. Run with
torchrun --nproc-per-node 8.
"""

import os
import json
import time
from argparse import ArgumentParser

import torch
import torch.distributed as dist
from safetensors import safe_open
from transformers import AutoTokenizer

import model as ds
from mimo_model import MiMoArgs, MiMoTransformer


def load_config(path):
    c = json.load(open(os.path.join(path, "config.json")))
    a = MiMoArgs()
    a.vocab_size = c["vocab_size"]; a.dim = c["hidden_size"]
    a.inter_dim = c["intermediate_size"]; a.moe_inter_dim = c["moe_intermediate_size"]
    a.n_layers = c["num_hidden_layers"]
    a.num_attention_heads = c["num_attention_heads"]; a.num_key_value_heads = c["num_key_value_heads"]
    a.head_dim = c["head_dim"]; a.v_head_dim = c["v_head_dim"]
    a.attention_value_scale = c.get("attention_value_scale") or 1.0
    a.rope_theta = c["rope_theta"]; a.partial_rotary_factor = c.get("partial_rotary_factor", 1.0)
    a.swa_num_attention_heads = c.get("swa_num_attention_heads") or a.num_attention_heads
    a.swa_num_key_value_heads = c.get("swa_num_key_value_heads") or a.num_key_value_heads
    a.swa_head_dim = c.get("swa_head_dim") or a.head_dim
    a.swa_v_head_dim = c.get("swa_v_head_dim") or a.v_head_dim
    a.swa_rope_theta = c.get("swa_rope_theta") or a.rope_theta
    a.sliding_window = c.get("sliding_window") or c.get("sliding_window_size") or 128
    a.add_full_attention_sink_bias = c.get("add_full_attention_sink_bias", False)
    a.add_swa_attention_sink_bias = c.get("add_swa_attention_sink_bias", False)
    a.hybrid_layer_pattern = c.get("hybrid_layer_pattern", [])
    a.n_routed_experts = c["n_routed_experts"]; a.num_experts_per_tok = c["num_experts_per_tok"]
    a.n_activated_experts = c["num_experts_per_tok"]
    a.score_func = c.get("scoring_func", "sigmoid")
    a.norm_topk_prob = c.get("norm_topk_prob", True)
    a.route_scale = c.get("routed_scaling_factor") or 1.0
    qc = c.get("quantization_config", {})
    a.mxfp4_block_size = qc.get("mxfp4_block_size", 32)
    a.eps = c.get("layernorm_epsilon", 1e-5)
    # first dense layer count from moe_layer_freq (leading zeros)
    mf = c.get("moe_layer_freq")
    if isinstance(mf, list):
        a.n_dense_layers = next((i for i, v in enumerate(mf) if v), len(mf))
    return a


def build_index(path):
    idx = json.load(open(os.path.join(path, "model.safetensors.index.json")))["weight_map"]
    return idx


def get_tensor(path, idx, key, cache):
    f = idx.get(key)
    if f is None:
        return None
    if f not in cache:
        cache[f] = safe_open(os.path.join(path, f), framework="pt", device="cpu")
    return cache[f].get_tensor(key)


def main(ckpt, max_new_tokens, prompt):
    ws = int(os.getenv("WORLD_SIZE", 1)); rank = int(os.getenv("RANK", 0))
    lr = int(os.getenv("LOCAL_RANK", 0))
    if ws > 1:
        dist.init_process_group("nccl")
    torch.cuda.set_device(lr)
    torch.set_default_dtype(torch.bfloat16)
    if rank != 0:
        import builtins; builtins.print = lambda *a, **k: None

    args = load_config(ckpt)
    args.max_seq_len = 2048
    print("MiMo args:", args.n_layers, "layers,", args.n_routed_experts, "experts, dense:", args.n_dense_layers)
    with torch.device("cuda"):
        modelm = MiMoTransformer(args)

    idx = build_index(ckpt)
    cache = {}
    t0 = time.time()
    _load_weights(modelm, args, ckpt, idx, cache, rank, ws)
    print(f"weights loaded in {time.time()-t0:.0f}s")

    tok = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
    ids = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                  add_generation_prompt=True, tokenize=True)
    if hasattr(ids, "input_ids"): ids = ids.input_ids
    if hasattr(ids, "tolist"): ids = ids.tolist()
    if ids and isinstance(ids[0], list): ids = ids[0]
    ids = [int(x) for x in ids]
    plen = len(ids)
    total = plen + max_new_tokens
    tokens = torch.full((1, total), -1, dtype=torch.long, device="cuda")
    tokens[0, :plen] = torch.tensor(ids, device="cuda")

    prev = 0
    for cur in range(plen, total):
        logits = modelm.forward(tokens[:, prev:cur], prev)
        nxt = logits.argmax(-1)[0]
        tokens[0, cur] = nxt
        prev = cur
        if rank == 0:
            print(tok.decode([int(nxt)], skip_special_tokens=True), end="", flush=True)
        if int(nxt) == tok.eos_token_id:
            break
    if rank == 0:
        out = tok.decode(tokens[0, plen:].tolist(), skip_special_tokens=True)
        print("\n\n=== MiMo completion ===\n" + out)
    if ws > 1:
        dist.destroy_process_group()


def _load_weights(modelm, args, ckpt, idx, cache, rank, ws):
    """Load HF weights into the MiMo modules with per-rank sharding."""
    g = lambda k: get_tensor(ckpt, idx, k, cache)
    dev = f"cuda:{rank}"

    # embedding (vocab-parallel: shard rows)
    emb = g("model.embed_tokens.weight")
    vp = args.vocab_size // ws
    modelm.embed.weight.data.copy_(emb[rank*vp:(rank+1)*vp].to(dev))
    # final norm + head
    modelm.norm.weight.data.copy_(g("model.norm.weight").to(dev))
    head = g("lm_head.weight")
    modelm.head.weight.data.copy_(head[rank*vp:(rank+1)*vp].to(dev).float())

    n_local_kv = max(1, args.num_key_value_heads // ws)
    for lid, blk in enumerate(modelm.layers):
        p = f"model.layers.{lid}."
        blk.attn_norm.weight.data.copy_(g(p+"input_layernorm.weight").to(dev))
        blk.ffn_norm.weight.data.copy_(g(p+"post_attention_layernorm.weight").to(dev))
        # attention: fused qkv (FP8) sharded by heads; o_proj bf16; sink
        _load_attn(blk.attn, p, g, dev, args, ws, rank)
        if lid < args.n_dense_layers:
            _load_fp8_linear(blk.ffn.w1, g(p+"mlp.gate_proj.weight"), g(p+"mlp.gate_proj.weight_scale_inv"), dev, ws, rank, "col")
            _load_fp8_linear(blk.ffn.w3, g(p+"mlp.up_proj.weight"), g(p+"mlp.up_proj.weight_scale_inv"), dev, ws, rank, "col")
            _load_fp8_linear(blk.ffn.w2, g(p+"mlp.down_proj.weight"), g(p+"mlp.down_proj.weight_scale_inv"), dev, ws, rank, "row")
        else:
            blk.ffn.gate.weight.data.copy_(g(p+"mlp.gate.weight").to(dev).float())
            eb = g(p+"mlp.gate.e_score_correction_bias")
            if eb is not None and blk.ffn.gate.bias is not None:
                blk.ffn.gate.bias.data.copy_(eb.to(dev).float())
            n_local = args.n_routed_experts // ws
            for i in range(n_local):
                gid = rank*n_local + i
                ep = f"{p}mlp.experts.{gid}."
                ex = blk.ffn.experts[i]
                for name, w, s in [("gate", ex.gate_w, ex.gate_s), ("up", ex.up_w, ex.up_s), ("down", ex.down_w, ex.down_s)]:
                    w.data.copy_(g(f"{ep}{name}_proj.weight").to(dev))
                    s.data.copy_(g(f"{ep}{name}_proj.weight_scale").to(dev))


def _load_attn(attn, p, g, dev, args, ws, rank):
    # fused qkv weight [Q+K+V, dim] FP8; split into per-rank head shards
    qkv = g(p+"self_attn.qkv_proj.weight")
    qkv_s = g(p+"self_attn.qkv_proj.weight_scale_inv")
    from mimo_quant import dequant_fp8_blockwise
    qkv_bf = dequant_fp8_blockwise(qkv.to(dev), qkv_s.to(dev))
    nh, nkv, hd, vhd = args.num_attention_heads, args.num_key_value_heads, args.head_dim, args.v_head_dim
    qsz, ksz, vsz = nh*hd, nkv*hd, nkv*vhd
    qp, kp, vp_ = qkv_bf[:qsz], qkv_bf[qsz:qsz+ksz], qkv_bf[qsz+ksz:]
    lh = nh // ws; lkv = max(1, nkv // ws)
    qp = qp.view(nh, hd)[rank*lh:(rank+1)*lh].reshape(-1, args.dim)
    kp = kp.view(nkv, hd)[rank*lkv:(rank+1)*lkv].reshape(-1, args.dim)
    vp_ = vp_.view(nkv, vhd)[rank*lkv:(rank+1)*lkv].reshape(-1, args.dim)
    attn.qkv_proj.weight.data.copy_(torch.cat([qp, kp, vp_], 0).to(attn.qkv_proj.weight.dtype))
    # o_proj bf16 [dim, nh*vhd] row-parallel
    o = g(p+"self_attn.o_proj.weight").to(dev)
    osh = o.shape[1] // ws
    attn.o_proj.weight.data.copy_(o[:, rank*osh:(rank+1)*osh].to(attn.o_proj.weight.dtype))
    if attn.attention_sink_bias is not None:
        sink = g(p+"self_attn.attention_sink_bias")
        if sink is not None:
            attn.attention_sink_bias.data.copy_(sink[rank*lh:(rank+1)*lh].to(dev))


def _load_fp8_linear(lin, w, s, dev, ws, rank, kind):
    from mimo_quant import dequant_fp8_blockwise
    wb = dequant_fp8_blockwise(w.to(dev), s.to(dev))
    if kind == "col":
        sh = wb.shape[0] // ws; wb = wb[rank*sh:(rank+1)*sh]
    else:
        sh = wb.shape[1] // ws; wb = wb[:, rank*sh:(rank+1)*sh]
    lin.weight.data.copy_(wb.to(lin.weight.dtype))


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument("--ckpt-path", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=20)
    ap.add_argument("--prompt", default="What is the capital of France? Answer in one sentence.")
    a = ap.parse_args()
    main(a.ckpt_path, a.max_new_tokens, a.prompt)
