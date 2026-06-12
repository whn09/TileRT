"""Expected backend op surface for MiMo-V2.5-Pro (documentation stub).

The existing ``deepseek_v3_2`` / ``glm_5`` backends expose ~35 fused kernels via
``torch.ops.tilert.*`` (e.g. ``flash_sparse_mla_op``, ``rmsnorm_up_gate_silu_op``,
``expert_select_up_gate_silu_op``). MiMo's attention is GQA + sliding-window
rather than MLA, and its MoE GEMM is MXFP4 rather than FP8, so it needs a
different — though analogous — set of kernels.

This file documents the ops we *expect* ``libtilert_mimo.so`` to register, so
that ``modules/end2end.py`` can be filled in quickly once the .so lands. None of
these exist yet; calling them will raise "no such operator".

Anticipated op surface (names illustrative, to be confirmed against the .so):

  Attention (GQA + SWA hybrid):
    torch.ops.tilert.mimo_qkv_proj_op            # fused QKV projection (GQA)
    torch.ops.tilert.mimo_rope_op                # RoPE (theta = 5e6)
    torch.ops.tilert.mimo_flash_gqa_full_op      # full-attention layers
    torch.ops.tilert.mimo_flash_gqa_swa_op       # sliding-window (window=128)
    torch.ops.tilert.mimo_attn_sink_bias_op      # SWA attention-sink bias
    torch.ops.tilert.mimo_o_proj_op              # o_proj (kept >= FP8, no FP4)

  MoE (MXFP4 experts, block 32):
    torch.ops.tilert.mimo_router_sigmoid_topk_op # sigmoid scoring, top-8 of 384
    torch.ops.tilert.mimo_expert_mxfp4_gemm_op   # MXFP4 grouped expert GEMM
    torch.ops.tilert.mimo_expert_down_allreduce_op

  Norms / dense MLP:
    torch.ops.tilert.mimo_rmsnorm_op
    torch.ops.tilert.mimo_dense_mlp_op           # first n_dense_layers use this

  DFlash block-diffusion speculative decoding:
    torch.ops.tilert.mimo_dflash_draft_op        # fill a masked block in 1 pass
    torch.ops.tilert.mimo_dflash_verify_op       # backbone verify + accept count
"""

__all__: list[str] = []
