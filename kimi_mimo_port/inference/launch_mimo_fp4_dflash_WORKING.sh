#!/usr/bin/env bash
# ✅ WORKING: MiMo-V2.5-Pro-FP4-DFlash on single-node 8x B200.
# Result: correct output, 97.6 tok/s long-text (FP4 MoE + official DFlash spec decode).
#
# Requires PR #27638 (csAugust:mimo-v2-fp4-dflash) overlaid via PYTHONPATH, plus
# ONE necessary patch: fp8.py:1430 `layer.w13_weight_scale_inv.format_ue8m0` ->
# `getattr(..., "format_ue8m0", False)` (matches safe pattern at lines 784/870/1109
# in the same file; PR's own oversight on the DeepGEMM ue8m0-requant branch).
#
# Key learnings:
#  - dp=1 (NOT dp=2) on 8 GPU: MiMo fused qkv is TP=8-interleaved => effective
#    attn-TP must = 8 = tp_size/dp_size/attn_cp_size. dp=1 satisfies it on 8 GPU.
#    (dp=2 needs 16 GPU as in Xiaomi's 2-node official cmd.)
#  - fa4 not fa3: fa3 backend SM-asserts (SM8/9 only) on B200 (SM100); fa4 has no
#    such assert AND supports attention sink (which MiMo SWA layers need).
set -eu
SGLANG_PR=/nvme/sglang-pr/python  # PR #27638 checkout + the fp8.py getattr patch
docker rm -f sglang >/dev/null 2>&1 || true
docker run -d --name sglang --gpus all --network host --ipc host --ulimit memlock=-1 --shm-size 32g \
  -v /opt/dlami/nvme:/nvme -e HF_HOME=/nvme/hf_cache -e PYTHONPATH=${SGLANG_PR} \
  -e SGLANG_ENABLE_SPEC_V2=1 -e SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256 \
  -e NCCL_NET_PLUGIN=none -e NCCL_P2P_LEVEL=NVL \
  lmsysorg/sglang:dev-cu13 \
  python3 -m sglang.launch_server \
    --model-path /nvme/models/MiMo-V2.5-Pro-FP4-DFlash --trust-remote-code \
    --speculative-algorithm DFLASH \
    --speculative-draft-model-path /nvme/models/MiMo-V2.5-Pro-FP4-DFlash/dflash \
    --speculative-num-draft-tokens 8 \
    --pp-size 1 --dp-size 1 --ep-size 8 --tp-size 8 --moe-dense-tp-size 1 \
    --enable-dp-attention --moe-a2a-backend deepep \
    --page-size 64 --attention-backend fa4 --quantization fp8 \
    --mem-fraction-static 0.8 --max-running-requests 64 \
    --chunked-prefill-size 32768 --context-length 65536 --tokenizer-worker-num 64 \
    --host 0.0.0.0 --port 29999 \
    --reasoning-parser mimo --tool-call-parser mimo --watchdog-timeout 3600 \
    --model-loader-extra-config '{"enable_multithread_load": "true","num_threads": 64}'
