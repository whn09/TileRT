#!/usr/bin/env bash
# MiMo-V2.5-Pro (FP8) single-node 8x B200 deployment.
# Based on Xiaomi's OFFICIAL recommended config (originally 2-node 16-GPU),
# adapted to single node. Stock latest sglang image, NO patches.
#
# Official 2-node config (for reference):
#   --pp-size 1 --dp-size 2 --ep-size 16 --tp-size 16 --moe-dense-tp-size 1
#   --enable-dp-attention --moe-a2a-backend deepep --nnodes 2 ...
#   --attention-backend fa3 --speculative-algorithm EAGLE ...
#
# Single-node adaptation: tp/ep 16->8, drop multi-node addr/rank/nnodes.
# Keep dp-size 2 (user: hard constraint), enable-dp-attention, EAGLE spec.
set -eu

MODEL_PATH="${MODEL_PATH:-/nvme/models/MiMo-V2.5-Pro}"
IMAGE="${IMAGE:-lmsysorg/sglang:dev-cu13}"
PORT="${PORT:-29999}"

docker rm -f sglang >/dev/null 2>&1 || true

docker run -d --name sglang \
    --gpus all --network host --ipc host \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    --shm-size 32g \
    -v /opt/dlami/nvme:/nvme \
    -e HF_HOME=/nvme/hf_cache \
    -e SGLANG_ENABLE_SPEC_V2=1 \
    -e SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256 \
    -e NCCL_NET_PLUGIN=none \
    -e NCCL_P2P_LEVEL=NVL \
    "${IMAGE}" \
    python3 -m sglang.launch_server \
        --model-path "${MODEL_PATH}" \
        --trust-remote-code \
        --pp-size 1 \
        --dp-size 2 \
        --ep-size 8 \
        --tp-size 8 \
        --moe-dense-tp-size 1 \
        --enable-dp-attention \
        --moe-a2a-backend deepep \
        --page-size 64 \
        --attention-backend fa3 \
        --quantization fp8 \
        --mem-fraction-static 0.7 \
        --max-running-requests 128 \
        --cuda-graph-max-bs 64 \
        --chunked-prefill-size 32768 \
        --context-length 1048576 \
        --tokenizer-worker-num 64 \
        --speculative-algorithm EAGLE \
        --speculative-num-steps 3 \
        --speculative-eagle-topk 1 \
        --speculative-num-draft-tokens 4 \
        --enable-multi-layer-eagle \
        --host 0.0.0.0 \
        --port "${PORT}" \
        --reasoning-parser mimo \
        --tool-call-parser mimo \
        --watchdog-timeout 3600 \
        --model-loader-extra-config '{"enable_multithread_load": "true","num_threads": 64}'
