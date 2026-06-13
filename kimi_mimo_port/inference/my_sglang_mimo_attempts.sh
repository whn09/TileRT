#!/usr/bin/env bash
# 我（Claude）之前尝试在单节点 8×B200 上用 SGLang 跑 MiMo-V2.5-Pro-FP4-DFlash 的命令记录。
# 供对照用户的可工作脚本 launch_sglang_single.sh 找差异。
# 环境：B200 (SM100)；镜像 lmsysorg/sglang:dev-cu13；PR #27638 经 PYTHONPATH 覆盖。

# ---- 尝试 1：dev 镜像原生（无 PR），崩在 KeyError w2_weight_scale ----
# docker run ... lmsysorg/sglang:dev-cu13 python3 -m sglang.launch_server \
#   --model-path /nvme/models/MiMo-V2.5-Pro-FP4-DFlash \
#   --speculative-algorithm DFLASH --speculative-draft-model-path .../dflash \
#   --speculative-num-draft-tokens 8 --tp 8 --ep-size 8 \
#   --quantization fp8 --attention-backend fa3 --moe-dense-tp-size 1 ...

# ---- 尝试 N（最后一次，崩在 num_tokens // None）----
docker run -d --name sglang --gpus all --ipc=host --network host --shm-size 32g \
  -v /opt/dlami/nvme:/nvme -e HF_HOME=/nvme/hf_cache \
  -e PYTHONPATH=/nvme/sglang-pr/python \
  -e NCCL_NET_PLUGIN=none -e NCCL_P2P_LEVEL=NVL \
  lmsysorg/sglang:dev-cu13 \
  python3 -m sglang.launch_server --trust-remote-code \
    --model-path /nvme/models/MiMo-V2.5-Pro-FP4-DFlash \
    --speculative-algorithm DFLASH \
    --speculative-draft-model-path /nvme/models/MiMo-V2.5-Pro-FP4-DFlash/dflash \
    --speculative-num-draft-tokens 8 \
    --tp 8 --ep-size 8 --enable-dp-attention \
    --quantization fp8 --attention-backend fa4 --moe-dense-tp-size 1 \
    --mem-fraction-static 0.8 --context-length 8192 \
    --disable-overlap-schedule --disable-cuda-graph --skip-server-warmup \
    --host 0.0.0.0 --port 29999

# 与用户可工作脚本(launch_sglang_single.sh)的关键差异：
#  1. 用户: --dp-size 1 (显式)；我: --enable-dp-attention 但没 --dp-size 1 -> 走了 dp 路径
#  2. 用户: 无 spec decode (纯 MiMo-Pro)；我: 上来就加 DFLASH spec decode
#  3. 用户: --disable-piecewise-cuda-graph (只禁 piecewise)；我: --disable-cuda-graph (全禁)
#  4. 用户: SGLANG_ENABLE_SPEC_V2=1 环境变量；我: 没设
#  5. 用户: 挂载自定义 mimo_v2_flash.py / _nextn.py patch；我: 用 PR #27638 的 mimo_v2.py
#  6. 用户在 H200(fa3 OK)；我在 B200(fa3 被 SM 断言挡, 故用 fa4)
