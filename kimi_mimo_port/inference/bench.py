"""Standalone decode-throughput benchmark for the Kimi-K2 tilelang port.
Measures pure single-request decode TPOT/OTPS (batch=1), same metric as the
TileRT / SGLang numbers in B200_PERFORMANCE_REPORT.md."""
import os, json, time
import torch, torch.distributed as dist
from transformers import AutoTokenizer
from safetensors.torch import load_model
from model import Transformer, ModelArgs


def main(ckpt_path, config, prompt, warmup_tokens=8, measure_tokens=128):
    world_size = int(os.getenv("WORLD_SIZE", 1))
    rank = int(os.getenv("RANK", 0))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    if world_size > 1:
        dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    torch.set_default_dtype(torch.bfloat16)
    torch.manual_seed(0)
    if rank != 0:
        import builtins; builtins.print = lambda *_, **__: None

    with open(config) as f:
        args = ModelArgs(**json.load(f))
    with torch.device("cuda"):
        model = Transformer(args)
    tok = AutoTokenizer.from_pretrained(ckpt_path, trust_remote_code=True)
    load_model(model, os.path.join(ckpt_path, f"model{rank}-mp{world_size}.safetensors"))

    ids = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                  add_generation_prompt=True, tokenize=True)
    if hasattr(ids, "input_ids"): ids = ids.input_ids
    if hasattr(ids, "tolist"): ids = ids.tolist()
    if ids and isinstance(ids[0], list): ids = ids[0]
    ids = [int(x) for x in ids]
    prompt_len = len(ids)
    # prefill writes 1 token at index prompt_len, then warmup+measure decode
    # steps each write one more -> +8 slack to be safe.
    total = prompt_len + 1 + warmup_tokens + measure_tokens + 8
    tokens = torch.full((1, total), -1, dtype=torch.long, device="cuda")
    tokens[0, :prompt_len] = torch.tensor(ids, device="cuda")

    # prefill
    prev = 0
    cur = prompt_len
    logits = model.forward(tokens[:, prev:cur], prev)
    nxt = logits.argmax(-1)
    tokens[0, cur] = nxt[0]
    prev = cur

    # warmup decode steps (JIT settle)
    for i in range(warmup_tokens):
        cur = prev + 1
        logits = model.forward(tokens[:, prev:cur], prev)
        tokens[0, cur] = logits.argmax(-1)[0]
        prev = cur
    torch.cuda.synchronize()

    # measured decode steps
    t0 = time.time()
    for i in range(measure_tokens):
        cur = prev + 1
        logits = model.forward(tokens[:, prev:cur], prev)
        tokens[0, cur] = logits.argmax(-1)[0]
        prev = cur
    torch.cuda.synchronize()
    dt = time.time() - t0

    if rank == 0:
        tpot = dt / measure_tokens * 1000
        print(f"\n=== Kimi-K2 tilelang decode benchmark (B200, bsz=1) ===")
        print(f"prompt_len={prompt_len} measured_tokens={measure_tokens}")
        print(f"decode time: {dt:.3f}s")
        print(f"TPOT: {tpot:.2f} ms/token")
        print(f"OTPS: {measure_tokens/dt:.2f} tokens/s")
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-path", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--prompt", default="Tell me about the history of computing.")
    p.add_argument("--measure-tokens", type=int, default=128)
    a = p.parse_args()
    main(a.ckpt_path, a.config, a.prompt, measure_tokens=a.measure_tokens)
