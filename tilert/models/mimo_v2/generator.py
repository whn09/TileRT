"""Generator for MiMo-V2.5-Pro (UltraSpeed).

The public surface mirrors ``GLM5Generator`` / ``DSAv32Generator`` so that
``tilert.generate`` and any benchmark harness can drive MiMo identically once
the backend ``.so`` is available. The only intentional difference is the decode
layer (:class:`ShowHandsMiMoLayer`), which raises a clear error at the kernel
boundary until ``libtilert_mimo.so`` ships.
"""

import os
import time

import torch
from transformers import AutoTokenizer

from tilert import logger
from tilert.models.mimo_v2.model_args import ModelArgsMiMoV2
from tilert.models.mimo_v2.modules.end2end import ShowHandsMiMoLayer

__all__ = [
    "MiMoV2Generator",
]


def stats_time(time_list: list[float], header: str = "") -> None:
    """Print simple latency / throughput stats for a list of per-step times."""
    if not time_list:
        return
    total = sum(time_list)
    n = len(time_list)
    avg_ms = total / n * 1000
    if header:
        logger.info(header)
    logger.info(f"--Number of steps: {n}")
    logger.info(f"--Avg step time: {avg_ms:.2f}ms ({1000 / avg_ms:.2f} steps/s)")
    logger.info(f"--Output TPS (no MTP): {n / total:.2f} tokens/s")


class MiMoV2Generator:
    """Show-hands generator for MiMo-V2.5-Pro.

    Args mirror the GLM-5 / DeepSeek generators so the CLI and benchmark code
    are model-agnostic. ``with_mtp`` here selects the DFlash block-diffusion
    drafter (MiMo's speculative-decoding path) rather than the per-layer LM-head
    MTP used by DeepSeek/GLM.
    """

    def __init__(
        self,
        model_args: ModelArgsMiMoV2,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        model_weights_dir: str = "",
        with_mtp: bool = False,
        top_p: float = 0.9,
        top_k: int = 256,
        use_topp: bool = False,
        enable_thinking: bool = False,
        sampling_seed: int = 42,
    ):
        torch.set_num_threads(64)
        self.model_weights_dir = model_weights_dir
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.with_mtp = with_mtp
        self.enable_thinking = enable_thinking
        self.sampling_seed = sampling_seed
        self.config = model_args

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_weights_dir, trust_remote_code=True
        )  # nosec B615
        # MiMo ships a chat template inside the tokenizer config; honor an
        # explicit chat_template.jinja override if the checkpoint provides one.
        jinja_file_path = os.path.join(self.model_weights_dir, "chat_template.jinja")
        if os.path.exists(jinja_file_path):
            with open(jinja_file_path, encoding="utf-8") as f:
                self.tokenizer.chat_template = f.read()
        self.eos_id = self.tokenizer.eos_token_id

        self.batch_size = 1
        # DFlash drafts a whole block per forward; reuse the config block size.
        self.mtp_seq_len = self.config.dflash_block_size

        self.stop_token_ids: set[int] = set()
        if self.eos_id is not None:
            self.stop_token_ids.add(self.eos_id)
        logger.info(f"Stop token IDs: {self.stop_token_ids}")

        self.default_device = torch.device("cuda:0")

        self.decode_layer = ShowHandsMiMoLayer(
            model_args=self.config,
            model_path=self.model_weights_dir,
            with_mtp=with_mtp,
            top_p=top_p,
            top_k=top_k,
            use_topp=use_topp,
        )

    # ---- lifecycle (parity with GLM5Generator) ----
    def init(self) -> None:
        from tilert.tilert_init import tilert_init

        tilert_init()

    def cleanup(self) -> None:
        self.decode_layer.cleanup()

    def init_random_weights(self) -> None:
        self.decode_layer.init_random_weights()

    def from_pretrained(self) -> None:
        self.decode_layer.from_pretrained(self.model_weights_dir)

    def update_sampling_params(
        self,
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 256,
        use_topp: bool = True,
    ) -> None:
        self.temperature = temperature
        self.decode_layer.update_sampling_config(
            temperature=temperature, top_p=top_p, top_k=top_k, use_topp=use_topp
        )

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        print_log: bool = True,
        with_mtp: bool | None = None,
        prompt_tokens: list[int] | None = None,
    ) -> tuple[str, list[float], list[int], int]:
        """Single-sequence generation. Mirrors GLM5Generator.generate().

        Returns (result_text, time_list, accepted_counts, prompt_len);
        accepted_counts is empty in non-MTP (no-DFlash) mode.
        """
        active_mtp = with_mtp if with_mtp is not None else self.with_mtp
        if active_mtp and not self.with_mtp:
            raise ValueError("Cannot use DFlash mode: drafter weights were not loaded")
        self.decode_layer.set_sampling_seed(self.sampling_seed, with_mtp=active_mtp)

        if active_mtp:
            return self._generate_with_dflash(prompt, print_log, prompt_tokens=prompt_tokens)
        result, time_list, prompt_len = self._generate_plain(
            prompt, print_log, prompt_tokens=prompt_tokens
        )
        return result, time_list, [], prompt_len

    def _encode(self, prompt: str, prompt_tokens: list[int] | None) -> list[int]:
        if prompt_tokens is not None:
            return prompt_tokens
        messages = [{"role": "user", "content": prompt}]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )

    def _generate_plain(
        self,
        prompt: str,
        print_log: bool = True,
        prompt_tokens: list[int] | None = None,
    ) -> tuple[str, list[float], int]:
        """Standard autoregressive decode (one token per forward)."""
        prompt_tokens = self._encode(prompt, prompt_tokens)
        prompt_len = len(prompt_tokens)
        total_len = min(self.config.max_seq_len, self.max_new_tokens + prompt_len)

        tokens = torch.full(
            (self.batch_size, total_len), -1, dtype=torch.long, device=self.default_device
        )
        tokens[0, :prompt_len] = torch.tensor(
            prompt_tokens, dtype=torch.long, device=self.default_device
        )
        prompt_mask = tokens != -1

        prev_pos = 0
        time_list: list[float] = []
        for cur_pos_val in range(1, total_len):
            start_time = time.time()
            results = self.decode_layer.forward(tokens[0, prev_pos], with_mtp=False)
            time_list.append(time.time() - start_time)

            next_token = self.decode_layer.extract_next_token(results)
            next_token = torch.where(
                prompt_mask[0, cur_pos_val], tokens[0, cur_pos_val], next_token
            )
            tokens[0, cur_pos_val] = next_token
            prev_pos = cur_pos_val

            if cur_pos_val >= prompt_len:
                if print_log:
                    print(
                        self.tokenizer.decode([next_token.item()], skip_special_tokens=True),
                        end="",
                        flush=True,
                    )
                if next_token.item() in self.stop_token_ids:
                    break

        if print_log:
            print("\n")
            stats_time(time_list, "==== Performance ====")

        self.decode_layer.reset_sequence()
        completion = tokens[0, prompt_len:].tolist()
        completion = [t for t in completion if t != -1 and t not in self.stop_token_ids]
        text = self.tokenizer.decode(completion, skip_special_tokens=True)
        return f"{text}\n", time_list, prompt_len

    def _generate_with_dflash(
        self,
        prompt: str,
        print_log: bool = True,
        prompt_tokens: list[int] | None = None,
    ) -> tuple[str, list[float], list[int], int]:
        """DFlash block-diffusion speculative decode.

        The block drafter proposes ``dflash_block_size`` tokens per forward; the
        backbone verifies them in a single step. Wired identically to the GLM-5
        MTP loop (draft -> forward -> num_accepted -> emit) so benchmark stats
        line up; the actual accept/verify logic lives in the backend kernels.
        """
        prompt_tokens = self._encode(prompt, prompt_tokens)
        prompt_len = len(prompt_tokens)
        total_len = min(self.config.max_seq_len, self.max_new_tokens + prompt_len)

        tokens = torch.full(
            (self.batch_size, total_len), -1, dtype=torch.long, device=self.default_device
        )
        tokens[0, :prompt_len] = torch.tensor(
            prompt_tokens, dtype=torch.long, device=self.default_device
        )

        decode_time_list: list[float] = []
        decode_accepted_counts: list[int] = []

        # Prefill the prompt block-by-block (mirrors GLM-5's MTP prefill loop).
        self.decode_layer.prefill(tokens[0, :prompt_len], block_size=self.mtp_seq_len)

        cur_pos = prompt_len - 1
        self.set_cur_pos(prompt_len - 1)
        finished = False

        while cur_pos < total_len - 1 and not finished:
            draft_tokens = self.decode_layer.get_next_draft_tokens(0).reshape(1, self.mtp_seq_len)

            start_time = time.time()
            self.decode_layer.forward(draft_tokens, with_mtp=True)
            decode_time_list.append(time.time() - start_time)

            num_accepted = self.decode_layer.get_num_accepted(0)
            predicted = self.decode_layer.get_predicted_tokens(0).flatten()
            decode_accepted_counts.append(num_accepted)

            for i in range(num_accepted):
                if cur_pos + 1 + i >= total_len:
                    break
                new_token = int(predicted[i].item())
                tokens[0, cur_pos + 1 + i] = new_token
                if cur_pos + 1 + i >= prompt_len and print_log:
                    print(
                        self.tokenizer.decode([new_token], skip_special_tokens=True),
                        end="",
                        flush=True,
                    )
                if new_token in self.stop_token_ids:
                    finished = True
                    break
            cur_pos += num_accepted

        if print_log:
            print("\n")
            total_tokens = sum(decode_accepted_counts)
            if decode_time_list:
                total_decode_time = sum(decode_time_list)
                eff_tps = total_tokens / total_decode_time if total_decode_time > 0 else 0.0
                logger.info(f"--Total tokens generated: {total_tokens}")
                logger.info(f"--Effective TPS (with DFlash): {eff_tps:.2f} tokens/s")
                if decode_accepted_counts:
                    mean_acc = total_tokens / len(decode_accepted_counts)
                    logger.info(f"--Accepted length: mean={mean_acc:.2f}")

        self.decode_layer.reset_sequence()
        completion = tokens[0, prompt_len:].tolist()
        completion = [t for t in completion if t != -1 and t not in self.stop_token_ids]
        text = self.tokenizer.decode(completion, skip_special_tokens=True)
        return f"{text}\n", decode_time_list, decode_accepted_counts, prompt_len

    def set_cur_pos(self, cur_pos: int) -> None:
        """Set the runtime RoPE position (e.g. after prefill / cache inject)."""
        self.decode_layer.set_cur_pos(cur_pos)
