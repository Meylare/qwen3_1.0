#!/usr/bin/env python3
import argparse
import gc

import torch
from transformers import BitsAndBytesConfig
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeForConditionalGeneration,
)


DEFAULT_MODEL_ID = "/home/ubuntu/models/Qwen3-Omni-30B-A3B-Thinking"
DEFAULT_SKIP_MODULES = [
    "mlp.gate",
    "mlp.shared_expert_gate",
    "thinker.audio_tower",
    "thinker.visual",
    "thinker.audio_tower.proj1",
    "thinker.audio_tower.proj2",
    "thinker.visual.merger",
    "thinker.visual.merger_list",
    "lm_head",
]


def gib(num_bytes: int) -> float:
    return num_bytes / 1024**3


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure Qwen3-Omni NF4 model memory footprint on GPU.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("This script is intended for CUDA devices.")
    device_index = device.index if device.index is not None else 0

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype_map[args.dtype],
        bnb_4bit_use_double_quant=True,
        llm_int8_skip_modules=DEFAULT_SKIP_MODULES,
    )

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.set_device(device_index)
    torch.cuda.reset_peak_memory_stats(device_index)

    free_before, total_before = torch.cuda.mem_get_info(device_index)
    alloc_before = torch.cuda.memory_allocated(device_index)
    reserved_before = torch.cuda.memory_reserved(device_index)

    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        args.model_id,
        device_map=str(device),
        torch_dtype=dtype_map[args.dtype],
        quantization_config=quant_config,
    )

    torch.cuda.synchronize(device_index)

    free_after, total_after = torch.cuda.mem_get_info(device_index)
    alloc_after = torch.cuda.memory_allocated(device_index)
    reserved_after = torch.cuda.memory_reserved(device_index)
    peak_alloc = torch.cuda.max_memory_allocated(device_index)
    peak_reserved = torch.cuda.max_memory_reserved(device_index)

    print(f"model_id: {args.model_id}")
    print(f"device: {device}")
    print(f"total_vram: {gib(total_after):.2f} GiB")
    print(f"free_before: {gib(free_before):.2f} GiB")
    print(f"free_after: {gib(free_after):.2f} GiB")
    print(f"allocated_delta: {gib(alloc_after - alloc_before):.2f} GiB")
    print(f"reserved_delta: {gib(reserved_after - reserved_before):.2f} GiB")
    print(f"peak_allocated: {gib(peak_alloc):.2f} GiB")
    print(f"peak_reserved: {gib(peak_reserved):.2f} GiB")

    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
