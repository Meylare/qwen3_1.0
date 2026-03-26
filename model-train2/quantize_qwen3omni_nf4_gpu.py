#!/usr/bin/env python3
"""Quantize Qwen3-Omni model to BitsAndBytes 4-bit NF4 on GPU."""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoProcessor, BitsAndBytesConfig
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeForConditionalGeneration,
)


DEFAULT_SRC = "/home/ubuntu/models/Qwen3-Omni-30B-A3B-Thinking"
DEFAULT_DST = "/home/ubuntu/models/Qwen3-Omni-30B-A3B-Thinking-bnb4-nf4"
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantize Qwen3-Omni to BitsAndBytes NF4 (4-bit) using GPU."
    )
    parser.add_argument("--src-model", default=DEFAULT_SRC, help="Source model path or HF id.")
    parser.add_argument("--dst-model", default=DEFAULT_DST, help="Output directory for quantized model.")
    parser.add_argument("--device", default="cuda:0", help='Target device, e.g. "cuda:0".')
    parser.add_argument(
        "--compute-dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="bnb_4bit_compute_dtype.",
    )
    parser.add_argument(
        "--double-quant",
        action="store_true",
        default=True,
        help="Enable nested quantization (default: enabled).",
    )
    parser.add_argument(
        "--no-double-quant",
        action="store_true",
        help="Disable nested quantization.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=True,
        help="Pass trust_remote_code=True (default: enabled).",
    )
    parser.add_argument(
        "--no-trust-remote-code",
        action="store_true",
        help="Disable trust_remote_code.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into a non-empty output directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This script is intended for GPU quantization.")

    if not args.device.startswith("cuda"):
        raise ValueError(f'Expected a CUDA device, got "{args.device}".')

    dst_model = Path(args.dst_model)
    if dst_model.exists() and any(dst_model.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {dst_model}. "
            "Use --overwrite to allow reusing this directory."
        )
    dst_model.mkdir(parents=True, exist_ok=True)

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    compute_dtype = dtype_map[args.compute_dtype]

    use_double_quant = args.double_quant and not args.no_double_quant
    trust_remote_code = args.trust_remote_code and not args.no_trust_remote_code

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=use_double_quant,
        llm_int8_skip_modules=DEFAULT_SKIP_MODULES,
    )

    print(f"[1/4] Loading source model: {args.src_model}")
    print(
        "[cfg] quantization=bitsandbytes, load_in_4bit=True, "
        f"bnb_4bit_quant_type=nf4, bnb_4bit_compute_dtype={args.compute_dtype}, "
        f"bnb_4bit_use_double_quant={use_double_quant}"
    )
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        args.src_model,
        torch_dtype=compute_dtype,
        quantization_config=quant_config,
        device_map=args.device,
        trust_remote_code=trust_remote_code,
    )

    print("[2/4] Loading processor")
    processor = AutoProcessor.from_pretrained(
        args.src_model,
        trust_remote_code=trust_remote_code,
    )

    print(f"[3/4] Saving quantized model to: {dst_model}")
    model.save_pretrained(str(dst_model), safe_serialization=True)
    processor.save_pretrained(str(dst_model))

    print("[4/4] Validating saved quantization_config")
    config_path = dst_model / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing saved config: {config_path}")
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    qcfg = cfg.get("quantization_config", {})
    quant_type = qcfg.get("bnb_4bit_quant_type")
    load_in_4bit = qcfg.get("load_in_4bit")
    print(f"[ok] quantization_config.load_in_4bit={load_in_4bit}")
    print(f"[ok] quantization_config.bnb_4bit_quant_type={quant_type}")

    if quant_type != "nf4" or load_in_4bit is not True:
        raise RuntimeError(
            "Saved config does not indicate NF4 4-bit. "
            f"Found load_in_4bit={load_in_4bit}, bnb_4bit_quant_type={quant_type}"
        )

    print("[done] NF4 4-bit quantized checkpoint is ready.")
    print(f"[done] Path: {dst_model}")


if __name__ == "__main__":
    main()
