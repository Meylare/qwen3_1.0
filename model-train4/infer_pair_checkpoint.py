"""
Run one pairwise inference against a saved LoRA checkpoint.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Infer one pair with a Qwen3.5 LoRA checkpoint.")
    parser.add_argument("--base_model", default="Qwen/Qwen3.5-9B-Base")
    parser.add_argument("--adapter_dir", type=Path, required=True)
    parser.add_argument("--dataset_jsonl", type=Path, required=True)
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--no_load_in_4bit", action="store_false", dest="load_in_4bit")
    parser.add_argument("--output_json", type=Path, default=Path(__file__).resolve().parent / "output" / "infer_pair_result.json")
    return parser.parse_args()


def load_jsonl_row(path: Path, index: int) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        for row_idx, line in enumerate(handle):
            if row_idx == index:
                return json.loads(line)
    raise IndexError(f"Sample index {index} out of range for {path}")


def main() -> None:
    args = parse_args()
    sample = load_jsonl_row(args.dataset_jsonl, args.sample_index)

    from peft import PeftModel
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3_5ForConditionalGeneration

    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
        "trust_remote_code": True,
    }
    if args.load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    processor.tokenizer.padding_side = "left"

    base_model = Qwen3_5ForConditionalGeneration.from_pretrained(args.base_model, **model_kwargs)
    model = PeftModel.from_pretrained(base_model, str(args.adapter_dir))
    model.eval()

    inputs = processor.apply_chat_template(
        sample["prompt"],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

    start = time.perf_counter()
    generated = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.temperature > 0,
        temperature=args.temperature,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
    )
    elapsed = time.perf_counter() - start

    prompt_len = inputs["input_ids"].shape[1]
    trimmed = generated[:, prompt_len:]
    answer = processor.batch_decode(trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]

    payload = {
        "base_model": args.base_model,
        "adapter_dir": str(args.adapter_dir.resolve()),
        "dataset_jsonl": str(args.dataset_jsonl.resolve()),
        "sample_index": args.sample_index,
        "answer": answer,
        "latency_sec": round(elapsed, 4),
        "label": sample.get("label"),
        "video_a": sample.get("video_a"),
        "video_b": sample.get("video_b"),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
