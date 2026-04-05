"""
Benchmark pairwise generation on multiple samples with token-level timing stats.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = PROJECT_ROOT.parents[1] / "models" / "Qwen3.5-9B-Base"
DEFAULT_OUTPUT_JSON = PROJECT_ROOT / "output" / "benchmark_pairs_result.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Qwen3.5 pairwise generations.")
    parser.add_argument("--base_model", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--adapter_dir", type=Path, required=True)
    parser.add_argument("--dataset_jsonl", type=Path, required=True)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--num_pairs", type=int, default=5)
    parser.add_argument("--num_generations", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=20000)
    parser.add_argument("--max_think_tokens", type=int, default=10000)
    parser.add_argument("--max_answer_tokens", type=int, default=10000)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--no_load_in_4bit", action="store_false", dest="load_in_4bit")
    parser.add_argument("--output_json", type=Path, default=DEFAULT_OUTPUT_JSON)
    return parser.parse_args()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def resolve_model_path(model_name: str) -> str:
    model_path = Path(model_name).expanduser()
    if not model_path.exists():
        raise FileNotFoundError(
            f"Local model path not found: {model_path}. "
            f"Download the base model into {DEFAULT_MODEL_DIR} or pass --base_model explicitly."
        )
    return str(model_path.resolve())


def ensure_chat_template(processor: Any, model_path: str) -> str | None:
    current_template = getattr(processor, "chat_template", None)
    tokenizer = getattr(processor, "tokenizer", None)
    tokenizer_template = getattr(tokenizer, "chat_template", None) if tokenizer is not None else None

    chat_template = current_template or tokenizer_template
    if not chat_template:
        tokenizer_config_path = Path(model_path) / "tokenizer_config.json"
        if tokenizer_config_path.exists():
            tokenizer_config = json.loads(tokenizer_config_path.read_text(encoding="utf-8"))
            chat_template = tokenizer_config.get("chat_template")

    if not chat_template:
        return None

    if tokenizer is not None and getattr(tokenizer, "chat_template", None) is None:
        tokenizer.chat_template = chat_template
    if getattr(processor, "chat_template", None) is None:
        processor.chat_template = chat_template
    return chat_template


def _append_instruction_to_system_message(messages: List[Dict[str, Any]], instruction: str) -> List[Dict[str, Any]]:
    enhanced = copy.deepcopy(messages)
    if not enhanced:
        return [
            {"role": "system", "content": [{"type": "text", "text": instruction}]},
        ]

    first = enhanced[0]
    if first.get("role") == "system":
        content = first.get("content")
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    item["text"] = f"{item.get('text', '').rstrip()}\n\n{instruction}"
                    return enhanced
            content.append({"type": "text", "text": instruction})
            first["content"] = content
            return enhanced
        if isinstance(content, str):
            first["content"] = f"{content.rstrip()}\n\n{instruction}"
            return enhanced

    enhanced.insert(0, {"role": "system", "content": [{"type": "text", "text": instruction}]})
    return enhanced


def build_limited_prompt(messages: List[Dict[str, Any]], max_think_tokens: int, max_answer_tokens: int) -> List[Dict[str, Any]]:
    instruction = (
        "Additional hard output limits:\n"
        f"- Keep the <think> section at or below {max_think_tokens} tokens.\n"
        f"- Keep the <answer> section at or below {max_answer_tokens} tokens.\n"
        "- Preserve exact output format: <think>...</think> then <answer>A</answer> or <answer>B</answer>."
    )
    return _append_instruction_to_system_message(messages, instruction)


def extract_answer_label(text: str) -> str | None:
    match = re.search(r"<answer>\s*([AB])\s*</answer>", text)
    if not match:
        return None
    return match.group(1)


def main() -> None:
    args = parse_args()
    resolved_base_model = resolve_model_path(args.base_model)
    rows = load_jsonl(args.dataset_jsonl)
    selected = rows[args.start_index : args.start_index + args.num_pairs]

    if len(selected) < args.num_pairs:
        raise ValueError(
            f"Requested {args.num_pairs} pairs starting at {args.start_index}, "
            f"but only {len(selected)} are available in {args.dataset_jsonl}."
        )

    from peft import PeftModel
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3_5ForConditionalGeneration

    model_kwargs: Dict[str, Any] = {
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

    processor = AutoProcessor.from_pretrained(resolved_base_model, trust_remote_code=True)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    processor.tokenizer.padding_side = "left"
    chat_template = ensure_chat_template(processor, resolved_base_model)

    base_model = Qwen3_5ForConditionalGeneration.from_pretrained(resolved_base_model, **model_kwargs)
    model = PeftModel.from_pretrained(base_model, str(args.adapter_dir))
    model.eval()

    records: List[Dict[str, Any]] = []
    run_start = time.perf_counter()

    for pair_offset, row in enumerate(selected):
        pair_index = args.start_index + pair_offset
        prompt = row.get("prompt")
        if not isinstance(prompt, list):
            raise ValueError(f"Row {pair_index} has no valid 'prompt' list.")

        prompt_with_limits = build_limited_prompt(
            prompt,
            max_think_tokens=args.max_think_tokens,
            max_answer_tokens=args.max_answer_tokens,
        )

        for generation_index in range(args.num_generations):
            encoded_inputs = processor.apply_chat_template(
                prompt_with_limits,
                chat_template=chat_template,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            context_tokens = int(encoded_inputs["input_ids"].shape[1])
            model_inputs = {
                key: value.to(model.device) if hasattr(value, "to") else value
                for key, value in encoded_inputs.items()
            }

            generate_kwargs: Dict[str, Any] = {
                "max_new_tokens": args.max_new_tokens,
                "do_sample": args.temperature > 0,
                "pad_token_id": processor.tokenizer.pad_token_id,
                "eos_token_id": processor.tokenizer.eos_token_id,
            }
            if args.temperature > 0:
                generate_kwargs["temperature"] = args.temperature

            start = time.perf_counter()
            generated = model.generate(**model_inputs, **generate_kwargs)
            elapsed = time.perf_counter() - start

            prompt_len = model_inputs["input_ids"].shape[1]
            trimmed = generated[:, prompt_len:]
            generation_text = processor.batch_decode(
                trimmed,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )[0]

            generated_tokens = int(trimmed.shape[1])
            tokens_per_sec = (generated_tokens / elapsed) if elapsed > 0 else 0.0
            prediction = extract_answer_label(generation_text)

            record = {
                "pair_index": pair_index,
                "generation_index": generation_index,
                "context_tokens": context_tokens,
                "generated_tokens": generated_tokens,
                "latency_sec": round(elapsed, 4),
                "tokens_per_sec": round(tokens_per_sec, 4),
                "prediction": prediction,
                "label": row.get("label"),
                "video_a": row.get("video_a"),
                "video_b": row.get("video_b"),
                "answer_text": generation_text,
            }
            records.append(record)

            print("=" * 80)
            print(
                f"pair_index={pair_index} generation_index={generation_index} "
                f"context_tokens={context_tokens} generated_tokens={generated_tokens} "
                f"latency_sec={elapsed:.4f} tokens_per_sec={tokens_per_sec:.4f}"
            )
            print(generation_text)

    total_sec = time.perf_counter() - run_start
    context_values = [row["context_tokens"] for row in records]
    generated_values = [row["generated_tokens"] for row in records]
    speed_values = [row["tokens_per_sec"] for row in records]

    summary = {
        "dataset_jsonl": str(args.dataset_jsonl.resolve()),
        "adapter_dir": str(args.adapter_dir.resolve()),
        "num_pairs": args.num_pairs,
        "num_generations_per_pair": args.num_generations,
        "total_generations": len(records),
        "max_new_tokens": args.max_new_tokens,
        "max_think_tokens": args.max_think_tokens,
        "max_answer_tokens": args.max_answer_tokens,
        "total_runtime_sec": round(total_sec, 4),
        "avg_context_tokens": round(statistics.mean(context_values), 2) if context_values else 0.0,
        "min_context_tokens": min(context_values) if context_values else 0,
        "max_context_tokens": max(context_values) if context_values else 0,
        "avg_generated_tokens": round(statistics.mean(generated_values), 2) if generated_values else 0.0,
        "avg_tokens_per_sec": round(statistics.mean(speed_values), 4) if speed_values else 0.0,
        "min_tokens_per_sec": round(min(speed_values), 4) if speed_values else 0.0,
        "max_tokens_per_sec": round(max(speed_values), 4) if speed_values else 0.0,
    }

    payload = {
        "summary": summary,
        "records": records,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("=" * 80)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved full benchmark output to {args.output_json}")


if __name__ == "__main__":
    main()
