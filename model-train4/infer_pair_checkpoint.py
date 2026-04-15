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

from pairwise_ab import get_choice_token_ids
from prompting import build_answer_only_prompt


DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "Qwen3.5-9B-Base"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Infer one pair with a Qwen3.5 LoRA checkpoint.")
    parser.add_argument("--base_model", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--adapter_dir", type=Path, required=True)
    parser.add_argument("--dataset_jsonl", type=Path, required=True)
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--response_mode", choices=("xml_reasoned", "answer_only"), default="xml_reasoned")
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


def main() -> None:
    args = parse_args()
    sample = load_jsonl_row(args.dataset_jsonl, args.sample_index)
    resolved_base_model = resolve_model_path(args.base_model)

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

    processor = AutoProcessor.from_pretrained(resolved_base_model, trust_remote_code=True)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    processor.tokenizer.padding_side = "left"
    chat_template = ensure_chat_template(processor, resolved_base_model)
    choice_token_ids = get_choice_token_ids(processor.tokenizer) if args.response_mode == "answer_only" else None

    base_model = Qwen3_5ForConditionalGeneration.from_pretrained(resolved_base_model, **model_kwargs)
    model = PeftModel.from_pretrained(base_model, str(args.adapter_dir))
    model.eval()

    prompt = sample.get("prompt")
    if args.response_mode == "answer_only" and {"video_a", "video_b"} <= sample.keys():
        prompt = build_answer_only_prompt(sample, fps=args.fps)
    if prompt is None:
        raise ValueError("Sample does not contain a usable prompt payload.")

    inputs = processor.apply_chat_template(
        prompt,
        chat_template=chat_template,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

    start = time.perf_counter()
    if args.response_mode == "answer_only":
        with torch.no_grad():
            outputs = model(**inputs, logits_to_keep=1)
        generated = None
    else:
        generated = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.temperature > 0,
            temperature=args.temperature,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
        )
    elapsed = time.perf_counter() - start

    prediction = None
    choice_logit_a = None
    choice_logit_b = None
    if args.response_mode == "answer_only":
        next_token_logits = outputs.logits[:, -1, :][0]
        choice_logit_a = float(next_token_logits[choice_token_ids["A"]].item())
        choice_logit_b = float(next_token_logits[choice_token_ids["B"]].item())
        prediction = "A" if choice_logit_a >= choice_logit_b else "B"
        answer = prediction
    else:
        prompt_len = inputs["input_ids"].shape[1]
        trimmed = generated[:, prompt_len:]
        answer = processor.batch_decode(trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]

    payload = {
        "base_model": resolved_base_model,
        "adapter_dir": str(args.adapter_dir.resolve()),
        "dataset_jsonl": str(args.dataset_jsonl.resolve()),
        "sample_index": args.sample_index,
        "response_mode": args.response_mode,
        "answer": answer,
        "prediction": prediction,
        "latency_sec": round(elapsed, 4),
        "choice_logit_a": round(choice_logit_a, 6) if choice_logit_a is not None else None,
        "choice_logit_b": round(choice_logit_b, 6) if choice_logit_b is not None else None,
        "label": sample.get("label"),
        "video_a": sample.get("video_a"),
        "video_b": sample.get("video_b"),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
