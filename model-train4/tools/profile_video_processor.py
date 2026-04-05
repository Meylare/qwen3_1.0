from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from transformers import AutoProcessor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train_gspo_qwen35_smoke import DEFAULT_MODEL_DIR, ensure_chat_template, resolve_model_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile Qwen video processor on a few example videos.")
    parser.add_argument("--model_name", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--video", action="append", dest="videos", required=True)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "output" / "video_processor_profile.json",
    )
    return parser.parse_args()


def ffprobe_video(path: str) -> Dict[str, Any]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,duration",
        "-of",
        "json",
        path,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    payload = json.loads(proc.stdout)
    stream = payload["streams"][0]
    return {
        "width": int(stream.get("width") or 0),
        "height": int(stream.get("height") or 0),
        "duration_sec": float(stream.get("duration") or 0.0),
        "r_frame_rate": stream.get("r_frame_rate"),
        "avg_frame_rate": stream.get("avg_frame_rate"),
        "nb_frames": stream.get("nb_frames"),
    }


def build_duplicate_prompt(video_path: str, fps: float) -> List[Dict[str, Any]]:
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Ты анализируешь техническую нагрузку мультимодального prompt. "
                        "Игнорируй смысл видео. Ответ не нужен."
                    ),
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video_path, "fps": fps},
                {"type": "video", "video": video_path, "fps": fps},
                {
                    "type": "text",
                    "text": (
                        "Video A и Video B здесь одинаковые. "
                        "Этот prompt нужен только для замера времени обработки и размера контекста."
                    ),
                },
            ],
        },
    ]


def count_mm_tokens(mm_token_type_ids: Any) -> Dict[str, Any]:
    if mm_token_type_ids is None:
        return {"present": False}
    flat = mm_token_type_ids.reshape(-1).tolist()
    counts: Dict[int, int] = {}
    for value in flat:
        counts[int(value)] = counts.get(int(value), 0) + 1
    mm_nonzero = sum(v for k, v in counts.items() if k != 0)
    return {
        "present": True,
        "value_counts": counts,
        "nonzero_count": mm_nonzero,
    }


def main() -> None:
    args = parse_args()
    resolved_model = resolve_model_path(args.model_name)
    processor = AutoProcessor.from_pretrained(resolved_model, trust_remote_code=True)
    ensure_chat_template(processor, resolved_model)

    video_processor = getattr(processor, "video_processor", None) or getattr(processor, "image_processor", None)
    patch_size = getattr(video_processor, "patch_size", None)
    temporal_patch_size = getattr(video_processor, "temporal_patch_size", None)
    merge_size = getattr(video_processor, "merge_size", None)

    report: Dict[str, Any] = {
        "model_name": resolved_model,
        "processor_class": processor.__class__.__name__,
        "video_processor_class": None if video_processor is None else video_processor.__class__.__name__,
        "video_processor_config": {
            "size": getattr(video_processor, "size", None),
            "patch_size": patch_size,
            "temporal_patch_size": temporal_patch_size,
            "merge_size": merge_size,
            "do_resize": getattr(video_processor, "do_resize", None),
            "do_rescale": getattr(video_processor, "do_rescale", None),
            "do_normalize": getattr(video_processor, "do_normalize", None),
            "fps": getattr(video_processor, "fps", None),
            "max_frames": getattr(video_processor, "max_frames", None),
            "min_frames": getattr(video_processor, "min_frames", None),
        },
        "repeats": args.repeats,
        "fps": args.fps,
        "results": [],
    }

    for raw_video in args.videos:
        video_path = str(Path(raw_video).expanduser().resolve())
        prompt = build_duplicate_prompt(video_path, fps=args.fps)
        ff = ffprobe_video(video_path)

        timings: List[float] = []
        last_encoded = None
        for _ in range(args.repeats):
            start = time.perf_counter()
            encoded = processor.apply_chat_template(
                prompt,
                chat_template=getattr(processor, "chat_template", None),
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            timings.append(time.perf_counter() - start)
            last_encoded = encoded

        assert last_encoded is not None
        encoded = last_encoded

        video_grid_thw = encoded.get("video_grid_thw")
        grid_list = video_grid_thw.tolist() if video_grid_thw is not None else None
        mm_info = count_mm_tokens(encoded.get("mm_token_type_ids"))
        input_tokens = int(encoded["input_ids"].shape[1])
        pixel_values_videos_shape = (
            list(encoded["pixel_values_videos"].shape) if "pixel_values_videos" in encoded else None
        )

        estimated_processed_resolution = None
        estimated_sampled_frames = None
        estimated_context_video_tokens_total = None
        estimated_context_video_tokens_per_video = None
        if grid_list and patch_size and merge_size and temporal_patch_size:
            first = grid_list[0]
            estimated_processed_resolution = {
                "height_px": int(first[1] * patch_size),
                "width_px": int(first[2] * patch_size),
            }
            estimated_sampled_frames = int(first[0] * temporal_patch_size)
            estimated_context_video_tokens_total = sum(
                int(item[0] * item[1] * item[2] / (merge_size * merge_size)) for item in grid_list
            )
            estimated_context_video_tokens_per_video = [
                int(item[0] * item[1] * item[2] / (merge_size * merge_size)) for item in grid_list
            ]

        report["results"].append(
            {
                "video_path": video_path,
                "original_video": ff,
                "processor_sec_runs": [round(x, 4) for x in timings],
                "processor_sec_avg": round(statistics.mean(timings), 4),
                "processor_sec_median": round(statistics.median(timings), 4),
                "input_ids_tokens_total": input_tokens,
                "mm_token_type_info": mm_info,
                "text_tokens_estimated": (
                    input_tokens - int(mm_info["nonzero_count"]) if mm_info.get("present") else None
                ),
                "pixel_values_videos_shape": pixel_values_videos_shape,
                "video_grid_thw": grid_list,
                "estimated_processed_resolution": estimated_processed_resolution,
                "estimated_sampled_frames_per_video": estimated_sampled_frames,
                "estimated_context_video_tokens_total": estimated_context_video_tokens_total,
                "estimated_context_video_tokens_per_video": estimated_context_video_tokens_per_video,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
