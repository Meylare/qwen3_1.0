from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval_ab_accuracy_s3 import ensure_local_video_path, get_s3_client
from prepare_smoke_dataset import (
    build_whisper_model,
    choose_smoke_subset,
    convert_rows,
    extract_video_side_info,
    load_jsonl,
    split_by_author,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Prepare GSPO train dataset from S3-backed pairs.')
    parser.add_argument('--input', type=Path, default=PROJECT_ROOT / 'data' / 'manifests' / 'train_test.jsonl')
    parser.add_argument('--output_dir', type=Path, default=PROJECT_ROOT / 'data' / 'train40_s3_prepared')
    parser.add_argument('--sideinfo_cache_dir', type=Path, default=PROJECT_ROOT / 'cache')
    parser.add_argument('--s3_cache_dir', type=Path, default=PROJECT_ROOT / 's3_cache')
    parser.add_argument('--smoke_pairs', type=int, default=40)
    parser.add_argument('--max_video_duration', type=float, default=45.0)
    parser.add_argument('--video_fps', type=float, default=2.0)
    parser.add_argument('--eval_ratio', type=float, default=0.05)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--whisper_model', default='large-v3')
    parser.add_argument('--whisper_device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--whisper_compute_type', default=None)
    parser.add_argument('--speech_segmentation_mode', choices=('off', 'inaspeech'), default='inaspeech')
    parser.add_argument('--speech_min_segment_sec', type=float, default=0.35)
    parser.add_argument('--speech_merge_gap_sec', type=float, default=0.25)
    parser.add_argument('--speech_keep_leading_trailing_pad_sec', type=float, default=0.10)
    parser.add_argument('--segmentation_fallback_to_full_audio', action='store_true', default=True)
    parser.add_argument('--no_segmentation_fallback_to_full_audio', action='store_false', dest='segmentation_fallback_to_full_audio')
    parser.add_argument('--music_genre_model', default='dima806/music_genres_classification')
    parser.add_argument('--music_genre_device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--skip_bad_pairs', action='store_true', default=True)
    parser.add_argument('--no_skip_bad_pairs', action='store_false', dest='skip_bad_pairs')
    parser.add_argument('--aws_profile', default=None)
    parser.add_argument('--aws_region', default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.sideinfo_cache_dir.mkdir(parents=True, exist_ok=True)
    args.s3_cache_dir.mkdir(parents=True, exist_ok=True)

    compute_type = args.whisper_compute_type
    if compute_type is None:
        compute_type = 'float16' if str(args.whisper_device).startswith('cuda') else 'int8'

    raw_rows = load_jsonl(args.input)
    subset = choose_smoke_subset(raw_rows, args.smoke_pairs, args.seed)
    whisper_model = build_whisper_model(args.whisper_model, args.whisper_device, compute_type)
    s3_client = get_s3_client(args.aws_profile, args.aws_region)

    processed_rows: List[Dict[str, Any]] = []
    pair_timings: List[float] = []
    video_timings: List[float] = []
    s3_cache_hits = 0
    side_cache_hits = 0
    segmentation_applied_count = 0
    segmentation_fallback_count = 0
    empty_transcript_count = 0
    segmentation_times: List[float] = []
    transcription_times_after_segmentation: List[float] = []
    speech_ratios: List[float] = []
    skipped_pairs = 0
    skipped_details: List[Dict[str, Any]] = []

    for idx, row in enumerate(subset):
        pair_start = time.perf_counter()
        try:
            local_a, s3_hit_a = ensure_local_video_path(row['video_a'], args.s3_cache_dir, s3_client)
            local_b, s3_hit_b = ensure_local_video_path(row['video_b'], args.s3_cache_dir, s3_client)
            s3_cache_hits += int(s3_hit_a) + int(s3_hit_b)

            meta_a = extract_video_side_info(
                video_path=local_a,
                cache_dir=args.sideinfo_cache_dir,
                max_duration=args.max_video_duration,
                whisper_model_name=args.whisper_model,
                whisper_model=whisper_model,
                speech_segmentation_mode=args.speech_segmentation_mode,
                speech_min_segment_sec=args.speech_min_segment_sec,
                speech_merge_gap_sec=args.speech_merge_gap_sec,
                speech_keep_leading_trailing_pad_sec=args.speech_keep_leading_trailing_pad_sec,
                segmentation_fallback_to_full_audio=args.segmentation_fallback_to_full_audio,
                music_genre_model=args.music_genre_model,
                music_genre_device=args.music_genre_device,
            )
            meta_b = extract_video_side_info(
                video_path=local_b,
                cache_dir=args.sideinfo_cache_dir,
                max_duration=args.max_video_duration,
                whisper_model_name=args.whisper_model,
                whisper_model=whisper_model,
                speech_segmentation_mode=args.speech_segmentation_mode,
                speech_min_segment_sec=args.speech_min_segment_sec,
                speech_merge_gap_sec=args.speech_merge_gap_sec,
                speech_keep_leading_trailing_pad_sec=args.speech_keep_leading_trailing_pad_sec,
                segmentation_fallback_to_full_audio=args.segmentation_fallback_to_full_audio,
                music_genre_model=args.music_genre_model,
                music_genre_device=args.music_genre_device,
            )
        except Exception as exc:
            if not args.skip_bad_pairs:
                raise
            skipped_pairs += 1
            skipped_details.append(
                {
                    'subset_index': idx,
                    'video_a': row.get('video_a'),
                    'video_b': row.get('video_b'),
                    'error_type': type(exc).__name__,
                    'error_message': str(exc),
                }
            )
            print(
                f"[{idx + 1}/{len(subset)}] skipped pair | "
                f"video_a={row.get('video_a')} | video_b={row.get('video_b')} | "
                f"error={type(exc).__name__}: {exc}"
            )
            continue

        side_cache_hits += int(meta_a.get('cache_hit', False)) + int(meta_b.get('cache_hit', False))
        video_timings.extend([meta_a['total_sec'], meta_b['total_sec']])
        for meta in (meta_a, meta_b):
            segmentation_applied_count += int(bool(meta.get('segmentation_applied', False)))
            segmentation_fallback_count += int(bool(meta.get('segmentation_fallback', False)))
            empty_transcript_count += int(not bool(str(meta.get('transcript', '')).strip()))
            segmentation_times.append(float(meta.get('segmentation_sec', 0.0) or 0.0))
            transcription_times_after_segmentation.append(float(meta.get('transcription_sec', 0.0) or 0.0))
            speech_ratios.append(float(meta.get('segmentation_speech_ratio', 0.0) or 0.0))

        enriched = dict(row)
        enriched['video_a'] = str(local_a)
        enriched['video_b'] = str(local_b)
        enriched['transcript_a'] = meta_a['transcript']
        enriched['transcript_b'] = meta_b['transcript']
        enriched['transcript_segments_a'] = meta_a.get('transcript_segments', [])
        enriched['transcript_segments_b'] = meta_b.get('transcript_segments', [])
        enriched['lyrics_a'] = meta_a.get('lyrics_transcript', '')
        enriched['lyrics_b'] = meta_b.get('lyrics_transcript', '')
        enriched['lyrics_segments_a'] = meta_a.get('lyrics_transcript_segments', [])
        enriched['lyrics_segments_b'] = meta_b.get('lyrics_transcript_segments', [])
        enriched['audio_summary_a'] = meta_a['audio_summary']
        enriched['audio_summary_b'] = meta_b['audio_summary']
        processed_rows.append(enriched)
        pair_timings.append(time.perf_counter() - pair_start)

        print(
            f"[{idx + 1}/{len(subset)}] processed pair | "
            f"A s3_cache={s3_hit_a} side_cache={meta_a.get('cache_hit', False)} | "
            f"B s3_cache={s3_hit_b} side_cache={meta_b.get('cache_hit', False)} | "
            f"pair_sec={pair_timings[-1]:.2f}"
        )

    train_raw, eval_raw = split_by_author(processed_rows, args.eval_ratio, args.seed)
    train_rows = convert_rows(train_raw, video_fps=args.video_fps)
    eval_rows = convert_rows(eval_raw, video_fps=args.video_fps)

    train_path = args.output_dir / 'train.jsonl'
    eval_path = args.output_dir / 'eval.jsonl'
    metrics_path = args.output_dir / 'preprocess_metrics.json'

    write_jsonl(train_path, train_rows)
    write_jsonl(eval_path, eval_rows)

    metrics = {
        'input_path': str(args.input.resolve()),
        'smoke_pairs_requested': args.smoke_pairs,
        'pairs_processed': len(processed_rows),
        'pairs_skipped': skipped_pairs,
        'skip_bad_pairs': args.skip_bad_pairs,
        'train_rows': len(train_rows),
        'eval_rows': len(eval_rows),
        'max_video_duration': args.max_video_duration,
        'video_fps': args.video_fps,
        'whisper_model': args.whisper_model,
        'whisper_device': args.whisper_device,
        'whisper_compute_type': compute_type,
        'speech_segmentation_mode': args.speech_segmentation_mode,
        'music_genre_model': args.music_genre_model,
        'music_genre_device': args.music_genre_device,
        'speech_min_segment_sec': args.speech_min_segment_sec,
        'speech_merge_gap_sec': args.speech_merge_gap_sec,
        'speech_keep_leading_trailing_pad_sec': args.speech_keep_leading_trailing_pad_sec,
        'segmentation_fallback_to_full_audio': args.segmentation_fallback_to_full_audio,
        'avg_pair_preprocess_sec': round(statistics.mean(pair_timings), 4) if pair_timings else 0.0,
        'avg_video_preprocess_sec': round(statistics.mean(video_timings), 4) if video_timings else 0.0,
        'segmentation_videos': segmentation_applied_count,
        'segmentation_fallback_videos': segmentation_fallback_count,
        'empty_transcript_videos': empty_transcript_count,
        'avg_segmentation_sec': round(statistics.mean(segmentation_times), 4) if segmentation_times else 0.0,
        'avg_transcription_after_segmentation_sec': round(statistics.mean(transcription_times_after_segmentation), 4)
        if transcription_times_after_segmentation
        else 0.0,
        'avg_segmentation_speech_ratio': round(statistics.mean(speech_ratios), 4) if speech_ratios else 0.0,
        's3_cache_hits': s3_cache_hits,
        's3_cache_misses': max(0, len(video_timings) - s3_cache_hits),
        'sideinfo_cache_hits': side_cache_hits,
        'sideinfo_cache_misses': max(0, len(video_timings) - side_cache_hits),
        'train_path': str(train_path.resolve()),
        'eval_path': str(eval_path.resolve()),
    }
    if skipped_details:
        metrics['skipped_examples_preview'] = skipped_details[:10]
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
