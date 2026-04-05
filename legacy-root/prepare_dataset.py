import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

URL_RE = re.compile(r"https?://\S+")


def _read_text_lines(path: Path) -> List[str]:
    for enc in ("utf-8-sig", "utf-8", "cp1251"):
        try:
            return path.read_text(encoding=enc).splitlines()
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("utf-8", b"", 0, 1, f"Failed to decode {path}")


def extract_url(line: str) -> Optional[str]:
    match = URL_RE.search(line)
    return match.group(0) if match else None


def url_to_filename(url: str, default_ext: str = ".mp4") -> str:
    parsed = urlparse(url)
    ext = Path(parsed.path).suffix or default_ext
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]
    return f"{digest}{ext}"


def download_url(url: str, dest: Path, timeout: int = 30) -> Tuple[bool, str]:
    if dest.exists() and dest.stat().st_size > 0:
        return True, "exists"

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest.with_suffix(dest.suffix + ".part")

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept": "*/*",
    }
    req = Request(url, headers=headers)

    try:
        with urlopen(req, timeout=timeout) as resp, open(tmp_path, "wb") as f:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
        tmp_path.replace(dest)
        return True, "downloaded"
    except (HTTPError, URLError, TimeoutError) as exc:
        if tmp_path.exists():
            tmp_path.unlink()
        return False, f"error: {exc}"


def iter_batch_items(batch_path: Path) -> Iterable[Dict]:
    with open(batch_path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def build_train_samples(
    batch_path: Path,
    videos_dir: Path,
    require_files: bool = True,
) -> Tuple[List[Dict], Dict[str, int]]:
    samples: List[Dict] = []
    stats = {
        "total_pairs": 0,
        "written": 0,
        "missing_roles": 0,
        "missing_files": 0,
        "low_ratio": 0,
    }

    for obj in iter_batch_items(batch_path):
        author_context = obj.get("niche") or obj.get("category") or ""
        for pair in obj.get("pairs", []):
            stats["total_pairs"] += 1
            items = pair.get("items", [])
            hit = next((i for i in items if i.get("role") == "hit"), None)
            normal = next(
                (i for i in items if i.get("role") in ("nearest_normal", "normal")),
                None,
            )
            if not hit or not normal:
                stats["missing_roles"] += 1
                continue

            views_a = hit.get("views")
            views_b = normal.get("views")
            if views_a is None or views_b is None:
                stats["missing_roles"] += 1
                continue

            ratio = max(views_a, views_b) / (min(views_a, views_b) + 1)
            if ratio < 3.0:
                stats["low_ratio"] += 1
                continue

            url_a = hit.get("video_url")
            url_b = normal.get("video_url")
            if not url_a or not url_b:
                stats["missing_roles"] += 1
                continue

            path_a = (videos_dir / url_to_filename(url_a)).resolve()
            path_b = (videos_dir / url_to_filename(url_b)).resolve()

            if require_files and (not path_a.exists() or not path_b.exists()):
                stats["missing_files"] += 1
                continue

            samples.append(
                {
                    "video_a": str(path_a),
                    "video_b": str(path_b),
                    "views_a": views_a,
                    "views_b": views_b,
                    "author_context": author_context,
                }
            )
            stats["written"] += 1

    return samples, stats


def write_jsonl(path: Path, rows: Iterable[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download videos and build train.jsonl dataset."
    )
    parser.add_argument(
        "--links",
        default="videos_links.txt",
        help="Path to videos_links.txt",
    )
    parser.add_argument(
        "--batch",
        default="batch_filtered.jsonl",
        help="Path to batch_filtered.jsonl",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Root data directory (videos go to data/pair_vid)",
    )
    parser.add_argument(
        "--out",
        default="train.jsonl",
        help="Output train.jsonl path",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip downloading videos (assume already present)",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Allow samples even if video files are missing",
    )
    args = parser.parse_args()

    links_path = Path(args.links)
    batch_path = Path(args.batch)
    data_dir = Path(args.data_dir)
    videos_dir = data_dir / "pair_vid"
    out_path = Path(args.out)
    videos_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_download:
        lines = _read_text_lines(links_path)
        urls = []
        for line in lines:
            url = extract_url(line)
            if url:
                urls.append(url)

        unique_urls = sorted(set(urls))
        ok = 0
        fail = 0
        for url in unique_urls:
            dest = videos_dir / url_to_filename(url)
            success, status = download_url(url, dest)
            if success:
                ok += 1
            else:
                fail += 1
                print(f"Download failed: {status} | {url}", file=sys.stderr)
        print(f"Downloads: {ok} ok, {fail} failed, {len(unique_urls)} total")

    samples, stats = build_train_samples(
        batch_path,
        videos_dir,
        require_files=not args.allow_missing,
    )
    write_jsonl(out_path, samples)

    print(
        "Dataset stats: "
        + ", ".join(f"{k}={v}" for k, v in stats.items())
        + f", output={out_path.resolve()}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
