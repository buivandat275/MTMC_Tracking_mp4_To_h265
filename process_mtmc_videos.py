import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path


MAX_SECONDS = 10 * 60
DEFAULT_INPUT_DIR = r"D:\119\MTMC_Tracking_2024\train"


def to_long_path(path: Path) -> str:
    resolved = path.resolve()
    text = str(resolved)
    if text.startswith("\\\\?\\"):
        return text
    if text.startswith("\\\\"):
        return "\\\\?\\UNC\\" + text[2:]
    return "\\\\?\\" + text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Flatten MTMC video folders, trim videos longer than 10 minutes, and rename them sequentially."
    )
    parser.add_argument(
        "input_dir",
        nargs="?",
        default=DEFAULT_INPUT_DIR,
        help="Root folder that contains the training video tree. Default: D:\\119\\MTMC_Tracking_2024\\train",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Destination folder. Default: <input_dir>\\processed_videos",
    )
    parser.add_argument(
        "--date",
        default=None,
        help="Date part for output filenames, format YYYY-MM-DD. Default: today.",
    )
    parser.add_argument(
        "--time",
        dest="time_text",
        default=None,
        help="Time part for output filenames, format HH-MM. Default: current time.",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=23,
        help="x265 CRF value for trimmed/converted output. Lower means higher quality. Default: 23",
    )
    parser.add_argument(
        "--preset",
        default="medium",
        help="x265 preset. Default: medium",
    )
    return parser.parse_args()


def ffprobe_duration_seconds(ffprobe_path: str, file_path: Path) -> float:
    result = subprocess.run(
        [
            ffprobe_path,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            to_long_path(file_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(result.stdout.strip())


def gather_videos(root: Path) -> list[Path]:
    videos: list[Path] = []
    for path in root.rglob("*.mp4"):
        if ".cache" in path.parts:
            continue
        if path.name.lower() != "video.mp4":
            continue
        if path.is_file():
            videos.append(path)
    return sorted(videos, key=video_sort_key)


def video_sort_key(path: Path) -> tuple[int, int, str]:
    scene_num = 10**9
    camera_num = 10**9
    for part in path.parts:
        if part.startswith("scene_"):
            try:
                scene_num = int(part.split("_", 1)[1])
            except ValueError:
                pass
        elif part.startswith("camera_"):
            try:
                camera_num = int(part.split("_", 1)[1])
            except ValueError:
                pass
    return scene_num, camera_num, str(path).lower()


def output_name(index: int, date_text: str, time_text: str) -> str:
    return f"cam_{index:02d}_{date_text}_{time_text}.mp4"


def run_ffmpeg(
    ffmpeg_path: str,
    source: Path,
    destination: Path,
    duration_limit: float | None,
    crf: int,
    preset: str,
) -> bool:
    command = [
        ffmpeg_path,
        "-hide_banner",
        "-y",
        "-i",
        to_long_path(source),
    ]

    if duration_limit is not None:
        command += ["-t", f"{duration_limit:.3f}"]

    command += [
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-c:v",
        "libx265",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-tag:v",
        "hvc1",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-movflags",
        "+faststart",
        to_long_path(destination),
    ]

    result = subprocess.run(command)
    return result.returncode == 0


def main() -> int:
    args = parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    if not input_dir.is_dir():
        print(f"Input folder not found: {input_dir}", file=sys.stderr)
        return 1

    ffmpeg_path = "ffmpeg"
    ffprobe_path = "ffprobe"

    date_text = args.date or datetime.now().strftime("%Y-%m-%d")
    time_text = args.time_text or datetime.now().strftime("%H-%M")
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else input_dir.parent / "processed_videos"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    videos = gather_videos(input_dir)
    if not videos:
        print(f"No .mp4 files found under: {input_dir}")
        return 0

    print(f"Found {len(videos)} video file(s).")
    print(f"Output folder: {output_dir}")

    for index, source in enumerate(videos, start=1):
        destination = output_dir / output_name(index, date_text, time_text)
        if destination.exists():
            print(f"[{index}/{len(videos)}] Skipping existing file: {destination.name}")
            continue

        try:
            duration = ffprobe_duration_seconds(ffprobe_path, source)
        except Exception as exc:
            print(f"[{index}/{len(videos)}] Failed to read duration for {source}: {exc}", file=sys.stderr)
            continue

        trim_seconds = min(duration, MAX_SECONDS)
        print(
            f"[{index}/{len(videos)}] {source.name} -> {destination.name} "
            f"({duration/60:.2f} min, output {trim_seconds/60:.2f} min)"
        )

        ok = run_ffmpeg(
            ffmpeg_path=ffmpeg_path,
            source=source,
            destination=destination,
            duration_limit=trim_seconds if duration > MAX_SECONDS else None,
            crf=args.crf,
            preset=args.preset,
        )

        if not ok:
            if destination.exists():
                destination.unlink()
            print(f"Failed: {source}", file=sys.stderr)

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
