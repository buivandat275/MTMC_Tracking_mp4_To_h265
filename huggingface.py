import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download


REPO_ID = "nvidia/PhysicalAI-SmartSpaces"
DATASET_ROOT = Path("MTMC_Tracking_2024") / "train"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download MTMC scenes from Hugging Face."
    )
    parser.add_argument(
        "--start-scene",
        type=int,
        default=11,
        help="First scene number to download. Default: 11",
    )
    parser.add_argument(
        "--end-scene",
        type=int,
        default=20,
        help="Last scene number to download. Default: 20",
    )
    parser.add_argument(
        "--output-root",
        default=str(DATASET_ROOT),
        help="Root output folder for downloaded scenes. Default: MTMC_Tracking_2024/train",
    )
    return parser.parse_args()


def scene_name(scene_number: int) -> str:
    return f"scene_{scene_number:03d}"


def download_scene(scene_number: int, output_root: Path, token: str | None) -> None:
    name = scene_name(scene_number)
    allow_pattern = f"MTMC_Tracking_2024/train/{name}/**"
    local_dir = output_root / name

    print(f"Downloading {name} -> {local_dir}")
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        allow_patterns=allow_pattern,
        local_dir=str(local_dir),
        token=token,
    )
    print(f"Finished {name}")


def main() -> int:
    args = parse_args()

    if args.start_scene > args.end_scene:
        raise SystemExit("--start-scene must be less than or equal to --end-scene")

    token = os.environ.get("HF_TOKEN")
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    print(
        f"Downloading scenes {scene_name(args.start_scene)} -> "
        f"{scene_name(args.end_scene)}"
    )
    print(f"Output root: {output_root}")
    print(f"HF token detected: {'yes' if token else 'no'}")

    for scene_number in range(args.start_scene, args.end_scene + 1):
        download_scene(scene_number, output_root, token)

    print("All requested scenes finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
