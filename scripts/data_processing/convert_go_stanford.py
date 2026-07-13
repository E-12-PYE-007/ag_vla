from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from flow_head.go_stanford_conversion import convert_go_stanford_sequence, discover_sequences


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert extracted GO Stanford 2 sequences into processed_mixed format.")
    parser.add_argument("--root", type=Path, required=True, help="Extracted gs2_withres directory.")
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--dataset-name", default="go_stanford_2")
    parser.add_argument("--side", choices=["left", "right", "auto"], default="left")
    parser.add_argument("--include-flipped", action="store_true")
    parser.add_argument("--max-sequences", type=int)
    parser.add_argument("--sequence-id", action="append", help="Convert only this sequence id. May be passed more than once.")
    parser.add_argument("--frame-dt", type=float, default=0.2, help="Assumed seconds per frame for integrating velocity commands.")
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--waypoint-dt", type=float, default=0.5)
    parser.add_argument("--sampling-mode", choices=["distance", "frame", "time"], default="distance")
    parser.add_argument("--target-spacing-m", type=float, default=0.25)
    parser.add_argument("--distance-tolerance-m", type=float)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--min-final-distance", type=float, default=0.5)
    parser.add_argument("--max-final-distance", type=float, default=4.0)
    parser.add_argument("--max-abs-yaw", type=float, default=3.14)
    parser.add_argument("--image-stride", type=int, default=5)
    parser.add_argument("--link-images", action="store_true", help="Symlink images instead of copying when possible.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sequences = discover_sequences(args.root)
    if not args.include_flipped:
        sequences = [seq for seq in sequences if not seq.is_flipped]
    if args.sequence_id:
        wanted = set(args.sequence_id)
        sequences = [seq for seq in sequences if seq.sequence_id in wanted]
    if args.max_sequences is not None:
        sequences = sequences[: args.max_sequences]
    if not sequences:
        raise SystemExit("No GO Stanford sequences selected.")

    args.out_root.mkdir(parents=True, exist_ok=True)
    processed = []
    failed = []
    total_samples = 0
    for seq in sequences:
        out_dir = args.out_root / seq.sequence_id
        trajectory_path = out_dir / "trajectory.npz"
        if trajectory_path.exists() and not args.overwrite:
            print(f"Skipping already processed sequence: {seq.sequence_id}")
            processed.append({"sequence_id": seq.sequence_id, "out_dir": str(out_dir), "skipped": True})
            continue
        print(f"Converting {seq.sequence_id}")
        try:
            metadata = convert_go_stanford_sequence(
                root=args.root,
                sequence=seq,
                out_dir=out_dir,
                side=args.side,
                dataset_name=args.dataset_name,
                frame_dt=args.frame_dt,
                horizon=args.horizon,
                waypoint_dt=args.waypoint_dt,
                sampling_mode=args.sampling_mode,
                target_spacing_m=args.target_spacing_m,
                distance_tolerance_m=args.distance_tolerance_m,
                frame_stride=args.frame_stride,
                min_final_distance=args.min_final_distance,
                max_final_distance=args.max_final_distance,
                max_abs_yaw=args.max_abs_yaw,
                image_stride=args.image_stride,
                copy_images=not args.link_images,
            )
            samples = int(metadata["num_saved_samples"])
            total_samples += samples
            processed.append({"sequence_id": seq.sequence_id, "out_dir": str(out_dir), "samples": samples})
            print(f"  saved {samples} samples")
        except Exception as exc:
            print(f"FAILED {seq.sequence_id}: {exc}")
            failed.append({"sequence_id": seq.sequence_id, "error": str(exc)})

    summary = {
        "root": str(args.root),
        "out_root": str(args.out_root),
        "processed": processed,
        "failed": failed,
        "total_samples": total_samples,
        "settings": {
            "side": args.side,
            "include_flipped": args.include_flipped,
            "frame_dt": args.frame_dt,
            "sampling_mode": args.sampling_mode,
            "target_spacing_m": args.target_spacing_m,
            "image_stride": args.image_stride,
        },
    }
    with (args.out_root / "batch_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved {args.out_root / 'batch_summary.json'}")
    print(f"Processed={len(processed)} failed={len(failed)} total_samples={total_samples}")


if __name__ == "__main__":
    main()
