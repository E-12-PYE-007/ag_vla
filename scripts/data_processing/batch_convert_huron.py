from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from flow_head.huron_conversion import DEFAULT_TARGET_DISTANCES, convert_huron_bag_to_training_data


def parse_distances(text: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in text.split(",") if item.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch convert HuRoN/SACSoN bags into processed_mixed format.")
    parser.add_argument("--raw-root", type=Path, required=True, help="Root containing HuRoN raw folder/bags.")
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--pattern", default="*.bag")
    parser.add_argument("--image-topic", default="/fisheye_image/compressed")
    parser.add_argument("--odom-topic", default="/odometry")
    parser.add_argument("--image-stride", type=int, default=5)
    parser.add_argument("--max-pose-time-error", type=float, default=0.20)
    parser.add_argument("--max-pose-jump", type=float, default=1.0)
    parser.add_argument("--max-time-gap", type=float, default=2.0)
    parser.add_argument("--max-abs-yaw", type=float, default=3.141592653589793)
    parser.add_argument("--target-distances", type=parse_distances, default=DEFAULT_TARGET_DISTANCES)
    parser.add_argument("--max-bags", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def trajectory_name_for_bag(bag: Path, raw_root: Path) -> str:
    rel = bag.relative_to(raw_root).with_suffix("")
    return "_".join(rel.parts)


def main() -> None:
    args = parse_args()
    bags = sorted(args.raw_root.rglob(args.pattern))
    if args.max_bags is not None:
        bags = bags[: args.max_bags]
    if not bags:
        raise SystemExit(f"No HuRoN bags found in {args.raw_root} matching {args.pattern}")

    args.out_root.mkdir(parents=True, exist_ok=True)
    processed = []
    failed = []
    total_samples = 0
    for bag in bags:
        name = trajectory_name_for_bag(bag, args.raw_root)
        out_dir = args.out_root / name
        if (out_dir / "trajectory.npz").exists() and not args.overwrite:
            print(f"Skipping already processed bag: {bag}")
            processed.append({"bag": str(bag), "out_dir": str(out_dir), "skipped": True})
            continue
        print(f"Converting {bag}")
        try:
            metadata = convert_huron_bag_to_training_data(
                bag=bag,
                image_topic=args.image_topic,
                odom_topic=args.odom_topic,
                out_dir=out_dir,
                target_distances=args.target_distances,
                image_stride=args.image_stride,
                max_pose_time_error=args.max_pose_time_error,
                max_pose_jump=args.max_pose_jump,
                max_time_gap=args.max_time_gap,
                max_abs_yaw=args.max_abs_yaw,
            )
            samples = int(metadata["num_saved_samples"])
            total_samples += samples
            processed.append({"bag": str(bag), "out_dir": str(out_dir), "samples": samples})
            print(f"  saved {samples} samples")
        except Exception as exc:
            print(f"FAILED {bag}: {exc}")
            failed.append({"bag": str(bag), "error": str(exc)})

    summary = {
        "raw_root": str(args.raw_root),
        "out_root": str(args.out_root),
        "processed": processed,
        "failed": failed,
        "total_samples": total_samples,
    }
    with (args.out_root / "batch_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved {args.out_root / 'batch_summary.json'}")
    print(f"Processed={len(processed)} failed={len(failed)} total_samples={total_samples}")


if __name__ == "__main__":
    main()
