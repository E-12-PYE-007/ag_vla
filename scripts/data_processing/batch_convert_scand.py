from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from flow_head.scand_conversion import convert_bag_to_training_data, inspect_bag_topics, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch convert SCAND/Jackal bags.")
    parser.add_argument("--bags-dir", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--image-topic", default="auto")
    parser.add_argument("--odom-topic", default="auto")
    parser.add_argument("--pattern", default="A_Jackal_*.bag")
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--waypoint-dt", type=float, default=0.5)
    parser.add_argument("--image-stride", type=int, default=5)
    parser.add_argument("--sync-threshold", type=float, default=0.10)
    parser.add_argument("--max-final-distance", type=float, default=5.0)
    parser.add_argument("--max-abs-yaw", type=float, default=3.14)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bags = sorted(args.bags_dir.glob(args.pattern))
    if not bags:
        raise SystemExit(f"No bags found in {args.bags_dir} matching {args.pattern}")

    processed = []
    failed = []
    total_samples = 0
    args.out_root.mkdir(parents=True, exist_ok=True)

    for bag in bags:
        out_dir = args.out_root / bag.stem
        trajectory_path = out_dir / "trajectory.npz"
        if trajectory_path.exists() and not args.overwrite:
            print(f"Skipping already processed bag: {bag}")
            processed.append({"bag": str(bag), "out_dir": str(out_dir), "skipped": True})
            continue

        print(f"Converting {bag}")
        try:
            image_topic = args.image_topic
            odom_topic = args.odom_topic
            topic_summary = None
            if image_topic == "auto" or odom_topic == "auto":
                topic_summary = inspect_bag_topics(bag)
                if image_topic == "auto":
                    image_topic = topic_summary["recommended"].get("image_topic")
                if odom_topic == "auto":
                    odom_topic = topic_summary["recommended"].get("odom_topic")
                if not image_topic or not odom_topic:
                    raise ValueError(
                        "Could not auto-detect required SCAND topics. "
                        f"image_topic={image_topic}, odom_topic={odom_topic}"
                    )
                write_json(out_dir / "topic_summary.json", topic_summary)
                print(f"  auto image_topic={image_topic}")
                print(f"  auto odom_topic={odom_topic}")

            metadata = convert_bag_to_training_data(
                bag=bag,
                image_topic=image_topic,
                odom_topic=odom_topic,
                out_dir=out_dir,
                horizon=args.horizon,
                waypoint_dt=args.waypoint_dt,
                image_stride=args.image_stride,
                sync_threshold=args.sync_threshold,
                max_final_distance=args.max_final_distance,
                max_abs_yaw=args.max_abs_yaw,
            )
            processed.append(
                {
                    "bag": str(bag),
                    "out_dir": str(out_dir),
                    "samples": metadata["num_saved_samples"],
                    "image_topic": image_topic,
                    "odom_topic": odom_topic,
                }
            )
            total_samples += int(metadata["num_saved_samples"])
        except Exception as exc:
            print(f"FAILED {bag}: {exc}")
            failed.append({"bag": str(bag), "error": str(exc)})

    summary = {
        "processed": processed,
        "failed": failed,
        "total_samples": total_samples,
    }
    write_json(args.out_root / "batch_summary.json", summary)
    print(f"Saved batch summary to {args.out_root / 'batch_summary.json'}")
    print(f"Processed={len(processed)} failed={len(failed)} total_samples={total_samples}")


if __name__ == "__main__":
    main()
