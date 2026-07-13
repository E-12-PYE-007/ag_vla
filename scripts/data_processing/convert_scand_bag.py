from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from flow_head.scand_conversion import convert_bag_to_training_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert one SCAND/Jackal ROS1 bag into flow-head labels.")
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--image-topic", required=True)
    parser.add_argument("--odom-topic", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--waypoint-dt", type=float, default=0.5)
    parser.add_argument("--image-stride", type=int, default=5)
    parser.add_argument("--sync-threshold", type=float, default=0.10)
    parser.add_argument("--max-final-distance", type=float, default=5.0)
    parser.add_argument("--max-abs-yaw", type=float, default=3.14)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = convert_bag_to_training_data(
        bag=args.bag,
        image_topic=args.image_topic,
        odom_topic=args.odom_topic,
        out_dir=args.out_dir,
        horizon=args.horizon,
        waypoint_dt=args.waypoint_dt,
        image_stride=args.image_stride,
        sync_threshold=args.sync_threshold,
        max_final_distance=args.max_final_distance,
        max_abs_yaw=args.max_abs_yaw,
    )
    print(f"Saved {metadata['num_saved_samples']} samples to {args.out_dir}")
    print(f"trajectory: {args.out_dir / 'trajectory.npz'}")
    print(f"metadata: {args.out_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()

