from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from flow_head.huron_conversion import DEFAULT_TARGET_DISTANCES, convert_huron_bag_to_training_data


def parse_distances(text: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in text.split(",") if item.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert one Huron ROS1 bag into distance-resampled waypoint labels.")
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--image-topic", default="/fisheye_image/compressed")
    parser.add_argument("--odom-topic", default="/odometry")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--image-stride", type=int, default=5)
    parser.add_argument("--max-pose-time-error", type=float, default=0.20)
    parser.add_argument("--max-pose-jump", type=float, default=1.0)
    parser.add_argument("--max-time-gap", type=float, default=2.0)
    parser.add_argument("--max-abs-yaw", type=float, default=3.141592653589793)
    parser.add_argument(
        "--target-distances",
        type=parse_distances,
        default=DEFAULT_TARGET_DISTANCES,
        help="Comma-separated cumulative future distances in metres.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = convert_huron_bag_to_training_data(
        bag=args.bag,
        image_topic=args.image_topic,
        odom_topic=args.odom_topic,
        out_dir=args.out_dir,
        target_distances=args.target_distances,
        image_stride=args.image_stride,
        max_pose_time_error=args.max_pose_time_error,
        max_pose_jump=args.max_pose_jump,
        max_time_gap=args.max_time_gap,
        max_abs_yaw=args.max_abs_yaw,
    )
    print(f"Saved {metadata['num_saved_samples']} samples to {args.out_dir}")
    print(f"trajectory: {args.out_dir / 'trajectory.npz'}")
    print(f"metadata: {args.out_dir / 'metadata.json'}")
    print(f"path_length_m: {metadata['path_length_m']:.3f}")
    print(
        "pose_time_error mean/median: "
        f"{metadata['mean_pose_time_error']:.3f}/{metadata['median_pose_time_error']:.3f} s"
    )
    print(f"mean endpoint distance: {metadata['mean_waypoint_endpoint_distance']:.3f} m")
    print(f"mean time to final distance: {metadata['mean_time_to_final_distance']:.3f} s")
    print(f"rejected: {metadata['rejected']}")


if __name__ == "__main__":
    main()
