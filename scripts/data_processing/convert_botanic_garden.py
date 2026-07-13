from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from flow_head.botanic_garden_conversion import (
    DEFAULT_BOTANIC_IMAGE_TOPIC,
    DEFAULT_BOTANIC_POSE_TOPIC,
    DEFAULT_TARGET_DISTANCES,
    convert_botanic_garden_bag_to_training_data,
    convert_botanic_garden_files_to_training_data,
)


def parse_distances(text: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in text.split(",") if item.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert BotanicGarden data into distance-resampled waypoint labels.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--bag", type=Path, help="BotanicGarden ROS bag containing image topic and /gt_poses.")
    source.add_argument("--image-dir", type=Path, help="Raw/exported image directory. Requires --timestamps-file and --tum.")
    parser.add_argument("--timestamps-file", type=Path, default=None, help="Image timestamp file for --image-dir mode.")
    parser.add_argument("--tum", type=Path, default=None, help="TUM trajectory file: timestamp x y z qx qy qz qw.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--sequence-name", default=None)
    parser.add_argument("--image-topic", default=DEFAULT_BOTANIC_IMAGE_TOPIC)
    parser.add_argument("--pose-topic", default=DEFAULT_BOTANIC_POSE_TOPIC)
    parser.add_argument("--image-stride", type=int, default=10)
    parser.add_argument("--max-pose-time-error", type=float, default=0.05)
    parser.add_argument("--max-pose-jump", type=float, default=1.0)
    parser.add_argument("--max-time-gap", type=float, default=1.0)
    parser.add_argument("--max-abs-yaw", type=float, default=3.141592653589793)
    parser.add_argument("--copy-images", action="store_true", help="Copy raw images into out-dir/images in --image-dir mode.")
    parser.add_argument(
        "--target-distances",
        type=parse_distances,
        default=DEFAULT_TARGET_DISTANCES,
        help="Comma-separated cumulative future distances in metres.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.bag is not None:
        metadata = convert_botanic_garden_bag_to_training_data(
            bag=args.bag,
            out_dir=args.out_dir,
            image_topic=args.image_topic,
            pose_topic=args.pose_topic,
            target_distances=args.target_distances,
            image_stride=args.image_stride,
            max_pose_time_error=args.max_pose_time_error,
            max_pose_jump=args.max_pose_jump,
            max_time_gap=args.max_time_gap,
            max_abs_yaw=args.max_abs_yaw,
        )
    else:
        if args.timestamps_file is None or args.tum is None:
            raise SystemExit("--image-dir mode requires --timestamps-file and --tum.")
        metadata = convert_botanic_garden_files_to_training_data(
            image_dir=args.image_dir,
            timestamps_file=args.timestamps_file,
            tum_trajectory=args.tum,
            out_dir=args.out_dir,
            sequence_name=args.sequence_name,
            target_distances=args.target_distances,
            image_stride=args.image_stride,
            max_pose_time_error=args.max_pose_time_error,
            max_pose_jump=args.max_pose_jump,
            max_time_gap=args.max_time_gap,
            max_abs_yaw=args.max_abs_yaw,
            copy_images=args.copy_images,
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
