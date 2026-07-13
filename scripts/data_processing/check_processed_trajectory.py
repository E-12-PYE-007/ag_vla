from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check one unified processed navigation trajectory.")
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--waypoint-dim", type=int, default=3)
    parser.add_argument("--min-samples", type=int, default=20)
    parser.add_argument("--large-final-distance", type=float, default=20.0)
    return parser.parse_args()


def stats(name: str, values: np.ndarray) -> None:
    values = np.asarray(values)
    print(
        f"{name}: mean={np.mean(values):.4f} std={np.std(values):.4f} "
        f"min={np.min(values):.4f} max={np.max(values):.4f}"
    )


def scalar_string(value: np.ndarray | str) -> str:
    arr = np.asarray(value)
    if arr.shape == ():
        return str(arr.item())
    return str(value)


def main() -> None:
    args = parse_args()
    data = np.load(args.npz, allow_pickle=True)
    required = ["image_paths", "times", "position", "yaw", "velocity", "target_waypoints"]
    missing = [key for key in required if key not in data.files]
    if missing:
        raise SystemExit(f"Missing required arrays: {missing}")

    image_paths = data["image_paths"]
    times = data["times"]
    position = data["position"]
    yaw = data["yaw"]
    velocity = data["velocity"]
    target_waypoints = data["target_waypoints"]

    dataset_name = scalar_string(data["dataset_name"]) if "dataset_name" in data.files else "unknown"
    trajectory_name = scalar_string(data["trajectory_name"]) if "trajectory_name" in data.files else args.npz.parent.name
    print(f"dataset_name: {dataset_name}")
    print(f"trajectory_name: {trajectory_name}")
    for key in data.files:
        print(f"{key} shape: {data[key].shape}")

    t = len(image_paths)
    errors = []
    if len(times) != t:
        errors.append(f"times length {len(times)} != image_paths length {t}")
    if position.shape != (t, 2):
        errors.append(f"position shape {position.shape} != ({t}, 2)")
    if yaw.shape != (t,):
        errors.append(f"yaw shape {yaw.shape} != ({t},)")
    if velocity.shape != (t, 2):
        errors.append(f"velocity shape {velocity.shape} != ({t}, 2)")
    if target_waypoints.shape != (t, args.horizon, args.waypoint_dim):
        errors.append(
            f"target_waypoints shape {target_waypoints.shape} "
            f"!= ({t}, {args.horizon}, {args.waypoint_dim})"
        )
    for key, arr in {
        "times": times,
        "position": position,
        "yaw": yaw,
        "velocity": velocity,
        "target_waypoints": target_waypoints,
    }.items():
        if not np.isfinite(arr).all():
            errors.append(f"{key} contains NaN or Inf")

    if errors:
        print("\nERRORS:")
        for error in errors:
            print(f"  {error}")
        raise SystemExit(1)

    print("\nFirst waypoint chunk:")
    print(target_waypoints[0])

    print("\nWaypoint statistics:")
    stats("delta_x", target_waypoints[:, :, 0])
    stats("delta_y", target_waypoints[:, :, 1])
    stats("delta_yaw", target_waypoints[:, :, 2])
    final_distance = np.linalg.norm(target_waypoints[:, -1, :2], axis=1)
    stats("final waypoint distance", final_distance)
    stats("velocity v", velocity[:, 0])
    stats("velocity omega", velocity[:, 1])

    print("\nWarnings:")
    warnings = 0
    negative_forward = np.mean(target_waypoints[:, -1, 0] < 0.0)
    if negative_forward > 0.25:
        warnings += 1
        print(f"  {negative_forward:.1%} of samples have negative final delta_x.")
    if float(np.max(final_distance)) > args.large_final_distance:
        warnings += 1
        print(f"  max final waypoint distance is large: {np.max(final_distance):.3f} m.")
    if float(np.max(np.abs(target_waypoints[:, :, 2]))) > 3.14:
        warnings += 1
        print("  yaw changes exceed pi radians.")
    if t < args.min_samples:
        warnings += 1
        print(f"  only {t} samples were produced.")
    if warnings == 0:
        print("  none")

    print("\nOK")


if __name__ == "__main__":
    main()
