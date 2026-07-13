from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check processed SCAND trajectory.npz output.")
    parser.add_argument("--npz", type=Path, required=True)
    return parser.parse_args()


def stats(name: str, values: np.ndarray) -> None:
    print(
        f"{name}: mean={np.mean(values):.4f} std={np.std(values):.4f} "
        f"min={np.min(values):.4f} max={np.max(values):.4f}"
    )


def main() -> None:
    args = parse_args()
    data = np.load(args.npz, allow_pickle=False)
    print(f"Loaded {args.npz}")
    for key in data.files:
        print(f"{key}: {data[key].shape} {data[key].dtype}")

    image_paths = data["image_paths"]
    position = data["position"]
    yaw = data["yaw"]
    velocity = data["velocity"]
    target_waypoints = data["target_waypoints"]
    t = target_waypoints.shape[0]

    checks = {
        "target_waypoints [T, 8, 3]": target_waypoints.ndim == 3 and target_waypoints.shape[1:] == (8, 3),
        "position [T, 2]": position.shape == (t, 2),
        "yaw [T]": yaw.shape == (t,),
        "velocity [T, 2]": velocity.shape == (t, 2),
        "image_paths length T": len(image_paths) == t,
    }
    print("\nShape checks:")
    for name, ok in checks.items():
        print(f"  {name}: {'OK' if ok else 'FAIL'}")

    print("\nFirst waypoint chunk:")
    print(target_waypoints[0])

    print("\nFinite checks:")
    for key in ("position", "yaw", "velocity", "target_waypoints"):
        arr = data[key]
        print(f"  {key}: {'OK' if np.isfinite(arr).all() else 'FAIL'}")

    print("\nWaypoint statistics:")
    stats("delta_x", target_waypoints[..., 0])
    stats("delta_y", target_waypoints[..., 1])
    stats("delta_yaw", target_waypoints[..., 2])

    negative_final_dx = np.mean(target_waypoints[:, -1, 0] < 0.0)
    print(f"\nNegative final delta_x fraction: {negative_final_dx:.3f}")
    if negative_final_dx > 0.25:
        print("WARNING: many samples end behind the robot; check coordinate frame or reverse-driving segments.")

    if not all(checks.values()):
        raise SystemExit("One or more shape checks failed.")
    if not np.isfinite(target_waypoints).all():
        raise SystemExit("target_waypoints contains NaN/Inf.")


if __name__ == "__main__":
    main()

