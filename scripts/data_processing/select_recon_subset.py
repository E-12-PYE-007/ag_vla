from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from flow_head.recon_loader import hdf5_keys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select a clean, useful subset of RECON HDF5 trajectories.")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--target-hours", type=float, default=5.0)
    parser.add_argument("--min-duration-sec", type=float, default=20.0)
    parser.add_argument("--max-stationary-fraction", type=float, default=0.4)
    parser.add_argument("--min-final-displacement", type=float, default=2.0)
    parser.add_argument("--max-pose-jump", type=float, default=2.0)
    parser.add_argument("--max-yaw-jump", type=float, default=1.0)
    parser.add_argument("--fps", type=float, default=5.0, help="Used only when --duration-mode=fps.")
    parser.add_argument(
        "--duration-mode",
        choices=["path_length", "fps"],
        default="path_length",
        help="Estimate RECON duration from path length by default because raw RECON HDF5 files do not include timestamps.",
    )
    parser.add_argument("--nominal-speed-mps", type=float, default=0.5)
    parser.add_argument(
        "--reject-collision-fields",
        nargs="*",
        default=[],
        help=(
            "Collision fields that reject a whole trajectory. Defaults to none because "
            "convert_recon.py drops physical/stuck/flipped windows at sample level."
        ),
    )
    return parser.parse_args()


def relpath(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def reject(filename: str, reason: str, metrics: dict[str, Any] | None = None) -> dict[str, Any]:
    item: dict[str, Any] = {"filename": filename, "reason": reason}
    if metrics:
        item.update(metrics)
    return item


def infer_environment(path: Path, root: Path) -> str:
    rel = path.relative_to(root)
    if len(rel.parts) > 1:
        return rel.parts[0]
    return path.stem.split("_")[0]


def inspect_recon_file(
    path: Path,
    root: Path,
    fps: float,
    duration_mode: str,
    nominal_speed_mps: float,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    filename = relpath(path, root)
    try:
        import h5py
    except ImportError as exc:
        raise ImportError("RECON subset selection needs h5py: python3 -m pip install h5py") from exc

    try:
        with h5py.File(path, "r") as f:
            datasets = hdf5_keys(f)
            missing = [key for key in ("images/rgb_left", "jackal/position", "jackal/yaw") if key not in datasets]
            if missing:
                return None, reject(filename, f"missing required datasets: {', '.join(missing)}")

            image_shape = datasets["images/rgb_left"].shape
            positions = np.asarray(datasets["jackal/position"], dtype=np.float32)
            yaws = np.asarray(datasets["jackal/yaw"], dtype=np.float32).reshape(-1)
            collision_arrays = {
                key: np.asarray(value, dtype=bool).reshape(-1)
                for key, value in datasets.items()
                if key.startswith("collision/")
            }
    except Exception as exc:
        return None, reject(filename, f"could not read hdf5: {exc}")

    has_valid_images = len(image_shape) >= 1
    has_valid_position = positions.ndim == 2 and positions.shape[1] >= 2
    has_valid_yaw = yaws.ndim == 1
    if not has_valid_images:
        return None, reject(filename, "images/rgb_left is not an image sequence")
    if not has_valid_position:
        return None, reject(filename, f"jackal/position must be [T, >=2], got {positions.shape}")
    if not has_valid_yaw:
        return None, reject(filename, "jackal/yaw is not a 1D sequence")

    positions = positions[:, :2]
    num_frames = min(int(image_shape[0]), int(len(positions)), int(len(yaws)))
    metrics: dict[str, Any] = {
        "filename": filename,
        "trajectory_name": "__".join(path.relative_to(root).with_suffix("").parts),
        "num_frames": int(num_frames),
        "has_valid_images": bool(has_valid_images),
        "has_valid_position": bool(has_valid_position),
        "has_valid_yaw": bool(has_valid_yaw),
    }
    if image_shape[0] != len(positions) or image_shape[0] != len(yaws):
        metrics.update({"image_count": int(image_shape[0]), "position_count": int(len(positions)), "yaw_count": int(len(yaws))})
        return None, reject(filename, "image/position/yaw count mismatch", metrics)

    positions = positions[:num_frames]
    yaws = yaws[:num_frames]
    if not np.isfinite(positions).all() or not np.isfinite(yaws).all():
        return None, reject(filename, "NaN or Inf in position/yaw", metrics)

    collision_counts = {
        key: int(np.count_nonzero(values[:num_frames]))
        for key, values in collision_arrays.items()
        if len(values) >= num_frames
    }
    has_collision = any(count > 0 for count in collision_counts.values())

    if num_frames < 2:
        return None, reject(filename, "too few frames", metrics)

    diffs = np.diff(positions, axis=0)
    step_dist = np.linalg.norm(diffs, axis=1)
    yaw_jump = np.abs(np.diff(np.unwrap(yaws)))
    total_path_length = float(np.sum(step_dist))
    if duration_mode == "fps":
        estimated_duration_sec = float(num_frames / fps) if fps > 0 else 0.0
    else:
        estimated_duration_sec = float(total_path_length / max(nominal_speed_mps, 1e-6))
    final_displacement = float(np.linalg.norm(positions[-1] - positions[0]))
    mean_speed = float(total_path_length / max(estimated_duration_sec, 1e-6))
    stationary_fraction = float(np.mean(step_dist < 0.02))
    max_position_jump = float(np.max(step_dist)) if len(step_dist) else 0.0
    max_yaw_jump_value = float(np.max(yaw_jump)) if len(yaw_jump) else 0.0

    metrics.update(
        {
            "estimated_duration_sec": estimated_duration_sec,
            "total_path_length": total_path_length,
            "path_length": total_path_length,
            "final_displacement": final_displacement,
            "mean_speed": mean_speed,
            "stationary_fraction": stationary_fraction,
            "max_position_jump": max_position_jump,
            "max_yaw_jump": max_yaw_jump_value,
            "environment": infer_environment(path, root),
            "has_collision": bool(has_collision),
            "collision_counts": collision_counts,
        }
    )
    return metrics, None


def rejection_reason(metrics: dict[str, Any], args: argparse.Namespace) -> str | None:
    collision_counts = metrics.get("collision_counts", {})
    rejected_collision_counts = {
        key: int(collision_counts.get(key, 0))
        for key in args.reject_collision_fields
        if int(collision_counts.get(key, 0)) > 0
    }
    if rejected_collision_counts:
        metrics["rejected_collision_counts"] = rejected_collision_counts
        return "rejected collision flag present"
    if metrics["estimated_duration_sec"] < args.min_duration_sec:
        return "too short"
    if metrics["final_displacement"] < args.min_final_displacement:
        return "final displacement too small"
    if metrics["stationary_fraction"] > args.max_stationary_fraction:
        return "too stationary"
    if metrics["max_position_jump"] > args.max_pose_jump:
        return "pose jump too large"
    if metrics["max_yaw_jump"] > args.max_yaw_jump:
        return "yaw jump too large"
    return None


def rank_key(item: dict[str, Any]) -> tuple[float, float, float, float]:
    speed_penalty = abs(float(item["mean_speed"]) - 0.5)
    return (
        float(item["stationary_fraction"]),
        speed_penalty,
        -float(item["total_path_length"]),
        float(item["max_yaw_jump"]),
    )


def select_diverse(accepted: list[dict[str, Any]], target_sec: float) -> list[dict[str, Any]]:
    by_env: dict[str, list[dict[str, Any]]] = {}
    for item in accepted:
        by_env.setdefault(str(item.get("environment", "unknown")), []).append(item)
    for items in by_env.values():
        items.sort(key=rank_key)

    selected = []
    total = 0.0
    while total < target_sec:
        made_progress = False
        for env in sorted(by_env):
            if not by_env[env]:
                continue
            item = by_env[env].pop(0)
            selected.append(item)
            total += float(item["estimated_duration_sec"])
            made_progress = True
            if total >= target_sec:
                break
        if not made_progress:
            break
    return selected


def main() -> None:
    args = parse_args()
    files = sorted([*args.input_root.rglob("*.h5"), *args.input_root.rglob("*.hdf5")])
    if not files:
        raise SystemExit(f"No .h5/.hdf5 files found under {args.input_root}")

    accepted = []
    rejected = []
    for path in files:
        metrics, initial_reject = inspect_recon_file(
            path,
            args.input_root,
            args.fps,
            args.duration_mode,
            args.nominal_speed_mps,
        )
        if initial_reject is not None:
            rejected.append(initial_reject)
            continue
        assert metrics is not None
        reason = rejection_reason(metrics, args)
        if reason:
            rejected.append(reject(metrics["filename"], reason, metrics))
        else:
            accepted.append(metrics)

    accepted.sort(key=rank_key)
    selected = select_diverse(accepted, target_sec=args.target_hours * 3600.0)
    selected_names = {item["filename"] for item in selected}
    not_selected = [item for item in accepted if item["filename"] not in selected_names]

    payload = {
        "target_hours": float(args.target_hours),
        "input_root": str(args.input_root),
        "selection_fps": float(args.fps),
        "duration_mode": args.duration_mode,
        "nominal_speed_mps": float(args.nominal_speed_mps),
        "selection_rules": {
            "min_duration_sec": float(args.min_duration_sec),
            "max_stationary_fraction": float(args.max_stationary_fraction),
            "min_final_displacement": float(args.min_final_displacement),
            "max_pose_jump": float(args.max_pose_jump),
            "max_yaw_jump": float(args.max_yaw_jump),
            "reject_collision_fields": list(args.reject_collision_fields),
        },
        "selected": selected,
        "accepted_not_selected": not_selected,
        "rejected": rejected,
        "total_selected_hours": float(sum(item["estimated_duration_sec"] for item in selected) / 3600.0),
        "num_selected": len(selected),
        "num_rejected": len(rejected),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Selected {len(selected)} trajectories ({payload['total_selected_hours']:.2f} h)")
    print(f"Rejected {len(rejected)} trajectories")
    print(f"Saved {args.out_json}")


if __name__ == "__main__":
    main()
