from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from flow_head.public_nav_conversion import (
    estimate_velocity,
    generate_target_waypoints,
    save_processed_trajectory,
)
from flow_head.recon_loader import load_recon_hdf5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert RECON trajectories into mixed flow-head format.")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--selection-json", type=Path)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--dataset-name", default="recon")
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--waypoint-dt", type=float, default=0.5)
    parser.add_argument("--sampling-mode", choices=["distance", "frame", "time"], default="distance")
    parser.add_argument("--target-spacing-m", type=float, default=0.25)
    parser.add_argument("--distance-tolerance-m", type=float, default=None)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--frame-dt", type=float, default=1.0)
    parser.add_argument("--min-final-distance", type=float, default=0.5)
    parser.add_argument("--max-final-distance", type=float, default=4.0)
    parser.add_argument("--max-abs-yaw", type=float, default=3.14)
    parser.add_argument(
        "--drop-collision-fields",
        nargs="*",
        default=["collision/physical", "collision/stuck", "collision/flipped"],
        help="Drop samples whose current/future window is near these collision fields.",
    )
    parser.add_argument("--collision-margin-frames", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-trajectories", type=int, default=None)
    parser.add_argument(
        "--convert-all",
        action="store_true",
        help="Convert every .h5/.hdf5 under input-root. Prefer --selection-json for normal use.",
    )
    return parser.parse_args()


def trajectory_name_from_path(path: Path, root: Path) -> str:
    rel = path.relative_to(root)
    if path.is_file():
        rel = rel.with_suffix("")
    return "__".join(rel.parts)


def load_selected_recon_files(input_root: Path, selection_json: Path | None, convert_all: bool) -> list[Path]:
    if convert_all:
        return sorted([*input_root.rglob("*.h5"), *input_root.rglob("*.hdf5")])
    if selection_json is None:
        raise SystemExit("RECON conversion needs --selection-json, or pass --convert-all explicitly.")
    with selection_json.open("r", encoding="utf-8") as f:
        selection = json.load(f)
    selected = selection.get("selected", [])
    if not selected:
        raise SystemExit(f"No selected RECON trajectories found in {selection_json}")
    paths = []
    for item in selected:
        filename = item.get("filename") or item.get("path")
        if not filename:
            raise ValueError(f"Selection item is missing filename/path: {item}")
        path = Path(filename)
        if not path.is_absolute():
            path = input_root / path
        paths.append(path)
    return paths


def convert_loaded_trajectory(
    image_sources: list[Path],
    positions: np.ndarray,
    yaws: np.ndarray,
    times: np.ndarray | None,
    velocity: np.ndarray | None,
    collision_masks: dict[str, np.ndarray],
    source: Path,
    out_dir: Path,
    dataset_name: str,
    trajectory_name: str,
    horizon: int,
    waypoint_dt: float,
    sampling_mode: str,
    target_spacing_m: float,
    distance_tolerance_m: float | None,
    frame_stride: int,
    frame_dt: float,
    min_final_distance: float,
    max_final_distance: float,
    max_abs_yaw: float,
    drop_collision_fields: list[str],
    collision_margin_frames: int,
    source_format: str,
) -> dict[str, Any]:
    length = min(len(image_sources), len(positions), len(yaws))
    if length <= horizon * frame_stride:
        raise ValueError(f"{source} is too short after alignment: {length} frames")

    image_sources = image_sources[:length]
    positions = np.asarray(positions, dtype=np.float32)[:length]
    yaws = np.asarray(yaws, dtype=np.float32).reshape(-1)[:length]
    if sampling_mode == "time" and times is None:
        raise ValueError(f"{source}: sampling_mode=time needs timestamps, but RECON HDF5 has none")
    use_frame_stride = sampling_mode == "frame"
    if times is None:
        times = np.arange(length, dtype=np.float64) * frame_dt
    else:
        times = np.asarray(times, dtype=np.float64).reshape(-1)[:length]
    if velocity is None:
        velocity = estimate_velocity(times, positions, yaws, frame_dt=frame_dt)
    else:
        velocity = np.asarray(velocity, dtype=np.float32)[:length]

    valid_indices, target_waypoints = generate_target_waypoints(
        times=times,
        positions=positions,
        yaws=yaws,
        horizon=horizon,
        waypoint_dt=waypoint_dt,
        frame_stride=frame_stride,
        sampling_mode=sampling_mode,
        target_spacing_m=target_spacing_m,
        distance_tolerance_m=distance_tolerance_m,
        min_final_distance=min_final_distance,
        max_final_distance=max_final_distance,
        max_abs_yaw=max_abs_yaw,
    )
    pre_collision_filter_samples = int(len(valid_indices))
    collision_mask = make_collision_window_mask(
        length=length,
        collision_masks=collision_masks,
        fields=drop_collision_fields,
        horizon=horizon,
        sampling_mode=sampling_mode,
        positions=positions,
        target_spacing_m=target_spacing_m,
        frame_stride=frame_stride if use_frame_stride else None,
        times=times,
        waypoint_dt=waypoint_dt,
        margin_frames=collision_margin_frames,
    )
    keep = ~collision_mask[valid_indices]
    valid_indices = valid_indices[keep]
    target_waypoints = target_waypoints[keep]
    if len(valid_indices) == 0:
        raise ValueError(
            "No valid waypoint chunks remained after collision-window filtering. "
            f"pre_collision_filter_samples={pre_collision_filter_samples}"
        )
    return save_processed_trajectory(
        out_dir=out_dir,
        dataset_name=dataset_name,
        trajectory_name=trajectory_name,
        image_sources=image_sources,
        times=times,
        positions=positions,
        yaws=yaws,
        velocity=velocity,
        target_waypoints=target_waypoints,
        valid_indices=valid_indices,
        copy_images=True,
        metadata_extra={
            "source": str(source),
            "source_format": source_format,
            "horizon": int(horizon),
            "sampling_mode": sampling_mode,
            "waypoint_dt": float(waypoint_dt) if sampling_mode == "time" else None,
            "target_spacing_m": float(target_spacing_m) if sampling_mode == "distance" else None,
            "distance_tolerance_m": None if distance_tolerance_m is None else float(distance_tolerance_m),
            "frame_stride": int(frame_stride),
            "frame_dt": float(frame_dt),
            "min_final_distance": float(min_final_distance),
            "max_final_distance": float(max_final_distance),
            "max_abs_yaw": float(max_abs_yaw),
            "image_mode": "extracted_from_hdf5",
            "pre_collision_filter_samples": pre_collision_filter_samples,
            "num_collision_filtered_samples": int(pre_collision_filter_samples - len(valid_indices)),
            "drop_collision_fields": list(drop_collision_fields),
            "collision_margin_frames": int(collision_margin_frames),
        },
    )


def make_collision_window_mask(
    length: int,
    collision_masks: dict[str, np.ndarray],
    fields: list[str],
    horizon: int,
    sampling_mode: str,
    positions: np.ndarray,
    target_spacing_m: float,
    frame_stride: int | None,
    times: np.ndarray,
    waypoint_dt: float,
    margin_frames: int,
) -> np.ndarray:
    collision = np.zeros(length, dtype=bool)
    for field in fields:
        values = collision_masks.get(field)
        if values is None:
            continue
        usable = np.asarray(values, dtype=bool).reshape(-1)[:length]
        collision[: len(usable)] |= usable
    if not collision.any():
        return np.zeros(length, dtype=bool)

    collision_indices = np.flatnonzero(collision)
    invalid = np.zeros(length, dtype=bool)
    if sampling_mode == "distance":
        positions = np.asarray(positions, dtype=np.float32)[:length]
        target_distance = horizon * target_spacing_m
        for idx in range(length):
            future_positions = positions[idx:]
            if len(future_positions) < 2:
                continue
            step_distances = np.linalg.norm(np.diff(future_positions, axis=0), axis=1)
            cumulative = np.concatenate([[0.0], np.cumsum(step_distances)])
            if target_distance > float(cumulative[-1]):
                continue
            end_offset = int(np.searchsorted(cumulative, target_distance, side="left"))
            start_idx = max(0, idx - margin_frames)
            end_idx = min(length - 1, idx + end_offset + margin_frames)
            if collision[start_idx : end_idx + 1].any():
                invalid[idx] = True
    elif frame_stride is not None:
        window_span = horizon * frame_stride
        for cidx in collision_indices:
            start = max(0, int(cidx) - window_span - margin_frames)
            end = min(length, int(cidx) + margin_frames + 1)
            invalid[start:end] = True
    else:
        for idx in range(length):
            start_time = float(times[idx])
            end_time = start_time + horizon * waypoint_dt
            start_idx = max(0, idx - margin_frames)
            end_idx = min(length - 1, int(np.searchsorted(times, end_time, side="right")) + margin_frames)
            if collision[start_idx : end_idx + 1].any():
                invalid[idx] = True
    return invalid


def main() -> None:
    args = parse_args()
    candidates = load_selected_recon_files(args.input_root, args.selection_json, args.convert_all)
    if args.max_trajectories is not None:
        candidates = candidates[: args.max_trajectories]
    if not candidates:
        raise SystemExit(f"No RECON .h5/.hdf5 trajectories found under {args.input_root}")

    converted = 0
    total = 0
    failed = []
    for source in candidates:
        if not source.exists():
            failed.append({"source": str(source), "error": "file does not exist"})
            print(f"FAILED {source}: file does not exist")
            continue
        traj_name = trajectory_name_from_path(source, args.input_root)
        out_dir = args.out_root / traj_name
        if (out_dir / "trajectory.npz").exists() and not args.overwrite:
            print(f"Skipping already processed trajectory: {out_dir}")
            continue
        try:
            staging_dir = out_dir / "_extracted_images"
            images, positions, yaws, times, velocity, collision_masks = load_recon_hdf5(source, staging_dir)
            metadata = convert_loaded_trajectory(
                image_sources=images,
                positions=positions,
                yaws=yaws,
                times=times,
                velocity=velocity,
                collision_masks=collision_masks,
                source=source,
                out_dir=out_dir,
                dataset_name=args.dataset_name,
                trajectory_name=traj_name,
                horizon=args.horizon,
                waypoint_dt=args.waypoint_dt,
                sampling_mode=args.sampling_mode,
                target_spacing_m=args.target_spacing_m,
                distance_tolerance_m=args.distance_tolerance_m,
                frame_stride=args.frame_stride,
                frame_dt=args.frame_dt,
                min_final_distance=args.min_final_distance,
                max_final_distance=args.max_final_distance,
                max_abs_yaw=args.max_abs_yaw,
                drop_collision_fields=args.drop_collision_fields,
                collision_margin_frames=args.collision_margin_frames,
                source_format="recon_hdf5",
            )
            converted += 1
            total += int(metadata["num_saved_samples"])
            print(f"Converted {source} -> {out_dir} ({metadata['num_saved_samples']} samples)")
        except Exception as exc:
            failed.append({"source": str(source), "error": str(exc)})
            print(f"FAILED {source}: {exc}")

    print(f"Done. Converted {converted} trajectories, total samples={total}, failed={len(failed)}")
    if failed:
        for item in failed[:20]:
            print(f"  {item['source']}: {item['error']}")


if __name__ == "__main__":
    main()
