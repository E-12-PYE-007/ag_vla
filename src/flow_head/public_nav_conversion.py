from __future__ import annotations

import json
import math
import os
import pickle
import shutil
from pathlib import Path
from typing import Any, Optional

import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def wrap_to_pi(angle: np.ndarray | float) -> np.ndarray | float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def quaternion_to_yaw_xyzw(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    x, y, z, w = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return np.arctan2(siny_cosp, cosy_cosp).astype(np.float32)


def yaw_from_rotation_matrix(rot: np.ndarray) -> np.ndarray:
    rot = np.asarray(rot)
    return np.arctan2(rot[..., 1, 0], rot[..., 0, 0]).astype(np.float32)


def nearest_index(sorted_times: np.ndarray, query_time: float) -> tuple[int, float]:
    idx = int(np.searchsorted(sorted_times, query_time))
    candidates = []
    if idx < len(sorted_times):
        candidates.append(idx)
    if idx > 0:
        candidates.append(idx - 1)
    if not candidates:
        return -1, float("inf")
    best = min(candidates, key=lambda i: abs(float(sorted_times[i]) - query_time))
    return best, abs(float(sorted_times[best]) - query_time)


def make_local_waypoint_chunk(
    times: np.ndarray,
    positions: np.ndarray,
    yaws: np.ndarray,
    current_index: int,
    horizon: int = 8,
    waypoint_dt: float = 0.5,
    frame_stride: Optional[int] = None,
    time_tolerance: Optional[float] = None,
) -> Optional[np.ndarray]:
    times = np.asarray(times, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float32)
    yaws = np.asarray(yaws, dtype=np.float32)
    current_pos = positions[current_index]
    current_yaw = float(yaws[current_index])
    waypoints = []

    for step in range(1, horizon + 1):
        if frame_stride is not None:
            future_index = current_index + step * frame_stride
            if future_index >= len(positions):
                return None
        else:
            desired_time = float(times[current_index]) + step * waypoint_dt
            if desired_time > float(times[-1]):
                return None
            future_index, time_diff = nearest_index(times, desired_time)
            tolerance = waypoint_dt * 0.5 if time_tolerance is None else time_tolerance
            if future_index < 0 or time_diff > tolerance:
                return None

        future_pos = positions[future_index]
        future_yaw = float(yaws[future_index])
        dx_world = float(future_pos[0] - current_pos[0])
        dy_world = float(future_pos[1] - current_pos[1])
        cos_yaw = math.cos(current_yaw)
        sin_yaw = math.sin(current_yaw)
        delta_x = cos_yaw * dx_world + sin_yaw * dy_world
        delta_y = -sin_yaw * dx_world + cos_yaw * dy_world
        delta_yaw = wrap_to_pi(future_yaw - current_yaw)
        waypoints.append([delta_x, delta_y, delta_yaw])

    return np.asarray(waypoints, dtype=np.float32)


def make_distance_waypoint_chunk(
    positions: np.ndarray,
    yaws: np.ndarray,
    current_index: int,
    horizon: int = 8,
    target_spacing_m: float = 0.25,
    distance_tolerance_m: Optional[float] = None,
) -> Optional[np.ndarray]:
    positions = np.asarray(positions, dtype=np.float32)
    yaws = np.asarray(yaws, dtype=np.float32)
    if current_index >= len(positions) - 1:
        return None

    current_pos = positions[current_index]
    current_yaw = float(yaws[current_index])
    future_positions = positions[current_index:]
    step_distances = np.linalg.norm(np.diff(future_positions, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(step_distances)])
    if distance_tolerance_m is None:
        distance_tolerance_m = target_spacing_m * 0.75

    waypoints = []
    for step in range(1, horizon + 1):
        desired_distance = step * target_spacing_m
        if desired_distance > float(cumulative[-1]):
            return None
        local_future_index = int(np.searchsorted(cumulative, desired_distance, side="left"))
        candidates = [local_future_index]
        if local_future_index > 0:
            candidates.append(local_future_index - 1)
        if local_future_index + 1 < len(cumulative):
            candidates.append(local_future_index + 1)
        local_future_index = min(candidates, key=lambda idx: abs(float(cumulative[idx]) - desired_distance))
        if abs(float(cumulative[local_future_index]) - desired_distance) > distance_tolerance_m:
            return None
        future_index = current_index + local_future_index

        future_pos = positions[future_index]
        future_yaw = float(yaws[future_index])
        dx_world = float(future_pos[0] - current_pos[0])
        dy_world = float(future_pos[1] - current_pos[1])
        cos_yaw = math.cos(current_yaw)
        sin_yaw = math.sin(current_yaw)
        delta_x = cos_yaw * dx_world + sin_yaw * dy_world
        delta_y = -sin_yaw * dx_world + cos_yaw * dy_world
        delta_yaw = wrap_to_pi(future_yaw - current_yaw)
        waypoints.append([delta_x, delta_y, delta_yaw])

    return np.asarray(waypoints, dtype=np.float32)


def estimate_velocity(
    times: Optional[np.ndarray],
    positions: np.ndarray,
    yaws: np.ndarray,
    frame_dt: float = 1.0,
) -> np.ndarray:
    positions = np.asarray(positions, dtype=np.float32)
    yaws = np.asarray(yaws, dtype=np.float32)
    if times is None:
        times = np.arange(len(positions), dtype=np.float64) * frame_dt
    else:
        times = np.asarray(times, dtype=np.float64)
    if len(positions) < 2:
        return np.zeros((len(positions), 2), dtype=np.float32)

    velocity = np.zeros((len(positions), 2), dtype=np.float32)
    unwrapped_yaw = np.unwrap(yaws)
    for i in range(len(positions)):
        if i == 0:
            j0, j1 = 0, 1
        elif i == len(positions) - 1:
            j0, j1 = len(positions) - 2, len(positions) - 1
        else:
            j0, j1 = i - 1, i + 1
        dt = float(times[j1] - times[j0])
        if abs(dt) < 1e-6:
            continue
        distance = float(np.linalg.norm(positions[j1] - positions[j0]))
        omega = float(wrap_to_pi(unwrapped_yaw[j1] - unwrapped_yaw[j0]) / dt)
        velocity[i] = [distance / dt, omega]
    return np.nan_to_num(velocity, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def validate_trajectory_arrays(
    image_paths: list[str] | np.ndarray,
    times: np.ndarray,
    positions: np.ndarray,
    yaws: np.ndarray,
    velocity: np.ndarray,
) -> int:
    length = len(image_paths)
    if len(times) != length or len(positions) != length or len(yaws) != length or len(velocity) != length:
        raise ValueError(
            "Mismatched trajectory lengths: "
            f"images={length}, times={len(times)}, position={len(positions)}, yaw={len(yaws)}, velocity={len(velocity)}"
        )
    if positions.shape != (length, 2):
        raise ValueError(f"Expected position [{length}, 2], got {positions.shape}")
    if yaws.shape != (length,):
        raise ValueError(f"Expected yaw [{length}], got {yaws.shape}")
    if velocity.shape != (length, 2):
        raise ValueError(f"Expected velocity [{length}, 2], got {velocity.shape}")
    return length


def generate_target_waypoints(
    times: np.ndarray,
    positions: np.ndarray,
    yaws: np.ndarray,
    horizon: int = 8,
    waypoint_dt: float = 0.5,
    frame_stride: Optional[int] = None,
    time_tolerance: Optional[float] = None,
    sampling_mode: str = "time",
    target_spacing_m: float = 0.25,
    distance_tolerance_m: Optional[float] = None,
    min_final_distance: float = 0.0,
    max_final_distance: float = 20.0,
    max_abs_yaw: float = 3.14,
) -> tuple[np.ndarray, np.ndarray]:
    if sampling_mode not in {"time", "frame", "distance"}:
        raise ValueError(f"sampling_mode must be one of time, frame, distance; got {sampling_mode!r}")
    chunks = []
    valid_indices = []
    for idx in range(len(positions)):
        if sampling_mode == "distance":
            chunk = make_distance_waypoint_chunk(
                positions=positions,
                yaws=yaws,
                current_index=idx,
                horizon=horizon,
                target_spacing_m=target_spacing_m,
                distance_tolerance_m=distance_tolerance_m,
            )
        else:
            chunk = make_local_waypoint_chunk(
                times=times,
                positions=positions,
                yaws=yaws,
                current_index=idx,
                horizon=horizon,
                waypoint_dt=waypoint_dt,
                frame_stride=frame_stride if sampling_mode == "frame" else None,
                time_tolerance=time_tolerance,
            )
        if chunk is None or not np.isfinite(chunk).all():
            continue
        final_distance = float(np.linalg.norm(chunk[-1, :2]))
        if final_distance < min_final_distance:
            continue
        if final_distance > max_final_distance:
            continue
        if float(np.max(np.abs(chunk[:, 2]))) > max_abs_yaw:
            continue
        valid_indices.append(idx)
        chunks.append(chunk)
    if not chunks:
        raise ValueError("No valid waypoint chunks were generated.")
    return np.asarray(valid_indices, dtype=np.int64), np.stack(chunks, axis=0).astype(np.float32)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def find_image_files(directory: Path) -> list[Path]:
    files = [p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    return sorted(files, key=image_sort_key)


def image_sort_key(path: Path) -> tuple[int, str]:
    try:
        return int(path.stem), path.name
    except ValueError:
        return 10**12, path.name


def link_or_copy_image(src: Path, dst: Path, copy_images: bool = False) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    if copy_images:
        shutil.copy2(src, dst)
    else:
        try:
            os.symlink(src.resolve(), dst)
        except OSError:
            shutil.copy2(src, dst)


def save_processed_trajectory(
    out_dir: Path,
    dataset_name: str,
    trajectory_name: str,
    image_sources: list[Path],
    times: np.ndarray,
    positions: np.ndarray,
    yaws: np.ndarray,
    velocity: np.ndarray,
    target_waypoints: np.ndarray,
    valid_indices: np.ndarray,
    copy_images: bool = False,
    metadata_extra: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    valid_image_paths = []
    for out_idx, src_idx in enumerate(valid_indices):
        src = image_sources[int(src_idx)]
        suffix = src.suffix.lower() if src.suffix else ".jpg"
        dst = images_dir / f"{out_idx:06d}{suffix}"
        link_or_copy_image(src, dst, copy_images=copy_images)
        valid_image_paths.append(dst.relative_to(out_dir).as_posix())

    times = np.asarray(times)[valid_indices]
    positions = np.asarray(positions, dtype=np.float32)[valid_indices]
    yaws = np.asarray(yaws, dtype=np.float32)[valid_indices]
    velocity = np.asarray(velocity, dtype=np.float32)[valid_indices]

    np.savez_compressed(
        out_dir / "trajectory.npz",
        image_paths=np.asarray(valid_image_paths, dtype=str),
        times=np.asarray(times, dtype=np.float64),
        position=positions,
        yaw=yaws,
        velocity=velocity,
        target_waypoints=target_waypoints.astype(np.float32),
        dataset_name=np.asarray(dataset_name),
        trajectory_name=np.asarray(trajectory_name),
    )

    metadata = {
        "dataset_name": dataset_name,
        "trajectory_name": trajectory_name,
        "num_saved_samples": int(len(valid_indices)),
        "target_waypoints_shape": list(target_waypoints.shape),
        "position_shape": list(positions.shape),
        "velocity_shape": list(velocity.shape),
    }
    if metadata_extra:
        metadata.update(metadata_extra)
    write_json(out_dir / "metadata.json", metadata)
    return metadata


def load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def maybe_get(data: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in data:
            return data[key]
    return None
