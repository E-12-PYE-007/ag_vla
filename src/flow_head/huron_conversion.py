from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .scand_conversion import (
    decode_image_msg,
    estimate_missing_velocities,
    load_records_from_bag,
    local_waypoint,
    nearest_index,
    relative_or_absolute_image_path,
    wrap_to_pi,
    write_json,
)


DEFAULT_TARGET_DISTANCES = tuple(0.25 * step for step in range(1, 9))


@dataclass
class DistanceSampleResult:
    waypoints: np.ndarray
    future_times: np.ndarray
    future_positions: np.ndarray
    future_yaw: np.ndarray


def interpolate_angle(a0: float, a1: float, alpha: float) -> float:
    delta = float(wrap_to_pi(a1 - a0))
    return float(wrap_to_pi(a0 + alpha * delta))


def cumulative_path_distance(positions: np.ndarray) -> np.ndarray:
    if len(positions) == 0:
        return np.asarray([], dtype=np.float64)
    segment_lengths = np.linalg.norm(np.diff(positions[:, :2], axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(segment_lengths)]).astype(np.float64)


def velocity_from_pose(times: np.ndarray, positions: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    """Estimate forward speed and yaw-rate from integrated odometry pose."""
    if len(times) < 2:
        return np.zeros((len(times), 2), dtype=np.float32)
    dt = np.gradient(times).astype(np.float64)
    dt = np.where(np.abs(dt) < 1e-6, np.nan, dt)
    dx = np.gradient(positions[:, 0].astype(np.float64))
    dy = np.gradient(positions[:, 1].astype(np.float64))
    world_vx = dx / dt
    world_vy = dy / dt
    heading = yaw.astype(np.float64)
    forward = np.cos(heading) * world_vx + np.sin(heading) * world_vy
    omega = np.gradient(np.unwrap(heading)) / dt
    velocity = np.stack([forward, omega], axis=1)
    return np.nan_to_num(velocity, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def sample_pose_at_distance(
    odom_times: np.ndarray,
    positions: np.ndarray,
    yaw: np.ndarray,
    cumulative_distance: np.ndarray,
    target_distance: float,
) -> tuple[float, np.ndarray, float] | None:
    if target_distance < float(cumulative_distance[0]) or target_distance > float(cumulative_distance[-1]):
        return None

    idx = int(np.searchsorted(cumulative_distance, target_distance, side="left"))
    if idx == 0:
        return float(odom_times[0]), positions[0].astype(np.float32), float(yaw[0])
    if idx >= len(cumulative_distance):
        return float(odom_times[-1]), positions[-1].astype(np.float32), float(yaw[-1])

    d0 = float(cumulative_distance[idx - 1])
    d1 = float(cumulative_distance[idx])
    alpha = 0.0 if math.isclose(d0, d1) else (target_distance - d0) / (d1 - d0)
    alpha = float(np.clip(alpha, 0.0, 1.0))

    time = (1.0 - alpha) * float(odom_times[idx - 1]) + alpha * float(odom_times[idx])
    position = ((1.0 - alpha) * positions[idx - 1] + alpha * positions[idx]).astype(np.float32)
    heading = interpolate_angle(float(yaw[idx - 1]), float(yaw[idx]), alpha)
    return time, position, heading


def build_distance_waypoint_target(
    odom_times: np.ndarray,
    positions: np.ndarray,
    yaw: np.ndarray,
    cumulative_distance: np.ndarray,
    current_idx: int,
    target_distances: tuple[float, ...] = DEFAULT_TARGET_DISTANCES,
    max_pose_jump: float = 1.0,
    max_time_gap: float = 2.0,
    max_abs_yaw: float = math.pi,
) -> DistanceSampleResult | None:
    if current_idx < 0 or current_idx >= len(odom_times):
        return None
    if len(target_distances) == 0:
        return None

    future_end_distance = float(cumulative_distance[current_idx] + target_distances[-1])
    if future_end_distance > float(cumulative_distance[-1]):
        return None

    end_idx = int(np.searchsorted(cumulative_distance, future_end_distance, side="right"))
    if end_idx <= current_idx:
        return None
    if np.max(np.linalg.norm(np.diff(positions[current_idx : end_idx + 1], axis=0), axis=1), initial=0.0) > max_pose_jump:
        return None
    if np.max(np.diff(odom_times[current_idx : end_idx + 1]), initial=0.0) > max_time_gap:
        return None

    current_pos = positions[current_idx]
    current_yaw = float(yaw[current_idx])
    local_waypoints: list[np.ndarray] = []
    future_times: list[float] = []
    future_positions: list[np.ndarray] = []
    future_yaws: list[float] = []

    for distance in target_distances:
        target = sample_pose_at_distance(
            odom_times=odom_times,
            positions=positions,
            yaw=yaw,
            cumulative_distance=cumulative_distance,
            target_distance=float(cumulative_distance[current_idx] + distance),
        )
        if target is None:
            return None
        future_time, future_pos, future_yaw = target
        local = local_waypoint(current_pos, current_yaw, future_pos, future_yaw)
        if not np.isfinite(local).all() or abs(float(local[2])) > max_abs_yaw:
            return None
        local_waypoints.append(local)
        future_times.append(future_time)
        future_positions.append(future_pos)
        future_yaws.append(future_yaw)

    waypoints = np.stack(local_waypoints, axis=0).astype(np.float32)
    return DistanceSampleResult(
        waypoints=waypoints,
        future_times=np.asarray(future_times, dtype=np.float64),
        future_positions=np.stack(future_positions, axis=0).astype(np.float32),
        future_yaw=np.asarray(future_yaws, dtype=np.float32),
    )


def convert_huron_bag_to_training_data(
    bag: Path,
    image_topic: str,
    odom_topic: str,
    out_dir: Path,
    target_distances: tuple[float, ...] = DEFAULT_TARGET_DISTANCES,
    image_stride: int = 5,
    max_pose_time_error: float = 0.20,
    max_pose_jump: float = 1.0,
    max_time_gap: float = 2.0,
    max_abs_yaw: float = math.pi,
    dataset_name: str = "huron",
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    image_records, odom_records, counts = load_records_from_bag(bag, image_topic, odom_topic)
    if not image_records:
        raise ValueError(f"No image messages found on {image_topic}")
    if len(odom_records) < len(target_distances) + 1:
        raise ValueError(f"Not enough odometry messages on {odom_topic}: {len(odom_records)}")

    odom_times = np.asarray([sample.time for sample in odom_records], dtype=np.float64)
    positions = np.stack([sample.position for sample in odom_records], axis=0).astype(np.float32)
    yaw = np.asarray([sample.yaw for sample in odom_records], dtype=np.float32)
    raw_velocity = np.stack([sample.velocity for sample in odom_records], axis=0).astype(np.float32)
    velocity = velocity_from_pose(odom_times, positions, yaw)
    cumulative_distance = cumulative_path_distance(positions)

    saved_image_paths: list[str] = []
    saved_times: list[float] = []
    saved_positions: list[np.ndarray] = []
    saved_yaw: list[float] = []
    saved_velocity: list[np.ndarray] = []
    saved_waypoints: list[np.ndarray] = []
    saved_pose_time_error: list[float] = []
    saved_time_to_final_distance: list[float] = []
    rejected = {
        "pose_sync": 0,
        "future_distance": 0,
        "image_decode": 0,
    }

    candidate_images = image_records[:: max(1, image_stride)]
    for image_record in candidate_images:
        current_idx, sync_error = nearest_index(odom_times, image_record.time)
        if current_idx < 0 or sync_error > max_pose_time_error:
            rejected["pose_sync"] += 1
            continue

        result = build_distance_waypoint_target(
            odom_times=odom_times,
            positions=positions,
            yaw=yaw,
            cumulative_distance=cumulative_distance,
            current_idx=current_idx,
            target_distances=target_distances,
            max_pose_jump=max_pose_jump,
            max_time_gap=max_time_gap,
            max_abs_yaw=max_abs_yaw,
        )
        if result is None:
            rejected["future_distance"] += 1
            continue

        image_idx = len(saved_image_paths)
        image_path = images_dir / f"{image_idx:06d}.jpg"
        try:
            decoded = decode_image_msg(image_record.msg, image_record.msgtype)
            decoded.save(image_path, quality=95)
        except Exception as exc:
            rejected["image_decode"] += 1
            print(f"Skipping image at t={image_record.time:.3f}: {exc}")
            continue

        saved_image_paths.append(relative_or_absolute_image_path(image_path, out_dir))
        saved_times.append(float(image_record.time))
        saved_positions.append(positions[current_idx])
        saved_yaw.append(float(yaw[current_idx]))
        saved_velocity.append(velocity[current_idx])
        saved_waypoints.append(result.waypoints)
        saved_pose_time_error.append(float(sync_error))
        saved_time_to_final_distance.append(float(result.future_times[-1] - odom_times[current_idx]))

    if not saved_waypoints:
        raise ValueError("No valid Huron image/waypoint samples were produced.")

    image_paths_arr = np.asarray(saved_image_paths, dtype=str)
    times_arr = np.asarray(saved_times, dtype=np.float64)
    position_arr = np.stack(saved_positions, axis=0).astype(np.float32)
    yaw_arr = np.asarray(saved_yaw, dtype=np.float32)
    velocity_arr = np.stack(saved_velocity, axis=0).astype(np.float32)
    waypoints_arr = np.stack(saved_waypoints, axis=0).astype(np.float32)

    np.savez_compressed(
        out_dir / "trajectory.npz",
        image_paths=image_paths_arr,
        times=times_arr,
        position=position_arr,
        yaw=yaw_arr,
        velocity=velocity_arr,
        target_waypoints=waypoints_arr,
        dataset_name=np.asarray(dataset_name),
        trajectory_name=np.asarray(bag.stem),
    )

    pose_errors = np.asarray(saved_pose_time_error, dtype=np.float64)
    time_to_final = np.asarray(saved_time_to_final_distance, dtype=np.float64)
    final_distances = np.linalg.norm(waypoints_arr[:, -1, :2], axis=1)
    segment_lengths = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    metadata = {
        "dataset_name": dataset_name,
        "trajectory_name": bag.stem,
        "bag": str(bag),
        "image_topic": image_topic,
        "odom_topic": odom_topic,
        "num_raw_images": counts["num_raw_images"],
        "num_raw_odom": counts["num_raw_odom"],
        "num_candidate_images": int(len(candidate_images)),
        "num_saved_samples": int(len(saved_waypoints)),
        "rejected": rejected,
        "image_stride": int(image_stride),
        "target_distances_m": [float(value) for value in target_distances],
        "max_pose_time_error": float(max_pose_time_error),
        "max_pose_jump": float(max_pose_jump),
        "max_time_gap": float(max_time_gap),
        "max_abs_yaw": float(max_abs_yaw),
        "path_length_m": float(cumulative_distance[-1]),
        "mean_pose_time_error": float(np.mean(pose_errors)),
        "median_pose_time_error": float(np.median(pose_errors)),
        "mean_waypoint_endpoint_distance": float(np.mean(final_distances)),
        "mean_time_to_final_distance": float(np.mean(time_to_final)),
        "median_time_to_final_distance": float(np.median(time_to_final)),
        "mean_robot_speed_path": float(cumulative_distance[-1] / max(odom_times[-1] - odom_times[0], 1e-6)),
        "velocity_source": "derived_from_pose",
        "raw_odom_twist_mean": [float(value) for value in np.mean(raw_velocity, axis=0)],
        "raw_odom_twist_min": [float(value) for value in np.min(raw_velocity, axis=0)],
        "raw_odom_twist_max": [float(value) for value in np.max(raw_velocity, axis=0)],
        "derived_velocity_mean": [float(value) for value in np.mean(velocity, axis=0)],
        "derived_velocity_min": [float(value) for value in np.min(velocity, axis=0)],
        "derived_velocity_max": [float(value) for value in np.max(velocity, axis=0)],
        "max_segment_jump": float(np.max(segment_lengths)) if len(segment_lengths) else 0.0,
        "target_waypoints_shape": list(waypoints_arr.shape),
        "position_shape": list(position_arr.shape),
        "velocity_shape": list(velocity_arr.shape),
    }
    write_json(out_dir / "metadata.json", metadata)
    return metadata
