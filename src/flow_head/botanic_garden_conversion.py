from __future__ import annotations

import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .huron_conversion import (
    DEFAULT_TARGET_DISTANCES,
    build_distance_waypoint_target,
    cumulative_path_distance,
)
from .scand_conversion import (
    ImageRecord,
    OdomSample,
    decode_image_msg,
    estimate_missing_velocities,
    load_records_from_bag,
    nearest_index,
    quaternion_to_yaw,
    relative_or_absolute_image_path,
    write_json,
)


DEFAULT_BOTANIC_IMAGE_TOPIC = "/dalsa_rgb/left/image_raw"
DEFAULT_BOTANIC_POSE_TOPIC = "/gt_poses"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass
class TimestampedImagePath:
    time: float
    path: Path


def load_tum_trajectory(path: Path) -> list[OdomSample]:
    samples: list[OdomSample] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) < 8:
                raise ValueError(f"{path}:{line_number}: expected 'timestamp x y z qx qy qz qw'.")
            timestamp, x, y, _z, qx, qy, qz, qw = map(float, parts[:8])

            class Quaternion:
                pass

            q = Quaternion()
            q.x = qx
            q.y = qy
            q.z = qz
            q.w = qw
            position = np.asarray([x, y], dtype=np.float32)
            samples.append(
                OdomSample(
                    time=float(timestamp),
                    position=position,
                    yaw=float(quaternion_to_yaw(q)),
                    velocity=np.asarray([np.nan, np.nan], dtype=np.float32),
                )
            )
    samples.sort(key=lambda sample: sample.time)
    return samples


def _read_timestamp_file(path: Path) -> list[tuple[float, str | None]]:
    entries: list[tuple[float, str | None]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            try:
                timestamp = float(parts[0])
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: first column must be a timestamp.") from exc
            entries.append((timestamp, parts[1] if len(parts) > 1 else None))
    return entries


def _list_image_files(image_dir: Path) -> list[Path]:
    return sorted(path for path in image_dir.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)


def load_timestamped_images(image_dir: Path, timestamps_file: Path) -> list[TimestampedImagePath]:
    entries = _read_timestamp_file(timestamps_file)
    image_files = _list_image_files(image_dir)
    if not entries:
        raise ValueError(f"No timestamps found in {timestamps_file}")
    if not image_files:
        raise ValueError(f"No image files found in {image_dir}")

    records: list[TimestampedImagePath] = []
    if all(name is not None for _, name in entries):
        for timestamp, name in entries:
            assert name is not None
            path = image_dir / name
            if not path.exists():
                candidates = list(image_dir.rglob(name))
                if not candidates:
                    raise FileNotFoundError(f"Timestamp file references missing image: {name}")
                path = candidates[0]
            records.append(TimestampedImagePath(time=float(timestamp), path=path))
    else:
        if len(entries) != len(image_files):
            raise ValueError(
                f"Timestamp count ({len(entries)}) does not match image count ({len(image_files)}). "
                "Use a timestamp file with 'timestamp relative/image/path' rows if counts differ."
            )
        for (timestamp, _), path in zip(entries, image_files):
            records.append(TimestampedImagePath(time=float(timestamp), path=path))

    records.sort(key=lambda record: record.time)
    return records


def _arrays_from_odom(odom_records: list[OdomSample]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    odom_times = np.asarray([sample.time for sample in odom_records], dtype=np.float64)
    positions = np.stack([sample.position for sample in odom_records], axis=0).astype(np.float32)
    yaw = np.asarray([sample.yaw for sample in odom_records], dtype=np.float32)
    velocity = np.stack([sample.velocity for sample in odom_records], axis=0).astype(np.float32)
    velocity = estimate_missing_velocities(odom_times, positions, yaw, velocity)
    cumulative_distance = cumulative_path_distance(positions)
    return odom_times, positions, yaw, velocity, cumulative_distance


def _append_sample(
    *,
    image_path: str,
    image_time: float,
    current_idx: int,
    sync_error: float,
    result_waypoints: np.ndarray,
    result_final_time: float,
    odom_times: np.ndarray,
    positions: np.ndarray,
    yaw: np.ndarray,
    velocity: np.ndarray,
    saved_image_paths: list[str],
    saved_times: list[float],
    saved_positions: list[np.ndarray],
    saved_yaw: list[float],
    saved_velocity: list[np.ndarray],
    saved_waypoints: list[np.ndarray],
    saved_pose_time_error: list[float],
    saved_time_to_final_distance: list[float],
) -> None:
    saved_image_paths.append(image_path)
    saved_times.append(float(image_time))
    saved_positions.append(positions[current_idx])
    saved_yaw.append(float(yaw[current_idx]))
    saved_velocity.append(velocity[current_idx])
    saved_waypoints.append(result_waypoints)
    saved_pose_time_error.append(float(sync_error))
    saved_time_to_final_distance.append(float(result_final_time - odom_times[current_idx]))


def _write_processed_npz_and_metadata(
    *,
    out_dir: Path,
    sequence_name: str,
    source: str,
    image_topic: str | None,
    pose_source: str,
    num_raw_images: int,
    num_raw_poses: int,
    num_candidate_images: int,
    rejected: dict[str, int],
    image_stride: int,
    target_distances: tuple[float, ...],
    max_pose_time_error: float,
    max_pose_jump: float,
    max_time_gap: float,
    max_abs_yaw: float,
    odom_times: np.ndarray,
    positions_all: np.ndarray,
    cumulative_distance: np.ndarray,
    image_paths: list[str],
    times: list[float],
    positions: list[np.ndarray],
    yaws: list[float],
    velocities: list[np.ndarray],
    waypoints: list[np.ndarray],
    pose_time_errors: list[float],
    time_to_final_distance: list[float],
) -> dict[str, Any]:
    if not waypoints:
        raise ValueError("No valid BotanicGarden image/waypoint samples were produced.")

    image_paths_arr = np.asarray(image_paths, dtype=str)
    times_arr = np.asarray(times, dtype=np.float64)
    position_arr = np.stack(positions, axis=0).astype(np.float32)
    yaw_arr = np.asarray(yaws, dtype=np.float32)
    velocity_arr = np.stack(velocities, axis=0).astype(np.float32)
    waypoints_arr = np.stack(waypoints, axis=0).astype(np.float32)

    np.savez_compressed(
        out_dir / "trajectory.npz",
        image_paths=image_paths_arr,
        times=times_arr,
        position=position_arr,
        yaw=yaw_arr,
        velocity=velocity_arr,
        target_waypoints=waypoints_arr,
        dataset_name=np.asarray("botanic_garden"),
        trajectory_name=np.asarray(sequence_name),
    )

    pose_errors = np.asarray(pose_time_errors, dtype=np.float64)
    time_to_final = np.asarray(time_to_final_distance, dtype=np.float64)
    final_distances = np.linalg.norm(waypoints_arr[:, -1, :2], axis=1)
    segment_lengths = np.linalg.norm(np.diff(positions_all, axis=0), axis=1)
    metadata = {
        "dataset_name": "botanic_garden",
        "trajectory_name": sequence_name,
        "source": source,
        "image_topic": image_topic,
        "pose_source": pose_source,
        "num_raw_images": int(num_raw_images),
        "num_raw_poses": int(num_raw_poses),
        "num_candidate_images": int(num_candidate_images),
        "num_saved_samples": int(len(waypoints)),
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
        "max_segment_jump": float(np.max(segment_lengths)) if len(segment_lengths) else 0.0,
        "target_waypoints_shape": list(waypoints_arr.shape),
        "position_shape": list(position_arr.shape),
        "velocity_shape": list(velocity_arr.shape),
    }
    write_json(out_dir / "metadata.json", metadata)
    return metadata


def convert_botanic_garden_bag_to_training_data(
    bag: Path,
    out_dir: Path,
    image_topic: str = DEFAULT_BOTANIC_IMAGE_TOPIC,
    pose_topic: str = DEFAULT_BOTANIC_POSE_TOPIC,
    target_distances: tuple[float, ...] = DEFAULT_TARGET_DISTANCES,
    image_stride: int = 10,
    max_pose_time_error: float = 0.05,
    max_pose_jump: float = 1.0,
    max_time_gap: float = 1.0,
    max_abs_yaw: float = math.pi,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    image_records, odom_records, counts = load_records_from_bag(bag, image_topic, pose_topic)
    odom_times, positions, yaw, velocity, cumulative_distance = _arrays_from_odom(odom_records)

    saved_image_paths: list[str] = []
    saved_times: list[float] = []
    saved_positions: list[np.ndarray] = []
    saved_yaw: list[float] = []
    saved_velocity: list[np.ndarray] = []
    saved_waypoints: list[np.ndarray] = []
    saved_pose_time_error: list[float] = []
    saved_time_to_final_distance: list[float] = []
    rejected = {"pose_sync": 0, "future_distance": 0, "image_decode": 0}

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

        image_path = images_dir / f"{len(saved_image_paths):06d}.jpg"
        try:
            decoded = decode_image_msg(image_record.msg, image_record.msgtype)
            decoded.save(image_path, quality=95)
        except Exception as exc:
            rejected["image_decode"] += 1
            print(f"Skipping image at t={image_record.time:.3f}: {exc}")
            continue

        _append_sample(
            image_path=relative_or_absolute_image_path(image_path, out_dir),
            image_time=image_record.time,
            current_idx=current_idx,
            sync_error=sync_error,
            result_waypoints=result.waypoints,
            result_final_time=float(result.future_times[-1]),
            odom_times=odom_times,
            positions=positions,
            yaw=yaw,
            velocity=velocity,
            saved_image_paths=saved_image_paths,
            saved_times=saved_times,
            saved_positions=saved_positions,
            saved_yaw=saved_yaw,
            saved_velocity=saved_velocity,
            saved_waypoints=saved_waypoints,
            saved_pose_time_error=saved_pose_time_error,
            saved_time_to_final_distance=saved_time_to_final_distance,
        )

    return _write_processed_npz_and_metadata(
        out_dir=out_dir,
        sequence_name=bag.stem,
        source=str(bag),
        image_topic=image_topic,
        pose_source=pose_topic,
        num_raw_images=counts["num_raw_images"],
        num_raw_poses=counts["num_raw_odom"],
        num_candidate_images=len(candidate_images),
        rejected=rejected,
        image_stride=image_stride,
        target_distances=target_distances,
        max_pose_time_error=max_pose_time_error,
        max_pose_jump=max_pose_jump,
        max_time_gap=max_time_gap,
        max_abs_yaw=max_abs_yaw,
        odom_times=odom_times,
        positions_all=positions,
        cumulative_distance=cumulative_distance,
        image_paths=saved_image_paths,
        times=saved_times,
        positions=saved_positions,
        yaws=saved_yaw,
        velocities=saved_velocity,
        waypoints=saved_waypoints,
        pose_time_errors=saved_pose_time_error,
        time_to_final_distance=saved_time_to_final_distance,
    )


def convert_botanic_garden_files_to_training_data(
    image_dir: Path,
    timestamps_file: Path,
    tum_trajectory: Path,
    out_dir: Path,
    sequence_name: str | None = None,
    target_distances: tuple[float, ...] = DEFAULT_TARGET_DISTANCES,
    image_stride: int = 10,
    max_pose_time_error: float = 0.05,
    max_pose_jump: float = 1.0,
    max_time_gap: float = 1.0,
    max_abs_yaw: float = math.pi,
    copy_images: bool = False,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = out_dir / "images"
    if copy_images:
        images_dir.mkdir(parents=True, exist_ok=True)

    image_records = load_timestamped_images(image_dir, timestamps_file)
    odom_records = load_tum_trajectory(tum_trajectory)
    odom_times, positions, yaw, velocity, cumulative_distance = _arrays_from_odom(odom_records)

    saved_image_paths: list[str] = []
    saved_times: list[float] = []
    saved_positions: list[np.ndarray] = []
    saved_yaw: list[float] = []
    saved_velocity: list[np.ndarray] = []
    saved_waypoints: list[np.ndarray] = []
    saved_pose_time_error: list[float] = []
    saved_time_to_final_distance: list[float] = []
    rejected = {"pose_sync": 0, "future_distance": 0, "image_missing": 0}

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
        if not image_record.path.exists():
            rejected["image_missing"] += 1
            continue
        if copy_images:
            target_path = images_dir / f"{len(saved_image_paths):06d}{image_record.path.suffix.lower()}"
            shutil.copy2(image_record.path, target_path)
            image_path = relative_or_absolute_image_path(target_path, out_dir)
        else:
            image_path = image_record.path.resolve().as_posix()

        _append_sample(
            image_path=image_path,
            image_time=image_record.time,
            current_idx=current_idx,
            sync_error=sync_error,
            result_waypoints=result.waypoints,
            result_final_time=float(result.future_times[-1]),
            odom_times=odom_times,
            positions=positions,
            yaw=yaw,
            velocity=velocity,
            saved_image_paths=saved_image_paths,
            saved_times=saved_times,
            saved_positions=saved_positions,
            saved_yaw=saved_yaw,
            saved_velocity=saved_velocity,
            saved_waypoints=saved_waypoints,
            saved_pose_time_error=saved_pose_time_error,
            saved_time_to_final_distance=saved_time_to_final_distance,
        )

    return _write_processed_npz_and_metadata(
        out_dir=out_dir,
        sequence_name=sequence_name or image_dir.name,
        source=str(image_dir),
        image_topic=None,
        pose_source=str(tum_trajectory),
        num_raw_images=len(image_records),
        num_raw_poses=len(odom_records),
        num_candidate_images=len(candidate_images),
        rejected=rejected,
        image_stride=image_stride,
        target_distances=target_distances,
        max_pose_time_error=max_pose_time_error,
        max_pose_jump=max_pose_jump,
        max_time_gap=max_time_gap,
        max_abs_yaw=max_abs_yaw,
        odom_times=odom_times,
        positions_all=positions,
        cumulative_distance=cumulative_distance,
        image_paths=saved_image_paths,
        times=saved_times,
        positions=saved_positions,
        yaws=saved_yaw,
        velocities=saved_velocity,
        waypoints=saved_waypoints,
        pose_time_errors=saved_pose_time_error,
        time_to_final_distance=saved_time_to_final_distance,
    )
