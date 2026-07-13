from __future__ import annotations

import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .public_nav_conversion import generate_target_waypoints, save_processed_trajectory, wrap_to_pi, write_json


SEQUENCE_RE = re.compile(r"^dataset_(?P<kind>L|R|refres)_(?P<sequence>.+)\.txt$")


@dataclass(frozen=True)
class GoStanfordSequence:
    sequence_id: str
    left_list: Path | None
    right_list: Path | None
    result_list: Path
    is_flipped: bool
    num_left: int
    num_right: int
    num_results: int


def read_list_file(path: Path) -> list[str]:
    lines: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                lines.append(parts[0])
    return lines


def count_list_file(path: Path | None) -> int:
    if path is None or not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def discover_sequences(root: Path) -> list[GoStanfordSequence]:
    root = Path(root)
    by_sequence: dict[str, dict[str, Path]] = {}
    for txt in root.glob("dataset_*.txt"):
        match = SEQUENCE_RE.match(txt.name)
        if not match:
            continue
        sequence_id = match.group("sequence")
        kind = match.group("kind")
        by_sequence.setdefault(sequence_id, {})[kind] = txt

    sequences: list[GoStanfordSequence] = []
    for sequence_id, paths in sorted(by_sequence.items()):
        result_list = paths.get("refres")
        if result_list is None:
            continue
        left_list = paths.get("L")
        right_list = paths.get("R")
        sequences.append(
            GoStanfordSequence(
                sequence_id=sequence_id,
                left_list=left_list,
                right_list=right_list,
                result_list=result_list,
                is_flipped="F" in sequence_id,
                num_left=count_list_file(left_list),
                num_right=count_list_file(right_list),
                num_results=count_list_file(result_list),
            )
        )
    return sequences


def load_command_pickle(path: Path) -> np.ndarray:
    with path.open("rb") as f:
        value = pickle.load(f, encoding="latin1")
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size < 2:
        raise ValueError(f"Expected at least two values in {path}, got shape {arr.shape}")
    return arr[:2].astype(np.float32)


def load_velocity_commands(root: Path, result_list: Path) -> tuple[list[Path], np.ndarray]:
    rel_paths = read_list_file(result_list)
    pickle_paths = [root / rel for rel in rel_paths]
    commands = []
    for path in pickle_paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing GO Stanford result pickle: {path}")
        commands.append(load_command_pickle(path))
    return pickle_paths, np.stack(commands, axis=0).astype(np.float32)


def integrate_velocity_commands(commands: np.ndarray, frame_dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    commands = np.asarray(commands, dtype=np.float32)
    n = len(commands)
    times = np.arange(n, dtype=np.float64) * float(frame_dt)
    positions = np.zeros((n, 2), dtype=np.float32)
    yaws = np.zeros(n, dtype=np.float32)

    for idx in range(1, n):
        v = float(commands[idx - 1, 0])
        omega = float(commands[idx - 1, 1])
        prev_yaw = float(yaws[idx - 1])
        mid_yaw = prev_yaw + 0.5 * omega * frame_dt
        positions[idx, 0] = positions[idx - 1, 0] + v * frame_dt * np.cos(mid_yaw)
        positions[idx, 1] = positions[idx - 1, 1] + v * frame_dt * np.sin(mid_yaw)
        yaws[idx] = float(wrap_to_pi(prev_yaw + omega * frame_dt))

    return times, positions, yaws


def choose_image_list(sequence: GoStanfordSequence, side: str) -> Path:
    if side == "left":
        if sequence.left_list is None:
            raise ValueError(f"{sequence.sequence_id}: no left image list")
        return sequence.left_list
    if side == "right":
        if sequence.right_list is None:
            raise ValueError(f"{sequence.sequence_id}: no right image list")
        return sequence.right_list
    if sequence.left_list is not None:
        return sequence.left_list
    if sequence.right_list is not None:
        return sequence.right_list
    raise ValueError(f"{sequence.sequence_id}: no image list")


def convert_go_stanford_sequence(
    root: Path,
    sequence: GoStanfordSequence,
    out_dir: Path,
    side: str = "left",
    dataset_name: str = "go_stanford_2",
    frame_dt: float = 0.2,
    horizon: int = 8,
    waypoint_dt: float = 0.5,
    sampling_mode: str = "distance",
    target_spacing_m: float = 0.25,
    distance_tolerance_m: float | None = None,
    frame_stride: int = 1,
    min_final_distance: float = 0.5,
    max_final_distance: float = 4.0,
    max_abs_yaw: float = 3.14,
    image_stride: int = 5,
    copy_images: bool = True,
) -> dict[str, Any]:
    root = Path(root)
    image_list = choose_image_list(sequence, side)
    image_rel_paths = read_list_file(image_list)
    image_sources = [root / rel for rel in image_rel_paths]
    missing_images = [str(path) for path in image_sources if not path.exists()]
    if missing_images:
        raise FileNotFoundError(f"{sequence.sequence_id}: missing {len(missing_images)} images, first={missing_images[0]}")

    _, commands = load_velocity_commands(root, sequence.result_list)
    length = min(len(image_sources), len(commands))
    if length <= horizon * max(1, frame_stride):
        raise ValueError(f"{sequence.sequence_id}: too short after alignment: {length}")

    image_sources = image_sources[:length]
    commands = commands[:length]
    times, positions, yaws = integrate_velocity_commands(commands, frame_dt=frame_dt)
    velocity = commands.astype(np.float32)

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

    stride = max(1, int(image_stride))
    if stride > 1:
        keep = (valid_indices % stride) == 0
        valid_indices = valid_indices[keep]
        target_waypoints = target_waypoints[keep]
    if len(valid_indices) == 0:
        raise ValueError(f"{sequence.sequence_id}: no valid waypoint chunks remained after image_stride={stride}")

    selected_side = "left" if image_list == sequence.left_list else "right"
    trajectory_name = f"{sequence.sequence_id}_{selected_side}"
    path_length = float(np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))) if len(positions) > 1 else 0.0
    metadata = save_processed_trajectory(
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
        copy_images=copy_images,
        metadata_extra={
            "source_root": str(root),
            "sequence_id": sequence.sequence_id,
            "image_list": str(image_list),
            "result_list": str(sequence.result_list),
            "side": selected_side,
            "is_flipped": bool(sequence.is_flipped),
            "frame_dt": float(frame_dt),
            "horizon": int(horizon),
            "sampling_mode": sampling_mode,
            "waypoint_dt": float(waypoint_dt) if sampling_mode == "time" else None,
            "target_spacing_m": float(target_spacing_m) if sampling_mode == "distance" else None,
            "distance_tolerance_m": None if distance_tolerance_m is None else float(distance_tolerance_m),
            "frame_stride": int(frame_stride),
            "image_stride": int(image_stride),
            "min_final_distance": float(min_final_distance),
            "max_final_distance": float(max_final_distance),
            "max_abs_yaw": float(max_abs_yaw),
            "num_raw_frames": int(length),
            "path_length_m": path_length,
            "velocity_command_mean": commands.mean(axis=0).astype(float).tolist(),
            "velocity_command_std": commands.std(axis=0).astype(float).tolist(),
            "velocity_command_min": commands.min(axis=0).astype(float).tolist(),
            "velocity_command_max": commands.max(axis=0).astype(float).tolist(),
            "warning": (
                "GO Stanford 2 poses are integrated from result pickle velocity commands. "
                "frame_dt and command semantics should be treated as assumptions."
            ),
        },
    )
    return metadata


def write_sequence_summary(path: Path, sequences: list[GoStanfordSequence]) -> None:
    write_json(
        path,
        {
            "num_sequences": len(sequences),
            "num_flipped": sum(1 for seq in sequences if seq.is_flipped),
            "num_non_flipped": sum(1 for seq in sequences if not seq.is_flipped),
            "sequences": [
                {
                    "sequence_id": seq.sequence_id,
                    "left_list": None if seq.left_list is None else str(seq.left_list),
                    "right_list": None if seq.right_list is None else str(seq.right_list),
                    "result_list": str(seq.result_list),
                    "is_flipped": seq.is_flipped,
                    "num_left": seq.num_left,
                    "num_right": seq.num_right,
                    "num_results": seq.num_results,
                }
                for seq in sequences
            ],
        },
    )
