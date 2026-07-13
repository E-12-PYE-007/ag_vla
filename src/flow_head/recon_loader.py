from __future__ import annotations

from pathlib import Path
from typing import Any
from io import BytesIO

import numpy as np
from PIL import Image


def hdf5_keys(group: Any, prefix: str = "") -> dict[str, Any]:
    keys = {}
    for key, value in group.items():
        full_key = f"{prefix}/{key}" if prefix else key
        if hasattr(value, "shape"):
            keys[full_key] = value
        elif hasattr(value, "items"):
            keys.update(hdf5_keys(value, full_key))
    return keys


def get_required_dataset(datasets: dict[str, Any], key: str, path: Path) -> Any:
    if key in datasets:
        return datasets[key]
    available = ", ".join(sorted(datasets.keys())[:120])
    raise ValueError(f"{path} is missing required RECON dataset '{key}'. Available keys: {available}")


def load_recon_hdf5(
    path: Path,
    image_out_dir: Path,
) -> tuple[list[Path], np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None, dict[str, np.ndarray]]:
    try:
        import h5py
    except ImportError as exc:
        raise ImportError("RECON HDF5 conversion needs h5py: python3 -m pip install h5py") from exc

    with h5py.File(path, "r") as f:
        datasets = hdf5_keys(f)
        image_ds = get_required_dataset(datasets, "images/rgb_left", path)
        position_ds = get_required_dataset(datasets, "jackal/position", path)
        yaw_ds = get_required_dataset(datasets, "jackal/yaw", path)
        time_ds = datasets.get("jackal/time") or datasets.get("time") or datasets.get("timestamps")
        velocity_ds = datasets.get("jackal/velocity") or datasets.get("jackal/cmd_vel")
        linear_velocity_ds = datasets.get("jackal/linear_velocity")
        angular_velocity_ds = datasets.get("jackal/angular_velocity")
        collision_masks = {
            key: np.asarray(value, dtype=bool).reshape(-1)
            for key, value in datasets.items()
            if key.startswith("collision/")
        }

        images_arr = np.asarray(image_ds)
        positions = np.asarray(position_ds, dtype=np.float32)
        if positions.ndim != 2 or positions.shape[1] < 2:
            raise ValueError(f"{path}: jackal/position must be [T, >=2], got {positions.shape}")
        positions = positions[:, :2]
        yaws = np.asarray(yaw_ds, dtype=np.float32).reshape(-1)
        times = None if time_ds is None else np.asarray(time_ds, dtype=np.float64).reshape(-1)
        velocity = None if velocity_ds is None else np.asarray(velocity_ds, dtype=np.float32)
        if velocity is not None and velocity.ndim == 2 and velocity.shape[1] > 2:
            velocity = velocity[:, :2]
        if velocity is None and linear_velocity_ds is not None and angular_velocity_ds is not None:
            linear_velocity = np.asarray(linear_velocity_ds, dtype=np.float32).reshape(-1)
            angular_velocity = np.asarray(angular_velocity_ds, dtype=np.float32).reshape(-1)
            velocity = np.stack([linear_velocity, angular_velocity], axis=1)

    image_out_dir.mkdir(parents=True, exist_ok=True)
    image_paths = []
    for idx, image in enumerate(images_arr):
        arr = np.asarray(image)
        if arr.dtype.kind in {"S", "O"} or isinstance(image, (bytes, bytearray, np.bytes_)):
            data = image.tobytes() if isinstance(image, np.bytes_) else bytes(image)
            decoded = Image.open(BytesIO(data)).convert("RGB")
            out = image_out_dir / f"{idx:06d}.jpg"
            decoded.save(out, quality=95)
            image_paths.append(out)
            continue
        if arr.ndim == 3 and arr.shape[0] in {1, 3, 4} and arr.shape[-1] not in {1, 3, 4}:
            arr = np.moveaxis(arr, 0, -1)
        if arr.ndim == 2:
            mode = "L"
        else:
            if arr.shape[-1] == 4:
                arr = arr[..., :3]
            mode = "RGB"
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        out = image_out_dir / f"{idx:06d}.jpg"
        Image.fromarray(arr, mode=mode).convert("RGB").save(out, quality=95)
        image_paths.append(out)

    return image_paths, positions, yaws, times, velocity, collision_masks
