from __future__ import annotations

import numpy as np


def wrap_to_pi(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def future_poses_to_local_waypoints(
    poses: np.ndarray,
    horizon: int = 8,
    stride: int = 1,
) -> np.ndarray:
    """
    Convert global [x, y, yaw] trajectory poses into local robot-frame waypoint chunks.

    Returns [T_valid, horizon, 3]. Samples near the end without enough future poses are dropped.
    """
    poses = np.asarray(poses, dtype=np.float32)
    if poses.ndim != 2 or poses.shape[-1] != 3:
        raise ValueError(f"Expected poses [T, 3], got {poses.shape}")

    max_offset = horizon * stride
    chunks = []
    for t in range(0, len(poses) - max_offset):
        x_t, y_t, yaw_t = poses[t]
        future = poses[t + stride : t + max_offset + 1 : stride]
        dx_world = future[:, 0] - x_t
        dy_world = future[:, 1] - y_t
        cos_yaw = np.cos(yaw_t)
        sin_yaw = np.sin(yaw_t)
        delta_x = cos_yaw * dx_world + sin_yaw * dy_world
        delta_y = -sin_yaw * dx_world + cos_yaw * dy_world
        delta_yaw = wrap_to_pi(future[:, 2] - yaw_t)
        chunks.append(np.stack([delta_x, delta_y, delta_yaw], axis=-1))
    return np.asarray(chunks, dtype=np.float32)

