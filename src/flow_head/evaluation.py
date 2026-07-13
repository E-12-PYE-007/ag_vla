from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor


def waypoint_metrics(pred: Tensor, target: Tensor) -> dict[str, float]:
    pred = pred.detach().float()
    target = target.detach().float()
    xy_error = torch.linalg.norm(pred[..., :2] - target[..., :2], dim=-1)
    yaw_error = torch.abs(pred[..., 2] - target[..., 2]) if pred.shape[-1] > 2 else torch.zeros_like(xy_error)
    return {
        "l1": float(torch.nn.functional.l1_loss(pred, target).cpu()),
        "rmse": float(torch.sqrt(torch.mean((pred - target) ** 2)).cpu()),
        "ade": float(xy_error.mean().cpu()),
        "fde": float(xy_error[:, -1].mean().cpu()),
        "yaw_mae": float(yaw_error.mean().cpu()),
        "final_yaw_mae": float(yaw_error[:, -1].mean().cpu()),
    }


class MetricAverager:
    def __init__(self) -> None:
        self.weighted: dict[str, float] = {}
        self.count = 0

    def update(self, metrics: dict[str, float], n: int) -> None:
        for key, value in metrics.items():
            self.weighted[key] = self.weighted.get(key, 0.0) + float(value) * n
        self.count += n

    def compute(self) -> dict[str, float]:
        return {key: value / max(self.count, 1) for key, value in self.weighted.items()}


def save_prediction_npz(
    path: str | Path,
    predictions: list[np.ndarray],
    targets: list[np.ndarray],
    trajectory_ids: list[str],
    timesteps: list[np.ndarray],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        pred_waypoints=np.concatenate(predictions, axis=0),
        target_waypoints=np.concatenate(targets, axis=0),
        trajectory_id=np.asarray(trajectory_ids),
        timestep=np.concatenate(timesteps, axis=0),
    )


def save_metrics(path: str | Path, metrics: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
