from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


KEY_ALIASES = {
    "action_embeddings": (
        "projected_actions",
        "projected_action_embeddings",
        "action_embeddings",
        "action_embedding",
        "omni_vla_action_embeddings",
        "embeddings",
    ),
    "raw_action_embeddings": (
        "actions_hidden_states",
        "raw_action_embeddings",
        "raw_action_hidden_states",
        "omni_vla_action_token_hidden_states",
    ),
    "waypoints": ("waypoints", "target_waypoints", "waypoint_chunks", "actions", "trajectory"),
    "robot_state": ("robot_state", "state", "proprio", "proprioception"),
    "image": ("image", "images", "rgb", "camera"),
    "image_paths": ("image_paths", "image_path", "rgb_paths", "rgb_path"),
    "goal_text": ("goal_text", "instruction", "instructions", "goal"),
    "trajectory_id": ("trajectory_id", "traj_id", "episode_id"),
    "timestep": ("timestep", "timesteps", "time_index", "t"),
    "modality_id": ("modality_id", "embedding_modality_id", "taskid", "task_id", "modality"),
    "poses": ("poses", "pose", "global_poses"),
}


@dataclass
class NormalizationStats:
    waypoint_mean: Tensor
    waypoint_std: Tensor
    robot_state_mean: Optional[Tensor] = None
    robot_state_std: Optional[Tensor] = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "waypoint_mean": self.waypoint_mean.cpu().tolist(),
            "waypoint_std": self.waypoint_std.cpu().tolist(),
        }
        if self.robot_state_mean is not None and self.robot_state_std is not None:
            data["robot_state_mean"] = self.robot_state_mean.cpu().tolist()
            data["robot_state_std"] = self.robot_state_std.cpu().tolist()
        return data

    @classmethod
    def from_file(cls, path: str | Path) -> "NormalizationStats":
        with Path(path).open("r", encoding="utf-8") as f:
            data = json.load(f)
        robot_state_mean = data.get("robot_state_mean")
        robot_state_std = data.get("robot_state_std")
        return cls(
            waypoint_mean=torch.tensor(data["waypoint_mean"], dtype=torch.float32),
            waypoint_std=torch.tensor(data["waypoint_std"], dtype=torch.float32),
            robot_state_mean=torch.tensor(robot_state_mean, dtype=torch.float32) if robot_state_mean is not None else None,
            robot_state_std=torch.tensor(robot_state_std, dtype=torch.float32) if robot_state_std is not None else None,
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


def _get_first(mapping: dict[str, Any], logical_key: str) -> Any:
    for key in KEY_ALIASES[logical_key]:
        if key in mapping:
            return mapping[key]
    if logical_key in (
        "raw_action_embeddings",
        "robot_state",
        "image",
        "image_paths",
        "goal_text",
        "trajectory_id",
        "timestep",
        "modality_id",
        "poses",
    ):
        return None
    raise KeyError(f"Could not find key for {logical_key}. Tried: {KEY_ALIASES[logical_key]}")


def _to_tensor(value: Any) -> Tensor:
    if isinstance(value, Tensor):
        return value
    if isinstance(value, np.ndarray):
        if value.dtype.kind in {"U", "S", "O"}:
            raise TypeError("String/object arrays cannot be converted to tensors.")
        return torch.from_numpy(value)
    return torch.as_tensor(value)


def _load_file(path: Path) -> dict[str, Any]:
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=True) as data:
            return {key: data[key].tolist() if data[key].dtype.kind == "O" else data[key] for key in data.files}
    if path.suffix in {".pt", ".pth"}:
        obj = torch.load(path, map_location="cpu")
        if not isinstance(obj, dict):
            raise TypeError(f"Expected {path} to contain a dict, got {type(obj)!r}")
        return obj
    raise ValueError(f"Unsupported data file type: {path.suffix}")


def _list_data_files(data_path: Path) -> list[Path]:
    if data_path.is_file():
        return [data_path]
    direct_files = sorted(path for path in data_path.iterdir() if path.suffix in {".npz", ".pt", ".pth"})
    embedded_files = sorted(data_path.rglob("trajectory_with_embeddings.npz"))
    files = embedded_files or direct_files
    if not files:
        raise FileNotFoundError(f"No .npz/.pt/.pth files found in {data_path}")
    return files


class TrajectoryEmbeddingDataset(Dataset):
    """
    Dataset for precomputed OmniVLA / AsyncVLA action embeddings.

    Supports either one trajectory per file:
        projected_actions/action_embeddings [T, 8, 1024], waypoints [T, H, 3]
        raw actions_hidden_states [T, 32, 4096] or [T, 8, 4, 4096]
        robot_state [T, S]

    Or one sample per file:
        projected_actions/action_embeddings [8, 1024], waypoints [H, 3], robot_state [S]
    """

    def __init__(
        self,
        data_path: str | Path,
        horizon: int = 8,
        waypoint_dim: int = 3,
        normalize_waypoints: bool = False,
        normalize_robot_state: bool = False,
        stats: Optional[NormalizationStats] = None,
        cache_trajectories: bool = False,
    ) -> None:
        self.data_path = Path(data_path)
        self.horizon = horizon
        self.waypoint_dim = waypoint_dim
        self.normalize_waypoints = normalize_waypoints
        self.normalize_robot_state = normalize_robot_state
        self.stats = stats
        self.cache_trajectories = cache_trajectories
        self.files = _list_data_files(self.data_path)
        self._cache: dict[int, dict[str, Any]] = {}
        self.index: list[tuple[int, int]] = []

        for file_idx, file_path in enumerate(self.files):
            data = self._load(file_idx, file_path)
            waypoints = _to_tensor(_get_first(data, "waypoints"))
            if waypoints.ndim == 2:
                self.index.append((file_idx, 0))
            elif waypoints.ndim == 3:
                self.index.extend((file_idx, t) for t in range(int(waypoints.shape[0])))
            else:
                raise ValueError(f"{file_path}: expected waypoints [H, D] or [T, H, D], got {tuple(waypoints.shape)}")

        if not self.index:
            raise ValueError(f"No valid samples found in {self.data_path}")

    def _load(self, file_idx: int, file_path: Optional[Path] = None) -> dict[str, Any]:
        if file_idx in self._cache:
            return self._cache[file_idx]
        if file_path is None:
            file_path = self.files[file_idx]
        data = _load_file(file_path)
        if self.cache_trajectories:
            self._cache[file_idx] = data
        return data

    def __len__(self) -> int:
        return len(self.index)

    def _select_timestep(self, value: Any, timestep: int, trajectory_len: int) -> Any:
        if value is None:
            return None
        if isinstance(value, (str, bytes)):
            return value
        if isinstance(value, np.ndarray) and value.dtype.kind in {"U", "S", "O"}:
            return value[timestep].item() if value.ndim > 0 and len(value) == trajectory_len else value.tolist()
        if isinstance(value, list):
            return value[timestep] if len(value) == trajectory_len else value
        tensor = _to_tensor(value)
        if tensor.ndim > 0 and tensor.shape[0] == trajectory_len:
            return tensor[timestep]
        return tensor

    def __getitem__(self, idx: int) -> dict[str, Any]:
        file_idx, timestep = self.index[idx]
        data = self._load(file_idx)
        file_path = self.files[file_idx]
        waypoints_all = _to_tensor(_get_first(data, "waypoints")).float()
        trajectory_len = int(waypoints_all.shape[0]) if waypoints_all.ndim == 3 else 1

        action_embeddings = None
        try:
            action_embeddings = self._select_timestep(_get_first(data, "action_embeddings"), timestep, trajectory_len)
        except KeyError:
            pass
        raw_action_embeddings = self._select_timestep(_get_first(data, "raw_action_embeddings"), timestep, trajectory_len)
        waypoints = self._select_timestep(waypoints_all, timestep, trajectory_len)
        robot_state = self._select_timestep(_get_first(data, "robot_state"), timestep, trajectory_len)
        modality_id = self._select_timestep(_get_first(data, "modality_id"), timestep, trajectory_len)

        waypoints = _to_tensor(waypoints).float()

        if action_embeddings is None and raw_action_embeddings is None:
            raise KeyError(
                f"{file_path}: expected projected action embeddings or raw action-token hidden states. "
                f"Projected aliases: {KEY_ALIASES['action_embeddings']}; raw aliases: {KEY_ALIASES['raw_action_embeddings']}"
            )
        if action_embeddings is not None:
            action_embeddings = _to_tensor(action_embeddings).float()
            if action_embeddings.ndim != 2:
                raise ValueError(f"{file_path}: expected action_embeddings [N, D], got {tuple(action_embeddings.shape)}")
        if waypoints.shape != (self.horizon, self.waypoint_dim):
            raise ValueError(
                f"{file_path}: expected waypoints [{self.horizon}, {self.waypoint_dim}], got {tuple(waypoints.shape)}"
            )

        if self.normalize_waypoints:
            if self.stats is None:
                raise ValueError("Normalization requested but stats were not provided.")
            waypoints = (waypoints - self.stats.waypoint_mean) / self.stats.waypoint_std

        sample: dict[str, Any] = {
            "waypoints": waypoints,
            "trajectory_id": str(_get_first(data, "trajectory_id") or file_path.stem),
            "timestep": torch.tensor(timestep, dtype=torch.long),
        }

        if action_embeddings is not None:
            sample["action_embeddings"] = action_embeddings

        if raw_action_embeddings is not None:
            raw_action_embeddings = _to_tensor(raw_action_embeddings).float()
            if raw_action_embeddings.ndim not in (2, 3):
                raise ValueError(
                    f"{file_path}: expected raw action tokens [32, 4096] or [8, 4, 4096], "
                    f"got {tuple(raw_action_embeddings.shape)}"
                )
            sample["raw_action_embeddings"] = raw_action_embeddings
            if modality_id is None:
                modality_id = 0.0
            sample["modality_id"] = _to_tensor(modality_id).float().reshape(())

        if robot_state is not None:
            robot_state = _to_tensor(robot_state).float()
            if self.normalize_robot_state:
                if self.stats is None or self.stats.robot_state_mean is None or self.stats.robot_state_std is None:
                    raise ValueError("Robot-state normalization requested but robot-state stats were not provided.")
                robot_state = (robot_state - self.stats.robot_state_mean) / self.stats.robot_state_std
            sample["robot_state"] = robot_state

        image = self._select_timestep(_get_first(data, "image"), timestep, trajectory_len)
        if image is not None:
            sample["image"] = _to_tensor(image).float()

        image_paths = self._select_timestep(_get_first(data, "image_paths"), timestep, trajectory_len)
        if image_paths is not None:
            sample["image_path"] = image_paths

        goal_text = self._select_timestep(_get_first(data, "goal_text"), timestep, trajectory_len)
        if goal_text is not None:
            sample["goal_text"] = goal_text

        poses = self._select_timestep(_get_first(data, "poses"), timestep, trajectory_len)
        if poses is not None:
            sample["pose"] = _to_tensor(poses).float()

        return sample

    def compute_normalization_stats(self) -> NormalizationStats:
        waypoint_sum = None
        waypoint_sq_sum = None
        waypoint_count = 0
        robot_sum = None
        robot_sq_sum = None
        robot_count = 0

        for file_idx, file_path in enumerate(self.files):
            data = self._load(file_idx, file_path)
            waypoints = _to_tensor(_get_first(data, "waypoints")).float()
            if waypoints.ndim == 2:
                waypoints = waypoints[None]
            flat_waypoints = waypoints.reshape(-1, waypoints.shape[-1])
            waypoint_sum = flat_waypoints.sum(dim=0) if waypoint_sum is None else waypoint_sum + flat_waypoints.sum(dim=0)
            waypoint_sq_sum = (
                (flat_waypoints**2).sum(dim=0)
                if waypoint_sq_sum is None
                else waypoint_sq_sum + (flat_waypoints**2).sum(dim=0)
            )
            waypoint_count += flat_waypoints.shape[0]

            robot_state = _get_first(data, "robot_state")
            if robot_state is not None:
                robot_state = _to_tensor(robot_state).float()
                if robot_state.ndim == 1:
                    robot_state = robot_state[None]
                flat_robot = robot_state.reshape(-1, robot_state.shape[-1])
                robot_sum = flat_robot.sum(dim=0) if robot_sum is None else robot_sum + flat_robot.sum(dim=0)
                robot_sq_sum = (
                    (flat_robot**2).sum(dim=0)
                    if robot_sq_sum is None
                    else robot_sq_sum + (flat_robot**2).sum(dim=0)
                )
                robot_count += flat_robot.shape[0]

        waypoint_mean = waypoint_sum / waypoint_count
        waypoint_var = waypoint_sq_sum / waypoint_count - waypoint_mean**2
        waypoint_std = torch.sqrt(torch.clamp(waypoint_var, min=1e-8))

        robot_mean = None
        robot_std = None
        if robot_count > 0 and robot_sum is not None and robot_sq_sum is not None:
            robot_mean = robot_sum / robot_count
            robot_var = robot_sq_sum / robot_count - robot_mean**2
            robot_std = torch.sqrt(torch.clamp(robot_var, min=1e-8))

        return NormalizationStats(
            waypoint_mean=waypoint_mean,
            waypoint_std=waypoint_std,
            robot_state_mean=robot_mean,
            robot_state_std=robot_std,
        )


WaypointChunkDataset = TrajectoryEmbeddingDataset
