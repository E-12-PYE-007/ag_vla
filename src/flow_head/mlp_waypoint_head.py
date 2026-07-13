from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn

from flow_head.flow_waypoint_head import clamp_waypoints


@dataclass
class MLPWaypointHeadConfig:
    context_dim: int = 1024
    num_context_tokens: int = 8
    horizon: int = 8
    waypoint_dim: int = 3
    hidden_dim: int = 4096
    num_blocks: int = 2
    dropout: float = 0.0
    robot_state_dim: int = 0
    use_modality_id: bool = True
    max_forward_distance: float = 3.0
    max_lateral_distance: float = 1.5
    max_yaw_change: float = 1.57


class MLPResNetBlock(nn.Module):
    """Residual MLP block matching AsyncVLA's deterministic action-head style."""

    def __init__(self, dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.ffn = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.ffn(x) + x


class MLPWaypointHead(nn.Module):
    """
    Deterministic residual-MLP waypoint head.

    This mirrors AsyncVLA's L1RegressionActionHead_idcat idea, but consumes the
    precomputed projected VLA action embeddings used by our flow head:

        action_embeddings [B, 8, 1024] + optional robot_state + modality_id
            -> target_waypoints [B, 8, 3]
    """

    def __init__(self, config: MLPWaypointHeadConfig | None = None, **kwargs) -> None:
        super().__init__()
        if config is None:
            config = MLPWaypointHeadConfig(**kwargs)
        self.config = config

        input_dim = config.context_dim * config.num_context_tokens
        input_dim += config.robot_state_dim
        if config.use_modality_id:
            input_dim += 1
        output_dim = config.horizon * config.waypoint_dim

        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, config.hidden_dim)
        self.relu = nn.ReLU()
        self.blocks = nn.ModuleList(
            [MLPResNetBlock(config.hidden_dim, dropout=config.dropout) for _ in range(config.num_blocks)]
        )
        self.layer_norm2 = nn.LayerNorm(config.hidden_dim)
        self.fc2 = nn.Linear(config.hidden_dim, output_dim)

    def _features(
        self,
        context: Tensor,
        robot_state: Optional[Tensor] = None,
        modality_id: Optional[Tensor] = None,
    ) -> Tensor:
        cfg = self.config
        if context.ndim != 3:
            raise ValueError(f"Expected context [B, N, D], got {tuple(context.shape)}")
        if context.shape[1:] != (cfg.num_context_tokens, cfg.context_dim):
            raise ValueError(
                f"Expected context [B, {cfg.num_context_tokens}, {cfg.context_dim}], got {tuple(context.shape)}"
            )
        features = [context.float().reshape(context.shape[0], -1)]

        if cfg.robot_state_dim > 0:
            if robot_state is None:
                raise ValueError("robot_state is required because robot_state_dim > 0.")
            if robot_state.shape[-1] != cfg.robot_state_dim:
                raise ValueError(f"Expected robot_state_dim={cfg.robot_state_dim}, got {robot_state.shape[-1]}")
            features.append(robot_state.float())

        if cfg.use_modality_id:
            if modality_id is None:
                modality_id = torch.zeros(context.shape[0], device=context.device, dtype=context.dtype)
            if modality_id.ndim == 0:
                modality_id = modality_id.expand(context.shape[0])
            features.append(modality_id.float().reshape(context.shape[0], 1))

        return torch.cat(features, dim=-1)

    def forward(
        self,
        context: Tensor,
        robot_state: Optional[Tensor] = None,
        modality_id: Optional[Tensor] = None,
    ) -> Tensor:
        x = self._features(context=context, robot_state=robot_state, modality_id=modality_id)
        x = self.relu(self.fc1(self.layer_norm1(x)))
        for block in self.blocks:
            x = block(x)
        x = self.fc2(self.layer_norm2(x))
        return x.reshape(context.shape[0], self.config.horizon, self.config.waypoint_dim)

    def loss(
        self,
        target_waypoints: Tensor,
        context: Tensor,
        robot_state: Optional[Tensor] = None,
        modality_id: Optional[Tensor] = None,
        loss_type: str = "l1",
    ) -> tuple[Tensor, dict[str, Tensor]]:
        pred = self(context=context, robot_state=robot_state, modality_id=modality_id)
        if loss_type == "l1":
            loss = torch.nn.functional.l1_loss(pred, target_waypoints)
        elif loss_type == "mse":
            loss = torch.nn.functional.mse_loss(pred, target_waypoints)
        elif loss_type == "smooth_l1":
            loss = torch.nn.functional.smooth_l1_loss(pred, target_waypoints)
        else:
            raise ValueError(f"Unknown loss_type {loss_type!r}")
        metrics = {
            "loss": loss.detach(),
            "l1": torch.nn.functional.l1_loss(pred.detach(), target_waypoints),
            "rmse": torch.sqrt(torch.mean((pred.detach() - target_waypoints) ** 2)),
        }
        return loss, metrics

    @torch.no_grad()
    def predict(
        self,
        context: Tensor,
        robot_state: Optional[Tensor] = None,
        modality_id: Optional[Tensor] = None,
        clamp: bool = False,
    ) -> Tensor:
        pred = self(context=context, robot_state=robot_state, modality_id=modality_id)
        if clamp:
            pred = clamp_waypoints(
                pred,
                max_forward_distance=self.config.max_forward_distance,
                max_lateral_distance=self.config.max_lateral_distance,
                max_yaw_change=self.config.max_yaw_change,
            )
        return pred
