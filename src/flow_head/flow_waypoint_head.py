from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn


@dataclass
class FlowWaypointHeadConfig:
    context_dim: int
    num_context_tokens: int | None = None
    horizon: int = 8
    waypoint_dim: int = 3
    model_dim: int = 256
    num_layers: int = 4
    num_heads: int = 4
    dropout: float = 0.1
    robot_state_dim: int = 0
    max_forward_distance: float = 3.0
    max_lateral_distance: float = 1.5
    max_yaw_change: float = 1.57


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: Tensor) -> Tensor:
        if t.ndim == 0:
            t = t[None]
        if t.ndim == 2 and t.shape[-1] == 1:
            t = t.squeeze(-1)

        half_dim = self.dim // 2
        freqs = torch.exp(torch.linspace(0, -math.log(10000.0), half_dim, device=t.device))
        args = t.float()[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = torch.nn.functional.pad(emb, (0, self.dim - emb.shape[-1]))
        return self.mlp(emb)


class FlowWaypointHead(nn.Module):
    """
    Flow-matching waypoint action head for frozen OmniVLA / AsyncVLA action tokens.

    Inputs:
        x_t: noisy/interpolated waypoint chunk [B, H, waypoint_dim]
        t: flow time in [0, 1] [B]
        context: OmniVLA action-token hidden states [B, N, D]
        robot_state: optional proprio/state vector [B, S]

    Output:
        pred_flow: vector field over waypoint chunks [B, H, waypoint_dim]
    """

    def __init__(self, config: FlowWaypointHeadConfig | None = None, **kwargs) -> None:
        super().__init__()
        if config is None:
            config = FlowWaypointHeadConfig(**kwargs)
        if config.model_dim % config.num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads.")
        self.config = config

        self.waypoint_proj = nn.Linear(config.waypoint_dim, config.model_dim)
        self.context_proj = nn.Linear(config.context_dim, config.model_dim)
        self.time_embed = SinusoidalTimeEmbedding(config.model_dim)
        self.waypoint_pos = nn.Parameter(torch.randn(1, config.horizon, config.model_dim) * 0.02)

        self.robot_state_proj = (
            nn.Sequential(
                nn.Linear(config.robot_state_dim, config.model_dim),
                nn.SiLU(),
                nn.Linear(config.model_dim, config.model_dim),
            )
            if config.robot_state_dim > 0
            else None
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.model_dim,
            nhead=config.num_heads,
            dim_feedforward=config.model_dim * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.num_layers)
        self.norm = nn.LayerNorm(config.model_dim)
        self.output_proj = nn.Linear(config.model_dim, config.waypoint_dim)

    def _validate_inputs(self, x_t: Tensor, t: Tensor, context: Tensor, robot_state: Optional[Tensor]) -> Tensor:
        if x_t.ndim != 3:
            raise ValueError(f"Expected x_t [B, H, waypoint_dim], got {tuple(x_t.shape)}")
        if x_t.shape[1:] != (self.config.horizon, self.config.waypoint_dim):
            raise ValueError(
                f"Expected x_t [B, {self.config.horizon}, {self.config.waypoint_dim}], "
                f"got {tuple(x_t.shape)}"
            )
        if context.ndim == 2:
            context = context[:, None, :]
        if context.ndim != 3:
            raise ValueError(f"Expected context [B, N, D], got {tuple(context.shape)}")
        if context.shape[-1] != self.config.context_dim:
            raise ValueError(f"Expected context_dim={self.config.context_dim}, got {context.shape[-1]}")
        if self.config.num_context_tokens is not None and context.shape[1] != self.config.num_context_tokens:
            raise ValueError(f"Expected {self.config.num_context_tokens} context tokens, got {context.shape[1]}")
        if context.shape[0] != x_t.shape[0]:
            raise ValueError("x_t and context batch sizes must match.")
        if t.ndim == 0:
            t = t.expand(x_t.shape[0])
        if t.shape[0] != x_t.shape[0]:
            raise ValueError("t and x_t batch sizes must match.")
        if self.robot_state_proj is not None:
            if robot_state is None:
                raise ValueError("robot_state is required because robot_state_dim > 0.")
            if robot_state.shape[-1] != self.config.robot_state_dim:
                raise ValueError(f"Expected robot_state_dim={self.config.robot_state_dim}, got {robot_state.shape[-1]}")
        return context

    def forward(
        self,
        x_t: Tensor,
        t: Tensor,
        context: Tensor,
        robot_state: Optional[Tensor] = None,
    ) -> Tensor:
        context = self._validate_inputs(x_t, t, context, robot_state)
        time_token = self.time_embed(t).to(dtype=x_t.dtype)

        waypoint_tokens = self.waypoint_proj(x_t.float())
        waypoint_tokens = waypoint_tokens + self.waypoint_pos[:, : x_t.shape[1]] + time_token[:, None, :]

        memory = self.context_proj(context.float())
        if self.robot_state_proj is not None and robot_state is not None:
            robot_token = self.robot_state_proj(robot_state.float())[:, None, :]
            memory = torch.cat([memory, robot_token], dim=1)

        decoded = self.decoder(tgt=waypoint_tokens, memory=memory)
        return self.output_proj(self.norm(decoded))

    def flow_matching_loss(
        self,
        x1: Tensor,
        context: Tensor,
        robot_state: Optional[Tensor] = None,
        noise: Optional[Tensor] = None,
        t: Optional[Tensor] = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        batch_size = x1.shape[0]
        if noise is None:
            noise = torch.randn_like(x1)
        if t is None:
            t = torch.rand(batch_size, device=x1.device)

        t_view = t[:, None, None]
        x_t = (1.0 - t_view) * noise + t_view * x1
        target_flow = x1 - noise
        pred_flow = self(x_t=x_t, t=t, context=context, robot_state=robot_state)
        loss = torch.mean((pred_flow - target_flow) ** 2)
        metrics = {
            "loss": loss.detach(),
            "flow_rmse": torch.sqrt(torch.mean((pred_flow.detach() - target_flow) ** 2)),
        }
        return loss, metrics

    @torch.no_grad()
    def sample(
        self,
        context: Tensor,
        robot_state: Optional[Tensor] = None,
        num_steps: int = 20,
        clamp: bool = False,
        generator: Optional[torch.Generator] = None,
    ) -> Tensor:
        if context.ndim == 2:
            batch_size = context.shape[0]
        else:
            batch_size = context.shape[0]
        x = torch.randn(
            batch_size,
            self.config.horizon,
            self.config.waypoint_dim,
            device=context.device,
            generator=generator,
        )
        dt = 1.0 / float(num_steps)
        for step in range(num_steps):
            t = torch.full((batch_size,), step / float(num_steps), device=context.device)
            flow = self(x_t=x, t=t, context=context, robot_state=robot_state)
            x = x + dt * flow
        if clamp:
            x = clamp_waypoints(
                x,
                max_forward_distance=self.config.max_forward_distance,
                max_lateral_distance=self.config.max_lateral_distance,
                max_yaw_change=self.config.max_yaw_change,
            )
        return x


def clamp_waypoints(
    waypoints: Tensor,
    max_forward_distance: float = 3.0,
    max_lateral_distance: float = 1.5,
    max_yaw_change: float = 1.57,
) -> Tensor:
    clamped = waypoints.clone()
    clamped[..., 0] = clamped[..., 0].clamp(0.0, max_forward_distance)
    if clamped.shape[-1] > 1:
        clamped[..., 1] = clamped[..., 1].clamp(-max_lateral_distance, max_lateral_distance)
    if clamped.shape[-1] > 2:
        clamped[..., 2] = clamped[..., 2].clamp(-max_yaw_change, max_yaw_change)
    return clamped
