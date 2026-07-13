from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from flow_head.asyncvla_projector import Proj_Actiontokens
from flow_head.dataset import NormalizationStats, TrajectoryEmbeddingDataset
from flow_head.flow_waypoint_head import FlowWaypointHead, FlowWaypointHeadConfig, clamp_waypoints


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample waypoint chunks from a trained flow head.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", "--data-dir", "--data_dir", dest="data", type=Path, required=True, help="Dataset file or directory to run inference on.")
    parser.add_argument("--output", type=Path, default=Path("predicted_waypoints.npz"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-steps", type=int, default=20)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--no-clamp", action="store_true")
    parser.add_argument("--default-modality-id", type=float, default=0.0)
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def collate_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    batch: dict[str, Any] = {}
    all_keys = set().union(*(sample.keys() for sample in samples))
    for key in sorted(all_keys):
        values = [sample.get(key) for sample in samples]
        if any(value is None for value in values):
            continue
        if all(torch.is_tensor(value) for value in values):
            batch[key] = torch.stack(values)
        else:
            batch[key] = values
    return batch


def load_model(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[FlowWaypointHead, Proj_Actiontokens | None, dict[str, Any] | None]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = FlowWaypointHeadConfig(**checkpoint["config"])
    model = FlowWaypointHead(config)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    projector = None
    if checkpoint.get("projector") is not None:
        projector_config = checkpoint["projector_config"]
        projector = Proj_Actiontokens(**projector_config)
        projector.load_state_dict(checkpoint["projector"])
        projector.to(device)
        projector.eval()
        for param in projector.parameters():
            param.requires_grad = False
    return model, projector, checkpoint.get("normalization_stats")


def stats_from_checkpoint(stats_dict: dict[str, Any] | None) -> NormalizationStats | None:
    if stats_dict is None:
        return None
    return NormalizationStats(
        waypoint_mean=torch.tensor(stats_dict["waypoint_mean"], dtype=torch.float32),
        waypoint_std=torch.tensor(stats_dict["waypoint_std"], dtype=torch.float32),
        robot_state_mean=(
            torch.tensor(stats_dict["robot_state_mean"], dtype=torch.float32)
            if stats_dict.get("robot_state_mean") is not None
            else None
        ),
        robot_state_std=(
            torch.tensor(stats_dict["robot_state_std"], dtype=torch.float32)
            if stats_dict.get("robot_state_std") is not None
            else None
        ),
    )


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    model, projector, stats_dict = load_model(args.checkpoint, device)
    stats = stats_from_checkpoint(stats_dict)
    config = model.config

    dataset = TrajectoryEmbeddingDataset(
        args.data,
        horizon=config.horizon,
        waypoint_dim=config.waypoint_dim,
        normalize_waypoints=stats is not None,
        normalize_robot_state=stats is not None and stats.robot_state_mean is not None,
        stats=stats,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_samples)

    predictions = []
    targets = []
    trajectory_ids = []
    timesteps = []

    with torch.no_grad():
        for batch in loader:
            if projector is not None:
                if "raw_action_embeddings" not in batch:
                    raise KeyError(
                        "This checkpoint includes an AsyncVLA projector, so inference data must include "
                        "raw_action_embeddings/actions_hidden_states."
                    )
                raw_context = batch["raw_action_embeddings"].to(device)
                modality_id = batch.get("modality_id")
                if modality_id is None:
                    modality_id = torch.full((raw_context.shape[0],), args.default_modality_id, device=device)
                else:
                    modality_id = modality_id.to(device)
                context = projector.predict_action(raw_context, modality_id)
            else:
                context = batch["action_embeddings"].to(device)
            robot_state = batch.get("robot_state")
            if robot_state is not None:
                robot_state = robot_state.to(device)

            pred = model.sample(
                context=context,
                robot_state=robot_state,
                num_steps=args.num_steps,
                clamp=False,
            )

            target = batch["waypoints"].to(device)
            if stats is not None:
                mean = stats.waypoint_mean.to(device)
                std = stats.waypoint_std.to(device)
                pred = pred * std + mean
                target = target * std + mean

            if not args.no_clamp:
                pred = clamp_waypoints(
                    pred,
                    max_forward_distance=config.max_forward_distance,
                    max_lateral_distance=config.max_lateral_distance,
                    max_yaw_change=config.max_yaw_change,
                )

            predictions.append(pred.cpu().numpy())
            targets.append(target.cpu().numpy())
            trajectory_ids.extend(batch.get("trajectory_id", [""] * pred.shape[0]))
            timesteps.append(batch["timestep"].cpu().numpy())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        pred_waypoints=np.concatenate(predictions, axis=0),
        target_waypoints=np.concatenate(targets, axis=0),
        trajectory_id=np.asarray(trajectory_ids),
        timestep=np.concatenate(timesteps, axis=0),
    )
    print(f"Saved predictions to {args.output}")


if __name__ == "__main__":
    main()
