from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from flow_head.asyncvla_projector import Proj_Actiontokens
from flow_head.dataset import NormalizationStats, TrajectoryEmbeddingDataset
from flow_head.evaluation import MetricAverager, save_metrics, save_prediction_npz, waypoint_metrics
from flow_head.flow_waypoint_head import FlowWaypointHead, FlowWaypointHeadConfig, clamp_waypoints
from flow_head.splits import dataset_indices_for_split, load_split_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a flow-matching waypoint head.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split-json", type=Path, default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", type=Path, default=Path("eval/flow_head"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-steps", type=int, default=20)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--no-clamp", action="store_true")
    parser.add_argument("--default-modality-id", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=7)
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
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = FlowWaypointHeadConfig(**checkpoint["config"])
    stats = stats_from_checkpoint(checkpoint.get("normalization_stats"))

    dataset = TrajectoryEmbeddingDataset(
        args.data,
        horizon=config.horizon,
        waypoint_dim=config.waypoint_dim,
        normalize_waypoints=stats is not None,
        normalize_robot_state=stats is not None and stats.robot_state_mean is not None,
        stats=stats,
    )
    if args.split_json is not None:
        indices = dataset_indices_for_split(dataset, load_split_file(args.split_json), args.split)
        dataset = Subset(dataset, indices)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_samples)

    model = FlowWaypointHead(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    projector = None
    if checkpoint.get("projector") is not None:
        projector = Proj_Actiontokens(**checkpoint["projector_config"]).to(device)
        projector.load_state_dict(checkpoint["projector"])
        projector.eval()

    averager = MetricAverager()
    predictions = []
    targets = []
    trajectory_ids = []
    timesteps = []
    with torch.no_grad():
        for batch in loader:
            if projector is not None:
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
            pred = model.sample(context=context, robot_state=robot_state, num_steps=args.num_steps, clamp=False)
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
            metrics = waypoint_metrics(pred, target)
            averager.update(metrics, n=pred.shape[0])
            predictions.append(pred.cpu().numpy())
            targets.append(target.cpu().numpy())
            trajectory_ids.extend(batch.get("trajectory_id", [""] * pred.shape[0]))
            timesteps.append(batch["timestep"].cpu().numpy())

    final_metrics = averager.compute() | {"num_samples": len(dataset), "split": args.split, "num_steps": args.num_steps}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_metrics(args.output_dir / "metrics.json", final_metrics)
    save_prediction_npz(args.output_dir / "predictions.npz", predictions, targets, trajectory_ids, timesteps)
    print(final_metrics)
    print(f"Saved {args.output_dir}")


if __name__ == "__main__":
    main()
