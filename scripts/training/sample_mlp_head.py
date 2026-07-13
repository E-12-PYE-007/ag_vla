from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from flow_head.asyncvla_projector import Proj_Actiontokens
from flow_head.dataset import NormalizationStats, TrajectoryEmbeddingDataset
from flow_head.mlp_waypoint_head import MLPWaypointHead, MLPWaypointHeadConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print one deterministic MLP waypoint-head prediction.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--clamp", action="store_true")
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = MLPWaypointHeadConfig(**checkpoint["config"])
    stats = None
    if checkpoint.get("normalization_stats") is not None:
        raw_stats = checkpoint["normalization_stats"]
        stats = NormalizationStats(
            waypoint_mean=torch.tensor(raw_stats["waypoint_mean"], dtype=torch.float32),
            waypoint_std=torch.tensor(raw_stats["waypoint_std"], dtype=torch.float32),
            robot_state_mean=(
                torch.tensor(raw_stats["robot_state_mean"], dtype=torch.float32)
                if raw_stats.get("robot_state_mean") is not None
                else None
            ),
            robot_state_std=(
                torch.tensor(raw_stats["robot_state_std"], dtype=torch.float32)
                if raw_stats.get("robot_state_std") is not None
                else None
            ),
        )

    dataset = TrajectoryEmbeddingDataset(
        args.data,
        horizon=config.horizon,
        waypoint_dim=config.waypoint_dim,
        normalize_waypoints=stats is not None,
        normalize_robot_state=stats is not None and stats.robot_state_mean is not None,
        stats=stats,
    )
    sample = dataset[args.index]
    model = MLPWaypointHead(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    projector = None
    if checkpoint.get("projector") is not None:
        projector = Proj_Actiontokens(**checkpoint["projector_config"]).to(device)
        projector.load_state_dict(checkpoint["projector"])
        projector.eval()

    modality_id = sample.get("modality_id")
    if modality_id is not None:
        modality_id = modality_id[None].to(device)
    if projector is not None:
        if "raw_action_embeddings" not in sample:
            raise KeyError("This checkpoint includes a projector, so the sample must contain raw_action_embeddings.")
        raw_context = sample["raw_action_embeddings"][None].to(device)
        if modality_id is None:
            modality_id = torch.zeros((raw_context.shape[0],), device=device)
        context = projector.predict_action(raw_context, modality_id)
    else:
        context = sample["action_embeddings"][None].to(device)

    robot_state = sample.get("robot_state")
    if robot_state is not None:
        robot_state = robot_state[None].to(device)

    pred = model.predict(context=context, robot_state=robot_state, modality_id=modality_id, clamp=args.clamp)[0].cpu()
    target = sample["waypoints"].cpu()
    if stats is not None:
        pred = pred * stats.waypoint_std + stats.waypoint_mean
        target = target * stats.waypoint_std + stats.waypoint_mean

    print(f"trajectory_id: {sample.get('trajectory_id')}")
    print(f"timestep: {int(sample['timestep'])}")
    print("prediction:")
    print(pred)
    print("target:")
    print(target)
    print("l1:", torch.nn.functional.l1_loss(pred, target).item())


if __name__ == "__main__":
    main()
