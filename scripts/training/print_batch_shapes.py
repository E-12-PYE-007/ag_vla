from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from flow_head.dataset import TrajectoryEmbeddingDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print one flow-head training sample's shapes.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--waypoint-dim", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = TrajectoryEmbeddingDataset(args.data, horizon=args.horizon, waypoint_dim=args.waypoint_dim)
    sample = dataset[args.index]
    image = sample.get("image")
    print("image:", None if image is None else tuple(image.shape))
    print("action_embeddings:", None if "action_embeddings" not in sample else tuple(sample["action_embeddings"].shape))
    print(
        "raw_action_embeddings:",
        None if "raw_action_embeddings" not in sample else tuple(sample["raw_action_embeddings"].shape),
    )
    print("waypoints:", tuple(sample["waypoints"].shape))
    if "robot_state" in sample:
        print("robot_state:", tuple(sample["robot_state"].shape))
    if "modality_id" in sample:
        print("modality_id:", sample["modality_id"])
    print("trajectory_id:", sample.get("trajectory_id"))
    print("timestep:", int(sample["timestep"]))
    print("first_waypoint_chunk:")
    print(sample["waypoints"])


if __name__ == "__main__":
    main()
