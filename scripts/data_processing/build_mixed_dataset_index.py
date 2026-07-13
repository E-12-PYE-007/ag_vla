from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an index over processed_mixed trajectories.")
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def scalar_string(value: np.ndarray | str, fallback: str) -> str:
    try:
        arr = np.asarray(value)
        if arr.shape == ():
            return str(arr.item())
    except Exception:
        pass
    return fallback


def main() -> None:
    args = parse_args()
    trajectories = []
    dataset_counts: dict[str, int] = {}
    total_samples = 0

    for npz_path in sorted(args.processed_root.glob("*/*/trajectory.npz")):
        data = np.load(npz_path, allow_pickle=True)
        num_samples = int(len(data["target_waypoints"]))
        dataset_name = scalar_string(
            data["dataset_name"] if "dataset_name" in data.files else "",
            npz_path.parent.parent.name,
        )
        trajectory_name = scalar_string(
            data["trajectory_name"] if "trajectory_name" in data.files else "",
            npz_path.parent.name,
        )
        item = {
            "dataset_name": dataset_name,
            "trajectory_name": trajectory_name,
            "npz_path": npz_path.relative_to(args.processed_root.parent).as_posix(),
            "num_samples": num_samples,
        }
        trajectories.append(item)
        dataset_counts[dataset_name] = dataset_counts.get(dataset_name, 0) + num_samples
        total_samples += num_samples

    payload = {
        "trajectories": trajectories,
        "total_samples": total_samples,
        "datasets": dataset_counts,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Indexed {len(trajectories)} trajectories")
    print(f"Total samples: {total_samples}")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
