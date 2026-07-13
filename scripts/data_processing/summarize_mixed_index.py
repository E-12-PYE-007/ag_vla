from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize a processed_mixed/mixed_index.json file.")
    parser.add_argument("--index", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.index.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    trajectories = payload.get("trajectories", [])
    datasets = payload.get("datasets", {})
    total_samples = int(payload.get("total_samples", 0))

    print(f"index: {args.index}")
    print(f"trajectories: {len(trajectories)}")
    print(f"total_samples: {total_samples}")
    print()
    print("datasets:")
    for name, count in sorted(datasets.items()):
        traj_count = sum(1 for item in trajectories if item.get("dataset_name") == name)
        fraction = 100.0 * float(count) / max(total_samples, 1)
        print(f"  {name}: {count} samples, {traj_count} trajectories, {fraction:.1f}%")

    if trajectories:
        sample_counts = sorted(int(item.get("num_samples", 0)) for item in trajectories)
        midpoint = len(sample_counts) // 2
        median = sample_counts[midpoint] if len(sample_counts) % 2 else (
            sample_counts[midpoint - 1] + sample_counts[midpoint]
        ) / 2
        print()
        print("trajectory sample counts:")
        print(f"  min: {sample_counts[0]}")
        print(f"  median: {median}")
        print(f"  max: {sample_counts[-1]}")

        print()
        print("smallest trajectories:")
        for item in sorted(trajectories, key=lambda row: int(row.get("num_samples", 0)))[:10]:
            print(
                f"  {item.get('dataset_name')}/{item.get('trajectory_name')}: "
                f"{int(item.get('num_samples', 0))} samples"
            )


if __name__ == "__main__":
    main()
