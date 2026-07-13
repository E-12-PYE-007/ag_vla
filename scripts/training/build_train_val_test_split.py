from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a fixed trajectory-level train/val/test split.")
    parser.add_argument("--data-root", type=Path, required=True, help="processed_mixed root.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--file-name",
        default="trajectory_with_embeddings.npz",
        help="Use trajectory_with_embeddings.npz for training splits; use trajectory.npz only for pre-embedding planning.",
    )
    return parser.parse_args()


def dataset_name(path: Path, root: Path) -> str:
    rel = path.relative_to(root)
    return rel.parts[0] if len(rel.parts) > 1 else "unknown"


def split_group(paths: list[Path], rng: random.Random, train_fraction: float, val_fraction: float) -> tuple[list[Path], list[Path], list[Path]]:
    paths = list(paths)
    rng.shuffle(paths)
    n = len(paths)
    if n == 0:
        return [], [], []
    train_n = int(round(n * train_fraction))
    val_n = int(round(n * val_fraction))
    if n >= 3:
        train_n = max(1, min(train_n, n - 2))
        val_n = max(1, min(val_n, n - train_n - 1))
    elif n == 2:
        train_n, val_n = 1, 0
    else:
        train_n, val_n = 1, 0
    test_n = n - train_n - val_n
    if n >= 3 and test_n == 0:
        test_n = 1
        train_n = max(1, train_n - 1)
    return paths[:train_n], paths[train_n : train_n + val_n], paths[train_n + val_n :]


def main() -> None:
    args = parse_args()
    total = args.train_fraction + args.val_fraction + args.test_fraction
    if abs(total - 1.0) > 1e-6:
        raise SystemExit(f"Fractions must sum to 1.0, got {total}")

    files = sorted(args.data_root.glob(f"*/*/{args.file_name}"))
    if not files:
        raise SystemExit(f"No {args.file_name} files found under {args.data_root}")

    by_dataset: dict[str, list[Path]] = {}
    for path in files:
        by_dataset.setdefault(dataset_name(path, args.data_root), []).append(path)

    rng = random.Random(args.seed)
    splits = {"train": [], "val": [], "test": []}
    dataset_counts: dict[str, dict[str, int]] = {}
    for name, paths in sorted(by_dataset.items()):
        train, val, test = split_group(paths, rng, args.train_fraction, args.val_fraction)
        groups = {"train": train, "val": val, "test": test}
        dataset_counts[name] = {}
        for split_name, split_paths in groups.items():
            splits[split_name].extend(path.relative_to(args.data_root).as_posix() for path in split_paths)
            dataset_counts[name][split_name] = len(split_paths)

    payload = {
        "data_root": str(args.data_root),
        "file_name": args.file_name,
        "seed": args.seed,
        "fractions": {
            "train": args.train_fraction,
            "val": args.val_fraction,
            "test": args.test_fraction,
        },
        "splits": {key: sorted(value) for key, value in splits.items()},
        "dataset_counts": dataset_counts,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved {args.out}")
    for split_name in ["train", "val", "test"]:
        print(f"{split_name}: {len(splits[split_name])} trajectories")
    print("by dataset:")
    for name, counts in dataset_counts.items():
        print(f"  {name}: {counts}")


if __name__ == "__main__":
    main()
