from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a line-based manifest of trajectory.npz files.")
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--missing-only",
        action="store_true",
        help="Include only trajectories that do not already have the requested embedding output file.",
    )
    parser.add_argument("--embedding-name", default="trajectory_with_embeddings.npz")
    parser.add_argument("--include", action="append", default=[], help="Dataset name to include, e.g. scand. Repeatable.")
    parser.add_argument("--exclude", action="append", default=[], help="Dataset name to exclude. Repeatable.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    include = set(args.include)
    exclude = set(args.exclude)
    files = []
    for path in sorted(args.processed_root.glob("*/*/trajectory.npz")):
        dataset_name = path.parent.parent.name
        if include and dataset_name not in include:
            continue
        if dataset_name in exclude:
            continue
        if args.missing_only and (path.parent / args.embedding_name).exists():
            continue
        files.append(path.resolve())

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(str(path) for path in files) + ("\n" if files else ""), encoding="utf-8")
    print(f"Wrote {len(files)} trajectories to {args.out}")
    if files:
        print(f"Submit array range: 0-{len(files) - 1}")


if __name__ == "__main__":
    main()
