from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from flow_head.scand_conversion import inspect_bag_topics, print_topic_summary, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect a SCAND/Jackal ROS1 bag without ROS.")
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--processed-root", type=Path, default=Path("processed_mixed/scand"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = inspect_bag_topics(args.bag)
    print_topic_summary(summary)

    out_dir = args.processed_root / args.bag.stem
    out_path = out_dir / "topic_summary.json"
    write_json(out_path, summary)
    print(f"\nSaved topic summary to {out_path}")


if __name__ == "__main__":
    main()
