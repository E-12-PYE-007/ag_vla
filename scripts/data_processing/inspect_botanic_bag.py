from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from flow_head.scand_conversion import inspect_bag_topics, print_topic_summary, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect a BotanicGarden ROS1 bag without modifying it.")
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = inspect_bag_topics(args.bag)
    expected = {
        "left_rgb": "/dalsa_rgb/left/image_raw",
        "right_rgb": "/dalsa_rgb/right/image_raw",
        "gt_poses": "/gt_poses",
    }
    topics = {item["topic"] for item in summary["topics"]}
    summary["expected_botanic_topics"] = expected
    summary["expected_topic_status"] = {name: topic in topics for name, topic in expected.items()}
    write_json(args.out_json, summary)
    print_topic_summary(summary)
    print("\nExpected BotanicGarden topic status:")
    print(json.dumps(summary["expected_topic_status"], indent=2))
    print(f"\nSaved {args.out_json}")


if __name__ == "__main__":
    main()
