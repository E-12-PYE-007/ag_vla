from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from flow_head.scand_conversion import decode_image_msg, import_rosbags, msg_time, msgtype_matches, write_json


IMAGE_TYPES = {"sensor_msgs/msg/Image", "sensor_msgs/Image", "sensor_msgs/msg/CompressedImage", "sensor_msgs/CompressedImage"}
ODOM_TYPES = {"nav_msgs/msg/Odometry", "nav_msgs/Odometry"}
TWIST_TYPES = {"geometry_msgs/msg/Twist", "geometry_msgs/Twist", "geometry_msgs/msg/TwistStamped", "geometry_msgs/TwistStamped"}
LASER_TYPES = {"sensor_msgs/msg/LaserScan", "sensor_msgs/LaserScan"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect a HuRoN ROS bag without modifying it.")
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--sample-images", type=int, default=1)
    return parser.parse_args()


def empty_topic_stats(connection: Any) -> dict[str, Any]:
    return {
        "topic": connection.topic,
        "msgtype": connection.msgtype,
        "count": 0,
        "first_time": None,
        "last_time": None,
        "duration_sec": None,
        "rate_hz": None,
        "image_width": None,
        "image_height": None,
        "image_encoding": None,
    }


def update_rate(item: dict[str, Any]) -> None:
    if item["first_time"] is None or item["last_time"] is None:
        return
    duration = float(item["last_time"] - item["first_time"])
    item["duration_sec"] = duration
    item["rate_hz"] = float(item["count"] - 1) / duration if duration > 0 and item["count"] > 1 else None


def likely_role(topic: str, msgtype: str) -> str | None:
    low = topic.lower()
    if msgtype_matches(msgtype, IMAGE_TYPES):
        if "depth" in low:
            return "depth_image"
        if "spherical" in low or "theta" in low:
            return "spherical_or_equirectangular_image"
        if "panorama" in low:
            return "panorama_image"
        if "fisheye" in low:
            return "forward_fisheye_candidate"
        return "image_candidate"
    if msgtype_matches(msgtype, ODOM_TYPES):
        return "odometry_candidate"
    if msgtype_matches(msgtype, TWIST_TYPES):
        return "velocity_candidate"
    if msgtype_matches(msgtype, LASER_TYPES):
        return "laser_candidate"
    return None


def inspect_bag(bag: Path, sample_images: int) -> dict[str, Any]:
    AnyReader = import_rosbags()
    topics: dict[int, dict[str, Any]] = {}
    image_samples_seen: defaultdict[int, int] = defaultdict(int)

    with AnyReader([bag]) as reader:
        for connection in reader.connections:
            topics[connection.id] = empty_topic_stats(connection)

        for connection, timestamp, rawdata in reader.messages():
            item = topics[connection.id]
            item["count"] += 1
            msg = None
            time = float(timestamp) * 1e-9
            if item["first_time"] is None:
                item["first_time"] = time
            item["last_time"] = time

            if msgtype_matches(connection.msgtype, IMAGE_TYPES) and image_samples_seen[connection.id] < sample_images:
                msg = reader.deserialize(rawdata, connection.msgtype)
                image_time = msg_time(msg, timestamp)
                item["first_header_time"] = item.get("first_header_time", image_time)
                try:
                    image = decode_image_msg(msg, connection.msgtype)
                    item["image_width"] = image.width
                    item["image_height"] = image.height
                    item["image_encoding"] = getattr(msg, "encoding", getattr(msg, "format", None))
                except Exception as exc:
                    item["image_decode_error"] = repr(exc)
                image_samples_seen[connection.id] += 1

    topic_items = []
    for item in topics.values():
        update_rate(item)
        role = likely_role(item["topic"], item["msgtype"])
        if role:
            item["likely_role"] = role
        topic_items.append(item)

    image_topics = [item for item in topic_items if msgtype_matches(item["msgtype"], IMAGE_TYPES)]
    odom_topics = [item for item in topic_items if msgtype_matches(item["msgtype"], ODOM_TYPES)]
    twist_topics = [item for item in topic_items if msgtype_matches(item["msgtype"], TWIST_TYPES)]

    recommended_image = None
    for item in image_topics:
        if item["topic"] == "/fisheye_image/compressed":
            recommended_image = item["topic"]
            break
    if recommended_image is None and image_topics:
        recommended_image = max(
            image_topics,
            key=lambda x: (
                "fisheye" in x["topic"].lower(),
                "depth" not in x["topic"].lower(),
                int(x["count"]),
            ),
        )["topic"]

    recommended_odom = "/odometry" if any(item["topic"] == "/odometry" for item in odom_topics) else (
        max(odom_topics, key=lambda x: int(x["count"]))["topic"] if odom_topics else None
    )

    first_times = [item["first_time"] for item in topic_items if item["first_time"] is not None]
    last_times = [item["last_time"] for item in topic_items if item["last_time"] is not None]
    bag_first = min(first_times) if first_times else None
    bag_last = max(last_times) if last_times else None

    return {
        "bag": str(bag),
        "bag_first_time": bag_first,
        "bag_last_time": bag_last,
        "bag_duration_sec": float(bag_last - bag_first) if bag_first is not None and bag_last is not None else None,
        "topics": sorted(topic_items, key=lambda x: x["topic"]),
        "image_topics": image_topics,
        "odometry_topics": odom_topics,
        "velocity_topics": twist_topics,
        "recommended": {
            "image_topic": recommended_image,
            "odom_topic": recommended_odom,
        },
    }


def main() -> None:
    args = parse_args()
    summary = inspect_bag(args.bag, args.sample_images)
    write_json(args.out_json, summary)

    print(f"bag: {args.bag}")
    print(f"duration_sec: {summary['bag_duration_sec']}")
    print("\nImage topics:")
    for item in summary["image_topics"]:
        print(f"  {item['topic']}  {item['msgtype']}  count={item['count']}  rate={item['rate_hz']}  size={item['image_width']}x{item['image_height']}")
    print("\nOdometry topics:")
    for item in summary["odometry_topics"]:
        print(f"  {item['topic']}  {item['msgtype']}  count={item['count']}  rate={item['rate_hz']}")
    print("\nRecommended:")
    print(json.dumps(summary["recommended"], indent=2))
    print(f"\nSaved {args.out_json}")


if __name__ == "__main__":
    main()
