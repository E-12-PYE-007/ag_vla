from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
from PIL import Image


IMAGE_TYPES = {
    "sensor_msgs/msg/Image",
    "sensor_msgs/Image",
    "sensor_msgs/msg/CompressedImage",
    "sensor_msgs/CompressedImage",
}
ODOM_TYPES = {
    "nav_msgs/msg/Odometry",
    "nav_msgs/Odometry",
    "geometry_msgs/msg/PoseStamped",
    "geometry_msgs/PoseStamped",
}
VELOCITY_TYPES = {
    "geometry_msgs/msg/Twist",
    "geometry_msgs/Twist",
    "sensor_msgs/msg/Joy",
    "sensor_msgs/Joy",
}


@dataclass
class OdomSample:
    time: float
    position: np.ndarray
    yaw: float
    velocity: np.ndarray


@dataclass
class ImageRecord:
    time: float
    msg: Any
    msgtype: str


def import_rosbags() -> Any:
    try:
        from rosbags.highlevel import AnyReader
    except ImportError as exc:
        raise ImportError(
            "Missing dependency 'rosbags'. Install with: "
            "python3 -m pip install rosbags numpy opencv-python pillow tqdm matplotlib"
        ) from exc
    return AnyReader


def msgtype_matches(msgtype: str, candidates: set[str]) -> bool:
    normalized = msgtype.replace("/", "/msg/", 1) if "/msg/" not in msgtype and "/" in msgtype else msgtype
    return msgtype in candidates or normalized in candidates or any(msgtype.endswith(candidate.split("/")[-1]) for candidate in candidates)


def stamp_to_sec(stamp: Any) -> float:
    if stamp is None:
        return float("nan")
    sec = getattr(stamp, "sec", getattr(stamp, "secs", 0))
    nsec = getattr(stamp, "nanosec", getattr(stamp, "nsecs", 0))
    return float(sec) + float(nsec) * 1e-9


def msg_time(msg: Any, fallback_ns: int | float) -> float:
    header = getattr(msg, "header", None)
    stamp = getattr(header, "stamp", None)
    time = stamp_to_sec(stamp)
    if math.isfinite(time) and time > 0:
        return time
    return float(fallback_ns) * 1e-9


def quaternion_to_yaw(q: Any) -> float:
    x = float(getattr(q, "x", 0.0))
    y = float(getattr(q, "y", 0.0))
    z = float(getattr(q, "z", 0.0))
    w = float(getattr(q, "w", 1.0))
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def wrap_to_pi(angle: np.ndarray | float) -> np.ndarray | float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def odom_from_msg(msg: Any, msgtype: str, time: float) -> OdomSample:
    if msgtype_matches(msgtype, {"nav_msgs/msg/Odometry", "nav_msgs/Odometry"}):
        pose = msg.pose.pose
        twist = msg.twist.twist
        position = np.asarray([pose.position.x, pose.position.y], dtype=np.float32)
        yaw = quaternion_to_yaw(pose.orientation)
        velocity = np.asarray([twist.linear.x, twist.angular.z], dtype=np.float32)
        return OdomSample(time=time, position=position, yaw=float(yaw), velocity=velocity)

    if msgtype_matches(msgtype, {"geometry_msgs/msg/PoseStamped", "geometry_msgs/PoseStamped"}):
        pose = msg.pose
        position = np.asarray([pose.position.x, pose.position.y], dtype=np.float32)
        yaw = quaternion_to_yaw(pose.orientation)
        velocity = np.asarray([np.nan, np.nan], dtype=np.float32)
        return OdomSample(time=time, position=position, yaw=float(yaw), velocity=velocity)

    raise ValueError(f"Unsupported odometry/pose type: {msgtype}")


def estimate_missing_velocities(times: np.ndarray, positions: np.ndarray, yaw: np.ndarray, velocity: np.ndarray) -> np.ndarray:
    if len(times) < 2:
        return np.nan_to_num(velocity, nan=0.0)
    out = velocity.copy()
    missing = ~np.isfinite(out).all(axis=1)
    if not missing.any():
        return out
    dt = np.gradient(times)
    dt = np.where(np.abs(dt) < 1e-6, np.nan, dt)
    dx = np.gradient(positions[:, 0])
    dy = np.gradient(positions[:, 1])
    speed_world = np.sqrt(dx * dx + dy * dy) / dt
    dyaw = np.gradient(np.unwrap(yaw)) / dt
    out[missing, 0] = speed_world[missing]
    out[missing, 1] = dyaw[missing]
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def nearest_index(sorted_times: np.ndarray, query_time: float) -> tuple[int, float]:
    idx = int(np.searchsorted(sorted_times, query_time))
    candidates = []
    if idx < len(sorted_times):
        candidates.append(idx)
    if idx > 0:
        candidates.append(idx - 1)
    if not candidates:
        return -1, float("inf")
    best = min(candidates, key=lambda i: abs(float(sorted_times[i]) - query_time))
    return best, abs(float(sorted_times[best]) - query_time)


def local_waypoint(current_pos: np.ndarray, current_yaw: float, future_pos: np.ndarray, future_yaw: float) -> np.ndarray:
    dx_world = float(future_pos[0] - current_pos[0])
    dy_world = float(future_pos[1] - current_pos[1])
    cos_yaw = math.cos(current_yaw)
    sin_yaw = math.sin(current_yaw)
    delta_x = cos_yaw * dx_world + sin_yaw * dy_world
    delta_y = -sin_yaw * dx_world + cos_yaw * dy_world
    delta_yaw = wrap_to_pi(float(future_yaw - current_yaw))
    return np.asarray([delta_x, delta_y, delta_yaw], dtype=np.float32)


def generate_waypoint_chunk(
    odom_times: np.ndarray,
    positions: np.ndarray,
    yaw: np.ndarray,
    current_idx: int,
    horizon: int,
    waypoint_dt: float,
    future_time_tolerance: float,
) -> Optional[np.ndarray]:
    current_time = float(odom_times[current_idx])
    current_pos = positions[current_idx]
    current_yaw = float(yaw[current_idx])
    waypoints = []
    for step in range(1, horizon + 1):
        desired_time = current_time + step * waypoint_dt
        if desired_time > float(odom_times[-1]):
            return None
        future_idx, time_diff = nearest_index(odom_times, desired_time)
        if future_idx < 0 or time_diff > future_time_tolerance:
            return None
        waypoints.append(local_waypoint(current_pos, current_yaw, positions[future_idx], float(yaw[future_idx])))
    return np.stack(waypoints, axis=0).astype(np.float32)


def waypoint_is_valid(waypoints: np.ndarray, max_final_distance: float, max_abs_yaw: float) -> bool:
    if waypoints.shape[-2:] != (len(waypoints), 3):
        return False
    if not np.isfinite(waypoints).all():
        return False
    final_distance = float(np.linalg.norm(waypoints[-1, :2]))
    if final_distance > max_final_distance:
        return False
    if float(np.max(np.abs(waypoints[:, 2]))) > max_abs_yaw:
        return False
    yaw_jumps = np.abs(np.diff(waypoints[:, 2]))
    if len(yaw_jumps) and float(np.max(yaw_jumps)) > max_abs_yaw:
        return False
    return True


def _bytes_from_ros_data(data: Any) -> bytes:
    if isinstance(data, bytes):
        return data
    if isinstance(data, bytearray):
        return bytes(data)
    if isinstance(data, memoryview):
        return data.tobytes()
    arr = np.asarray(data)
    return arr.tobytes()


def decode_image_msg(msg: Any, msgtype: str) -> Image.Image:
    if msgtype_matches(msgtype, {"sensor_msgs/msg/CompressedImage", "sensor_msgs/CompressedImage"}):
        data = _bytes_from_ros_data(msg.data)
        try:
            import cv2

            arr = np.frombuffer(data, dtype=np.uint8)
            decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if decoded is None:
                raise ValueError("cv2.imdecode returned None")
            decoded = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
            return Image.fromarray(decoded)
        except Exception:
            from io import BytesIO

            return Image.open(BytesIO(data)).convert("RGB")

    height = int(msg.height)
    width = int(msg.width)
    encoding = str(msg.encoding).lower()
    data = np.frombuffer(_bytes_from_ros_data(msg.data), dtype=np.uint8)

    if encoding in {"rgb8", "bgr8"}:
        arr = data.reshape(height, int(msg.step))[:, : width * 3].reshape(height, width, 3)
        if encoding == "bgr8":
            arr = arr[:, :, ::-1]
        return Image.fromarray(arr, mode="RGB")

    if encoding in {"mono8", "8uc1"}:
        arr = data.reshape(height, int(msg.step))[:, :width]
        return Image.fromarray(arr, mode="L").convert("RGB")

    raise ValueError(f"Unsupported sensor_msgs/Image encoding: {msg.encoding}")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def relative_or_absolute_image_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def inspect_bag_topics(bag: Path) -> dict[str, Any]:
    AnyReader = import_rosbags()
    summary: dict[str, Any] = {
        "bag": str(bag),
        "topics": [],
        "likely_image_topics": [],
        "likely_odom_topics": [],
        "likely_velocity_topics": [],
        "recommended": {},
    }

    with AnyReader([bag]) as reader:
        counts = {connection.id: 0 for connection in reader.connections}
        for connection, _, _ in reader.messages():
            counts[connection.id] += 1

        for connection in reader.connections:
            item = {
                "topic": connection.topic,
                "msgtype": connection.msgtype,
                "count": counts.get(connection.id, 0),
            }
            summary["topics"].append(item)
            if msgtype_matches(connection.msgtype, IMAGE_TYPES):
                summary["likely_image_topics"].append(item)
            if msgtype_matches(connection.msgtype, ODOM_TYPES):
                summary["likely_odom_topics"].append(item)
            if msgtype_matches(connection.msgtype, VELOCITY_TYPES):
                summary["likely_velocity_topics"].append(item)

    def image_topic_score(item: dict[str, Any]) -> tuple[int, int]:
        topic = str(item["topic"]).lower()
        score = 0
        if any(token in topic for token in ("rgb", "color", "colour", "left")):
            score += 100
        if any(token in topic for token in ("depth", "disparity", "infra", "ir")):
            score -= 100
        if "compressed" in topic:
            score += 5
        return score, int(item["count"])

    if summary["likely_image_topics"]:
        summary["recommended"]["image_topic"] = max(summary["likely_image_topics"], key=image_topic_score)["topic"]
    if summary["likely_odom_topics"]:
        summary["recommended"]["odom_topic"] = max(summary["likely_odom_topics"], key=lambda x: x["count"])["topic"]
    return summary


def print_topic_summary(summary: dict[str, Any]) -> None:
    print("All topics:")
    for item in summary["topics"]:
        print(f"  {item['topic']}  {item['msgtype']}  {item['count']} messages")

    def print_group(title: str, key: str) -> None:
        print(f"\n{title}:")
        if not summary[key]:
            print("  none found")
            return
        for item in summary[key]:
            print(f"  {item['topic']}  {item['msgtype']}  {item['count']} messages")

    print_group("Likely image topics", "likely_image_topics")
    print_group("Likely odom topics", "likely_odom_topics")
    print_group("Likely velocity/control topics", "likely_velocity_topics")
    print("\nRecommended:")
    print(f"  image_topic = {summary['recommended'].get('image_topic')}")
    print(f"  odom_topic = {summary['recommended'].get('odom_topic')}")


def load_records_from_bag(bag: Path, image_topic: str, odom_topic: str) -> tuple[list[ImageRecord], list[OdomSample], dict[str, int]]:
    AnyReader = import_rosbags()
    images: list[ImageRecord] = []
    odom: list[OdomSample] = []
    counts = {"num_raw_images": 0, "num_raw_odom": 0}

    with AnyReader([bag]) as reader:
        connections = [
            connection
            for connection in reader.connections
            if connection.topic in {image_topic, odom_topic}
        ]
        if not connections:
            raise ValueError(f"No requested topics found in bag: image={image_topic}, odom={odom_topic}")

        for connection, timestamp, rawdata in reader.messages(connections=connections):
            msg = reader.deserialize(rawdata, connection.msgtype)
            time = msg_time(msg, timestamp)
            if connection.topic == image_topic:
                counts["num_raw_images"] += 1
                images.append(ImageRecord(time=time, msg=msg, msgtype=connection.msgtype))
            elif connection.topic == odom_topic:
                counts["num_raw_odom"] += 1
                odom.append(odom_from_msg(msg, connection.msgtype, time))

    images.sort(key=lambda item: item.time)
    odom.sort(key=lambda item: item.time)
    return images, odom, counts


def convert_bag_to_training_data(
    bag: Path,
    image_topic: str,
    odom_topic: str,
    out_dir: Path,
    horizon: int = 8,
    waypoint_dt: float = 0.5,
    image_stride: int = 5,
    sync_threshold: float = 0.10,
    max_final_distance: float = 5.0,
    max_abs_yaw: float = 3.14,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    image_records, odom_records, counts = load_records_from_bag(bag, image_topic, odom_topic)
    if not image_records:
        raise ValueError(f"No image messages found on {image_topic}")
    if len(odom_records) < horizon + 1:
        raise ValueError(f"Not enough odometry messages on {odom_topic}: {len(odom_records)}")

    odom_times = np.asarray([sample.time for sample in odom_records], dtype=np.float64)
    positions = np.stack([sample.position for sample in odom_records], axis=0).astype(np.float32)
    yaw = np.asarray([sample.yaw for sample in odom_records], dtype=np.float32)
    velocity = np.stack([sample.velocity for sample in odom_records], axis=0).astype(np.float32)
    velocity = estimate_missing_velocities(odom_times, positions, yaw, velocity)

    saved_image_paths: list[str] = []
    saved_times: list[float] = []
    saved_positions: list[np.ndarray] = []
    saved_yaw: list[float] = []
    saved_velocity: list[np.ndarray] = []
    saved_waypoints: list[np.ndarray] = []

    future_tolerance = max(sync_threshold, waypoint_dt * 0.5)
    candidate_images = image_records[:: max(1, image_stride)]

    for image_record in candidate_images:
        current_idx, sync_error = nearest_index(odom_times, image_record.time)
        if current_idx < 0 or sync_error > sync_threshold:
            continue

        waypoints = generate_waypoint_chunk(
            odom_times=odom_times,
            positions=positions,
            yaw=yaw,
            current_idx=current_idx,
            horizon=horizon,
            waypoint_dt=waypoint_dt,
            future_time_tolerance=future_tolerance,
        )
        if waypoints is None or not waypoint_is_valid(waypoints, max_final_distance, max_abs_yaw):
            continue

        image_idx = len(saved_image_paths)
        image_path = images_dir / f"{image_idx:06d}.jpg"
        try:
            decoded = decode_image_msg(image_record.msg, image_record.msgtype)
            decoded.save(image_path, quality=95)
        except Exception as exc:
            print(f"Skipping image at t={image_record.time:.3f}: {exc}")
            continue

        saved_image_paths.append(relative_or_absolute_image_path(image_path, out_dir))
        saved_times.append(float(image_record.time))
        saved_positions.append(positions[current_idx])
        saved_yaw.append(float(yaw[current_idx]))
        saved_velocity.append(velocity[current_idx])
        saved_waypoints.append(waypoints)

    if not saved_waypoints:
        raise ValueError("No valid synchronized image/waypoint samples were produced.")

    image_paths_arr = np.asarray(saved_image_paths, dtype=str)
    times_arr = np.asarray(saved_times, dtype=np.float64)
    position_arr = np.stack(saved_positions, axis=0).astype(np.float32)
    yaw_arr = np.asarray(saved_yaw, dtype=np.float32)
    velocity_arr = np.stack(saved_velocity, axis=0).astype(np.float32)
    waypoints_arr = np.stack(saved_waypoints, axis=0).astype(np.float32)

    np.savez_compressed(
        out_dir / "trajectory.npz",
        image_paths=image_paths_arr,
        times=times_arr,
        position=position_arr,
        yaw=yaw_arr,
        velocity=velocity_arr,
        target_waypoints=waypoints_arr,
        dataset_name=np.asarray("scand"),
        trajectory_name=np.asarray(bag.stem),
    )

    metadata = {
        "dataset_name": "scand",
        "trajectory_name": bag.stem,
        "bag": str(bag),
        "image_topic": image_topic,
        "odom_topic": odom_topic,
        "num_raw_images": counts["num_raw_images"],
        "num_raw_odom": counts["num_raw_odom"],
        "num_saved_samples": int(len(saved_waypoints)),
        "horizon": int(horizon),
        "waypoint_dt": float(waypoint_dt),
        "image_stride": int(image_stride),
        "sync_threshold": float(sync_threshold),
        "max_final_distance": float(max_final_distance),
        "max_abs_yaw": float(max_abs_yaw),
        "target_waypoints_shape": list(waypoints_arr.shape),
        "position_shape": list(position_arr.shape),
        "velocity_shape": list(velocity_arr.shape),
    }
    write_json(out_dir / "metadata.json", metadata)
    return metadata


def array_stats(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }
