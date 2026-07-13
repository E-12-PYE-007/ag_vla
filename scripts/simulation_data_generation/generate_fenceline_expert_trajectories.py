from __future__ import annotations

import argparse
import heapq
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml


@dataclass(frozen=True)
class Circle:
    name: str
    kind: str
    xy: np.ndarray
    radius: float


@dataclass(frozen=True)
class Segment:
    name: str
    a: np.ndarray
    b: np.ndarray
    radius: float


def wrap_to_pi(angle: np.ndarray | float) -> np.ndarray | float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def xy(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if len(arr) < 2:
        raise ValueError(f"Expected x/y coordinates, got {value!r}")
    return arr[:2]


def unit(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < 1e-6:
        raise ValueError("Zero-length vector.")
    return (vec / norm).astype(np.float32)


def seg_dist(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom < 1e-8:
        return np.linalg.norm(points - a, axis=-1)
    t = np.clip(((points - a) @ ab) / denom, 0.0, 1.0)
    return np.linalg.norm(points - (a + t[..., None] * ab), axis=-1)


def estimate_velocity(times: np.ndarray, positions: np.ndarray, yaws: np.ndarray) -> np.ndarray:
    velocity = np.zeros((len(positions), 2), dtype=np.float32)
    if len(positions) < 2:
        return velocity
    unwrapped = np.unwrap(yaws)
    for i in range(len(positions)):
        i0 = max(0, i - 1)
        i1 = min(len(positions) - 1, i + 1)
        dt = float(times[i1] - times[i0])
        if dt <= 1e-6:
            continue
        velocity[i, 0] = float(np.linalg.norm(positions[i1] - positions[i0]) / dt)
        velocity[i, 1] = float((unwrapped[i1] - unwrapped[i0]) / dt)
    return velocity


def local_waypoints(
    positions: np.ndarray,
    yaws: np.ndarray,
    horizon: int,
    spacing: float,
    min_final_distance: float,
    max_final_distance: float,
) -> tuple[np.ndarray, np.ndarray]:
    valid, chunks = [], []
    for i in range(len(positions) - 1):
        future = positions[i:]
        d = np.linalg.norm(np.diff(future, axis=0), axis=1)
        cumulative = np.concatenate([[0.0], np.cumsum(d)])
        if horizon * spacing > cumulative[-1]:
            continue
        chunk = []
        for h in range(1, horizon + 1):
            idx = int(np.searchsorted(cumulative, h * spacing, side="left"))
            idx = min(idx, len(future) - 1)
            p = future[idx]
            dx, dy = p - positions[i]
            c, s = math.cos(float(yaws[i])), math.sin(float(yaws[i]))
            local_x = c * dx + s * dy
            local_y = -s * dx + c * dy
            local_yaw = wrap_to_pi(float(yaws[i + idx] - yaws[i]))
            chunk.append([local_x, local_y, local_yaw])
        chunk_arr = np.asarray(chunk, dtype=np.float32)
        final_distance = float(np.linalg.norm(chunk_arr[-1, :2]))
        if min_final_distance <= final_distance <= max_final_distance:
            valid.append(i)
            chunks.append(chunk_arr)
    if not chunks:
        raise ValueError("No valid waypoint chunks were generated.")
    return np.asarray(valid, dtype=np.int64), np.stack(chunks).astype(np.float32)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate expert fenceline trajectories from simulation YAML files.")
    p.add_argument("--input", type=Path, required=True, help="YAML file or directory.")
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--num-trajectories", type=int, default=1)
    p.add_argument("--dataset-name", default="fenceline_sim")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--goal-x", type=float)
    p.add_argument("--goal-y", type=float)
    p.add_argument("--goal-end", choices=["auto", "start", "end"], default="auto")
    p.add_argument("--side", choices=["auto", "left", "right"], default="auto")
    p.add_argument("--fence-offset", type=float)
    p.add_argument("--rover-radius", type=float, default=0.28)
    p.add_argument("--safety-margin", type=float, default=0.12)
    p.add_argument("--fence-radius", type=float, default=0.10)
    p.add_argument("--post-radius", type=float, default=0.16)
    p.add_argument("--grid-resolution", type=float, default=0.08)
    p.add_argument("--bounds-margin", type=float, default=1.5)
    p.add_argument("--path-step", type=float, default=0.08)
    p.add_argument("--speed", type=float, default=0.35)
    p.add_argument("--horizon", type=int, default=8)
    p.add_argument("--target-spacing-m", type=float, default=0.25)
    p.add_argument("--min-final-distance", type=float, default=0.4)
    p.add_argument("--max-final-distance", type=float, default=5.0)
    p.add_argument("--plot", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not contain a YAML mapping.")
    return data


def obstacles(data: dict[str, Any], args: argparse.Namespace) -> tuple[list[Circle], list[Segment]]:
    circles: list[Circle] = []
    segments: list[Segment] = []
    specs: dict[str, Any] = {}
    assets = data.get("assets", {})
    for group in ("plants", "obstacles"):
        if isinstance(assets.get(group), dict):
            specs.update(assets[group])
    for obj in data.get("obstacles", []) or []:
        name = str(obj.get("name", "obstacle"))
        spec = specs.get(name, {})
        bbox = np.asarray(spec.get("bbox_size", [0.8, 0.8, 0.0]), dtype=np.float32)
        scale = float(spec.get("scale", 1.0))
        asset_radius = float(np.max(bbox[:2]) * scale * 0.5) if len(bbox) >= 2 else 0.4
        circles.append(
            Circle(
                name=name,
                kind=str(obj.get("type", "obstacle")),
                xy=xy(obj.get("position", [0, 0])),
                radius=asset_radius + args.rover_radius + args.safety_margin,
            )
        )
    for fence in data.get("fences", []) or []:
        a, b = xy(fence["start"]), xy(fence["end"])
        segments.append(Segment(str(fence.get("name", "fence")), a, b, args.fence_radius + 0.7 * args.rover_radius))
        spacing = float(fence.get("spacing", 0.0) or 0.0)
        if spacing > 1e-6:
            length = float(np.linalg.norm(b - a))
            direction = unit(b - a)
            for i, dist in enumerate(np.arange(0.0, length + 0.5 * spacing, spacing)):
                circles.append(
                    Circle(
                        f"{fence.get('name', 'fence')}_post_{i}",
                        "fence_post",
                        a + direction * min(float(dist), length),
                        args.post_radius + 0.7 * args.rover_radius,
                    )
                )
    return circles, segments


def infer_start_goal(data: dict[str, Any], args: argparse.Namespace) -> tuple[np.ndarray, float, np.ndarray, dict[str, Any]]:
    rover = data.get("rover_pose", {})
    start = xy(rover.get("position", [0, 0]))
    start_yaw = float(rover.get("yaw", 0.0))
    if args.goal_x is not None and args.goal_y is not None:
        return start, start_yaw, np.asarray([args.goal_x, args.goal_y], dtype=np.float32), {"goal_mode": "explicit"}
    fences = data.get("fences", []) or []
    if not fences:
        raise ValueError("No fence found; pass --goal-x and --goal-y.")
    fence = fences[0]
    a, b = xy(fence["start"]), xy(fence["end"])
    direction = unit(b - a)
    left = np.asarray([-direction[1], direction[0]], dtype=np.float32)
    length = float(np.linalg.norm(b - a))
    lateral = float(np.dot(start - a, left))
    side = 1.0 if lateral >= 0.0 else -1.0
    if args.side == "left":
        side = 1.0
    elif args.side == "right":
        side = -1.0
    offset = max(abs(lateral) if args.fence_offset is None else args.fence_offset, args.rover_radius + args.safety_margin)
    projection = float(np.dot(start - a, direction))
    if args.goal_end == "start":
        endpoint = 0.0
        mode = "fence_start"
    elif args.goal_end == "end" or abs(projection) <= abs(projection - length):
        endpoint = length
        mode = "fence_end" if args.goal_end == "end" else "auto_far_end"
    else:
        endpoint = 0.0
        mode = "auto_far_start"
    goal = a + direction * endpoint + left * side * float(offset)
    return start, start_yaw, goal, {"goal_mode": mode, "fence_offset_m": float(offset), "fence_side_sign": side}


def make_grid(start: np.ndarray, goal: np.ndarray, circles: list[Circle], segments: list[Segment], args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs, ys = [start[0], goal[0]], [start[1], goal[1]]
    for c in circles:
        xs += [c.xy[0] - c.radius, c.xy[0] + c.radius]
        ys += [c.xy[1] - c.radius, c.xy[1] + c.radius]
    for s in segments:
        xs += [s.a[0], s.b[0]]
        ys += [s.a[1], s.b[1]]
    x = np.arange(min(xs) - args.bounds_margin, max(xs) + args.bounds_margin, args.grid_resolution, dtype=np.float32)
    y = np.arange(min(ys) - args.bounds_margin, max(ys) + args.bounds_margin, args.grid_resolution, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    return x, y, np.stack([xx, yy], axis=-1)


def costmap(points: np.ndarray, circles: list[Circle], segments: list[Segment], rng: np.random.Generator, variant: int, prior_paths: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    blocked = np.zeros(points.shape[:2], dtype=bool)
    cost = np.ones(points.shape[:2], dtype=np.float32)
    for c in circles:
        d = np.linalg.norm(points - c.xy, axis=-1)
        blocked |= d <= c.radius
        cost += 2.5 * np.exp(-np.maximum(d - c.radius, 0.0) / 0.35)
    for s in segments:
        d = seg_dist(points, s.a, s.b)
        blocked |= d <= s.radius
        cost += 2.0 * np.exp(-np.maximum(d - s.radius, 0.0) / 0.30)
    if variant > 0:
        coarse = rng.normal(size=(8, 8)).astype(np.float32)
        yi = np.linspace(0, 7, points.shape[0]).astype(np.int32)
        xi = np.linspace(0, 7, points.shape[1]).astype(np.int32)
        noise = coarse[yi[:, None], xi[None, :]]
        cost += 0.18 * variant * (noise - noise.min()) / (np.ptp(noise) + 1e-6)
    for path in prior_paths:
        if len(path) < 2:
            continue
        nearest = np.full(points.shape[:2], np.inf, dtype=np.float32)
        for a, b in zip(path[:-1], path[1:]):
            nearest = np.minimum(nearest, seg_dist(points, a, b))
        cost += 1.4 * np.exp(-nearest / 0.45)
    return blocked, cost


def nearest_idx(point: np.ndarray, x: np.ndarray, y: np.ndarray) -> tuple[int, int]:
    return int(np.argmin(abs(y - point[1]))), int(np.argmin(abs(x - point[0])))


def astar(blocked: np.ndarray, cost: np.ndarray, start: tuple[int, int], goal: tuple[int, int], res: float) -> tuple[list[tuple[int, int]], float]:
    if blocked[start]:
        raise ValueError("Start is inside inflated obstacle space.")
    if blocked[goal]:
        raise ValueError("Goal is inside inflated obstacle space.")
    nbrs = [(-1, 0, 1), (1, 0, 1), (0, -1, 1), (0, 1, 1), (-1, -1, 2**0.5), (-1, 1, 2**0.5), (1, -1, 2**0.5), (1, 1, 2**0.5)]
    open_heap = [(0.0, 0.0, start)]
    came: dict[tuple[int, int], tuple[int, int]] = {}
    g = {start: 0.0}
    h, w = blocked.shape
    while open_heap:
        _, current_g, cur = heapq.heappop(open_heap)
        if cur == goal:
            path = [cur]
            while cur in came:
                cur = came[cur]
                path.append(cur)
            return path[::-1], current_g
        if current_g > g.get(cur, float("inf")):
            continue
        cy, cx = cur
        for dy, dx, mult in nbrs:
            ny, nx = cy + dy, cx + dx
            if ny < 0 or nx < 0 or ny >= h or nx >= w or blocked[ny, nx]:
                continue
            ng = current_g + res * mult * 0.5 * (float(cost[cy, cx]) + float(cost[ny, nx]))
            nxt = (ny, nx)
            if ng < g.get(nxt, float("inf")):
                came[nxt] = cur
                g[nxt] = ng
                heapq.heappush(open_heap, (ng + res * math.hypot(ny - goal[0], nx - goal[1]), ng, nxt))
    raise ValueError("No collision-free path found.")


def resample(points: np.ndarray, step: float) -> np.ndarray:
    d = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(d)])
    total = float(cumulative[-1])
    samples = np.arange(0, total, step, dtype=np.float32)
    samples = np.concatenate([samples, np.asarray([total], dtype=np.float32)])
    out = []
    for s in samples:
        i = min(max(int(np.searchsorted(cumulative, float(s), side="right") - 1), 0), len(points) - 2)
        t = (float(s) - float(cumulative[i])) / max(float(d[i]), 1e-6)
        out.append(points[i] * (1 - t) + points[i + 1] * t)
    return np.asarray(out, dtype=np.float32)


def yaws_from_positions(positions: np.ndarray, start_yaw: float) -> np.ndarray:
    if len(positions) < 2:
        return np.asarray([start_yaw], dtype=np.float32)
    grad = np.gradient(positions, axis=0)
    return np.asarray(wrap_to_pi(np.arctan2(grad[:, 1], grad[:, 0])), dtype=np.float32)


def plan(data: dict[str, Any], path: Path, args: argparse.Namespace, variant: int, rng: np.random.Generator, prior_paths: list[np.ndarray]) -> tuple[dict[str, Any], list[Circle], list[Segment]]:
    start, start_yaw, goal, meta = infer_start_goal(data, args)
    circles, segments = obstacles(data, args)
    x, y, points = make_grid(start, goal, circles, segments, args)
    blocked, cost = costmap(points, circles, segments, rng, variant, prior_paths)
    grid_path, planner_cost = astar(blocked, cost, nearest_idx(start, x, y), nearest_idx(goal, x, y), args.grid_resolution)
    raw = np.asarray([[x[ix], y[iy]] for iy, ix in grid_path], dtype=np.float32)
    raw[0], raw[-1] = start, goal
    positions = resample(raw, args.path_step)
    yaws = yaws_from_positions(positions, start_yaw)
    distances = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(positions, axis=0), axis=1))])
    times = distances / max(args.speed, 1e-3)
    velocity = estimate_velocity(times, positions, yaws)
    valid, waypoints = local_waypoints(positions, yaws, args.horizon, args.target_spacing_m, args.min_final_distance, args.max_final_distance)
    poses = np.column_stack([positions[:, 0], positions[:, 1], yaws]).astype(np.float32)
    traj_id = f"{path.stem}_traj{variant:03d}"
    return {
        "trajectory_id": traj_id,
        "times": times[valid],
        "position": positions[valid],
        "yaw": yaws[valid],
        "velocity": velocity[valid],
        "poses": poses[valid],
        "robot_state": np.column_stack([positions[valid], yaws[valid], velocity[valid]]).astype(np.float32),
        "waypoints": waypoints,
        "valid_indices": valid,
        "full_times": times,
        "full_position": positions,
        "full_yaw": yaws,
        "full_velocity": velocity,
        "full_poses": poses,
        "source_yaml": str(path),
        "metadata": {**meta, "start": start.tolist(), "goal": goal.tolist(), "planner_cost": float(planner_cost)},
    }, circles, segments


def save_npz(out_dir: Path, args: argparse.Namespace, traj: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    image_paths = np.asarray([f"sim://{traj['trajectory_id']}/{int(i):06d}" for i in traj["valid_indices"]], dtype=str)
    np.savez_compressed(
        out_dir / "trajectory.npz",
        image_paths=image_paths,
        times=traj["times"],
        position=traj["position"].astype(np.float32),
        yaw=traj["yaw"].astype(np.float32),
        velocity=traj["velocity"].astype(np.float32),
        poses=traj["poses"].astype(np.float32),
        global_poses=traj["poses"].astype(np.float32),
        robot_state=traj["robot_state"].astype(np.float32),
        target_waypoints=traj["waypoints"].astype(np.float32),
        waypoints=traj["waypoints"].astype(np.float32),
        valid_indices=traj["valid_indices"].astype(np.int64),
        full_times=traj["full_times"],
        full_position=traj["full_position"].astype(np.float32),
        full_yaw=traj["full_yaw"].astype(np.float32),
        full_velocity=traj["full_velocity"].astype(np.float32),
        full_poses=traj["full_poses"].astype(np.float32),
        dataset_name=np.asarray(args.dataset_name),
        trajectory_id=np.asarray(traj["trajectory_id"]),
        trajectory_name=np.asarray(traj["trajectory_id"]),
        source_yaml=np.asarray(traj["source_yaml"]),
    )
    with (out_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump({**traj["metadata"], "num_training_samples": int(len(traj["valid_indices"]))}, f, indent=2)


def plot_scene(path: Path, trajectories: list[dict[str, Any]], circles: list[Circle], segments: list[Segment]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    for s in segments:
        ax.plot([s.a[0], s.b[0]], [s.a[1], s.b[1]], color="black", linewidth=2, label="fence")
    for c in circles:
        color = "0.75" if c.kind == "fence_post" else "tab:red"
        alpha = 0.45 if c.kind == "fence_post" else 0.25
        ax.add_patch(plt.Circle(c.xy, c.radius, color=color, alpha=alpha, linewidth=0))
        if c.kind != "fence_post":
            ax.scatter([c.xy[0]], [c.xy[1]], c="tab:red", s=18)
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, max(1, len(trajectories))))
    for i, t in enumerate(trajectories):
        p = t["full_position"]
        ax.plot(p[:, 0], p[:, 1], color=colors[i], linewidth=2, label=t["trajectory_id"])
        ax.scatter([p[0, 0]], [p[0, 1]], c=[colors[i]], marker="o")
        ax.scatter([p[-1, 0]], [p[-1, 1]], c=[colors[i]], marker="x")
    ax.set_xlabel("world x (m)")
    ax.set_ylabel("world y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def yaml_files(path: Path) -> list[Path]:
    return [path] if path.is_file() else sorted([*path.glob("*.yaml"), *path.glob("*.yml")])


def main() -> None:
    args = parse_args()
    files = yaml_files(args.input)
    if not files:
        raise SystemExit(f"No YAML files found at {args.input}")
    args.out_root.mkdir(parents=True, exist_ok=True)
    index = []
    for file in files:
        data = load_yaml(file)
        rng = np.random.default_rng(int(data.get("seed", 0)) + args.seed)
        scenario_out = args.out_root / file.stem
        if scenario_out.exists() and not args.overwrite:
            raise SystemExit(f"{scenario_out} exists; pass --overwrite.")
        trajectories: list[dict[str, Any]] = []
        prior_paths: list[np.ndarray] = []
        circles: list[Circle] = []
        segments: list[Segment] = []
        for variant in range(args.num_trajectories):
            traj, circles, segments = plan(data, file, args, variant, rng, prior_paths)
            save_npz(scenario_out / traj["trajectory_id"], args, traj)
            trajectories.append(traj)
            prior_paths.append(traj["full_position"])
            index.append(
                {
                    "source_yaml": str(file),
                    "trajectory_id": traj["trajectory_id"],
                    "trajectory_npz": str(scenario_out / traj["trajectory_id"] / "trajectory.npz"),
                    "num_training_samples": int(len(traj["valid_indices"])),
                }
            )
        if args.plot:
            plot_scene(scenario_out / "scene_trajectories.png", trajectories, circles, segments)
    with (args.out_root / "index.json").open("w", encoding="utf-8") as f:
        json.dump({"dataset_name": args.dataset_name, "trajectories": index}, f, indent=2)
    print(f"Wrote {len(index)} trajectories under {args.out_root}")


if __name__ == "__main__":
    main()
