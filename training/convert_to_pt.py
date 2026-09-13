"""
Convert raw episode directories into per-sample .pt files for finetune.py.

For each frame k with NUM_WAYPOINTS clean future frames, one .pt file is written:
  actions  = robot poses at frames k+1..k+NUM_WAYPOINTS in frame k's ego frame (metres)
  c_image  = frame k (current image for the edge adapter)
  p_image / pixel_values = frame j nearest to (time of k - delay), delay drawn from DELAYS_S (stale VLA image)

Frames are sorted by image time with repeated images dropped. Frames whose waypoints or delay window
cross a physically implausible pose change (a simulator pose glitch) are not used.
"""

import json
import random
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TVF
from PIL import Image
from torchvision import transforms

NUM_WAYPOINTS = 8
DELAYS_S      = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]

# Pose changes faster than this between consecutive frames are glitches (robot limits: 0.3 m/s, 0.45 rad/s)
MAX_SPEED_MPS    = 1.0
MAX_YAW_RATE_RPS = 1.0

# ── Prismatic image transform (from AsyncVLA preprocessor_config.json) ────────
# Fused DinoV2 + SigLIP backbone: apply_transform returns (6, 224, 224) for
# a single PIL image — DinoV2-normalised channels stacked on SigLIP channels.
_DINO_MEAN = [0.484375,    0.455078125, 0.40625      ]
_DINO_STD  = [0.228515625, 0.2236328125, 0.224609375 ]
_SIGLIP_MEAN = [0.5, 0.5, 0.5]
_SIGLIP_STD  = [0.5, 0.5, 0.5]

def apply_transform(pil_img: Image.Image) -> torch.Tensor:
    """Replicates PrismaticImageProcessor.apply_transform for AsyncVLA."""
    img = pil_img.convert("RGB")
    img_r = TVF.resize(img, [224, 224], interpolation=TVF.InterpolationMode.BICUBIC, antialias=True)
    img_r = TVF.center_crop(img_r, [224, 224])
    t = TVF.to_tensor(img_r)
    dino   = TVF.normalize(t, mean=_DINO_MEAN,   std=_DINO_STD)    # (3, 224, 224)
    siglip = TVF.normalize(t, mean=_SIGLIP_MEAN, std=_SIGLIP_STD)  # (3, 224, 224)
    return torch.cat([dino, siglip], dim=0)  # (6, 224, 224)

# ── 96×96 transform for Edge_adapter inputs ───────────────────────────────────
_to_tensor_96 = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor(),  # → [0, 1]
])


def _load_instruction(episode_dir: Path):
    """Language instruction from whichever metadata format is present, or None."""
    manifest_path = episode_dir / "training_manifest.jsonl"
    if manifest_path.exists():
        with open(manifest_path) as f:
            return json.load(f)["instruction"]

    meta_path = episode_dir / "metadata.json"
    if not meta_path.exists():
        return None
    with open(meta_path) as f:
        return json.load(f)["language_instruction"]


def _load_poses(poses_path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Per-frame image path, image time (s) and world pose [x, y, yaw] from poses.jsonl.

    Rows are sorted by image time and rows repeating an earlier image are dropped.
    """
    rows = []
    with open(poses_path) as f:
        for line in f:
            row = json.loads(line)
            _, x, y, yaw = row["pose"]  # [timestamp, x, y, yaw]
            rows.append((row["img_time"], f"img/{row['image']}", x, y, yaw))
    rows.sort(key=lambda r: r[0])

    seen, image_names, times, poses = set(), [], [], []
    for t, name, x, y, yaw in rows:
        if name in seen:
            continue
        seen.add(name)
        image_names.append(name)
        times.append(t)
        poses.append([x, y, yaw])
    return image_names, np.array(times, dtype=np.float64), np.array(poses, dtype=np.float64)


def _bad_transitions(times: np.ndarray, poses: np.ndarray) -> np.ndarray:
    """(N-1,) bool: True where the change from frame i to i+1 is physically implausible."""
    dt   = np.diff(times)
    dist = np.linalg.norm(np.diff(poses[:, :2], axis=0), axis=1)
    dyaw = np.abs((np.diff(poses[:, 2]) + np.pi) % (2 * np.pi) - np.pi)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (dt <= 0) | (dist / dt > MAX_SPEED_MPS) | (dyaw / dt > MAX_YAW_RATE_RPS)


def _future_waypoints(poses: np.ndarray, k: int) -> np.ndarray:
    """Poses at frames k+1..k+NUM_WAYPOINTS in frame k's ego frame: (NUM_WAYPOINTS, 4) [x, y, cosθ, sinθ]."""
    x0, y0, yaw0 = poses[k]
    future = poses[k + 1 : k + 1 + NUM_WAYPOINTS]
    dx, dy = future[:, 0] - x0, future[:, 1] - y0
    c, s = np.cos(yaw0), np.sin(yaw0)
    dyaw = future[:, 2] - yaw0
    return np.stack(
        [c * dx + s * dy, -s * dx + c * dy, np.cos(dyaw), np.sin(dyaw)], axis=1
    ).astype(np.float32)


def _feasible_delays(times: np.ndarray, bad: np.ndarray, k: int) -> list[tuple[float, int]]:
    """(delay, stale frame j) pairs that stay inside the episode with no glitch between j and k."""
    feasible = []
    for delay in DELAYS_S:
        target = times[k] - delay
        if target < times[0]:
            continue
        j = int(np.argmin(np.abs(times[: k + 1] - target)))
        if not bad[j:k].any():
            feasible.append((delay, j))
    return feasible


def process_episode(episode_dir: Path, out_dir: Path, episode_id: str | None = None) -> tuple[int, np.ndarray]:
    """Write samples for one episode. Returns (samples written, clean per-frame step distances in metres).

    episode_id prefixes every filename (default: folder name) and must be unique across the dataset.
    """
    episode_id = episode_id or episode_dir.name
    no_steps = np.empty(0)

    instruction = _load_instruction(episode_dir)
    if instruction is None:
        print(f"  [skip] no manifest or metadata found in {episode_dir.name}")
        return 0, no_steps

    poses_path = episode_dir / "poses.jsonl"
    if not poses_path.exists():
        print(f"  [skip] poses.jsonl not found in {episode_dir.name}")
        return 0, no_steps

    image_names, times, poses = _load_poses(poses_path)
    missing = [name for name in image_names if not (episode_dir / name).exists()]
    if missing:
        print(f"  [skip] {len(missing)} images listed in poses.jsonl are missing in {episode_dir.name}")
        return 0, no_steps

    bad = _bad_transitions(times, poses)

    saved = 0
    for k in range(len(poses) - NUM_WAYPOINTS):
        if bad[k : k + NUM_WAYPOINTS].any():
            continue  # waypoints cross a pose glitch
        delay, j = random.choice(_feasible_delays(times, bad, k))

        try:
            p_pil = Image.open(episode_dir / image_names[j]).convert("RGB")
            c_pil = Image.open(episode_dir / image_names[k]).convert("RGB")
        except (OSError, SyntaxError):
            continue  # skip samples with unreadable images

        sample = {
            "instruction":  instruction,
            "pixel_values": apply_transform(p_pil),                          # (6, 224, 224) stale frame for the VLA
            "c_image":      _to_tensor_96(c_pil),                            # (3, 96, 96)  current frame for Edge_adapter
            "p_image":      _to_tensor_96(p_pil),                            # (3, 96, 96)  stale frame for Edge_adapter
            "actions":      torch.from_numpy(_future_waypoints(poses, k)),   # (8, 4) metres, ego frame of frame k
        }

        torch.save(sample, out_dir / f"{episode_id}__j{j:04d}_k{k:04d}_d{round(delay * 10):03d}.pt")
        saved += 1

    steps = np.linalg.norm(np.diff(poses[:, :2], axis=0), axis=1)[~bad]
    return saved, steps


# ── Configure paths here before running ──────────────────────────────────────
EPISODES_DIR = Path("./")         # searched recursively for episode folders containing poses.jsonl
OUT_DIR      = Path("./pt_data")  # where to write .pt files
SEED         = 42
# ─────────────────────────────────────────────────────────────────────────────


def main():
    random.seed(SEED)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    episode_dirs = sorted(p.parent for p in EPISODES_DIR.rglob("poses.jsonl"))
    if not episode_dirs:
        raise RuntimeError(f"No episode folders with poses.jsonl found under {EPISODES_DIR}")

    total, all_steps, seen_names = 0, [], set()
    for i, ep_dir in enumerate(episode_dirs):
        # The same folder name in two batches is a different recording, so keep filenames unique
        episode_id = ep_dir.name
        if ep_dir.name in seen_names:
            episode_id = f"{ep_dir.name}--{ep_dir.relative_to(EPISODES_DIR).parts[0]}"
        seen_names.add(ep_dir.name)

        print(f"[{i+1}/{len(episode_dirs)}] {episode_id}")
        n, steps = process_episode(ep_dir, OUT_DIR, episode_id)
        print(f"  → {n} samples")
        total += n
        if n:
            all_steps.append(steps)

    print(f"\nDone — {total} samples written to {OUT_DIR}")
    print(f"Mean per-frame step (waypoint spacing S): {np.concatenate(all_steps).mean():.4f} m")


if __name__ == "__main__":
    main()
