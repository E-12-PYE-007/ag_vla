"""
Convert raw episode directories into per-sample .pt files for finetune.py.

For each frame k with NUM_WAYPOINTS future frames, one .pt file is written:
  actions  = robot poses at frames k+1..k+NUM_WAYPOINTS in frame k's ego frame (metres)
  c_image  = frame k (current image for the edge adapter)
  p_image / pixel_values = frame j = k - lt, with lt sampled from 0..MAX_DELAY_FRAMES (stale VLA image)
"""

import json
import random
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TVF
from PIL import Image
from torchvision import transforms

NUM_WAYPOINTS    = 8
MAX_DELAY_FRAMES = 3

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


def _load_episode_meta(episode_dir: Path):
    """Load instruction and ordered image list from whichever format is present."""
    manifest_path = episode_dir / "training_manifest.jsonl"
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        return manifest["instruction"], manifest["images"]  # images already "img/XXXX.jpg"

    # New format: metadata.json + postprocessed_samples.jsonl
    meta_path     = episode_dir / "metadata.json"
    samples_path  = episode_dir / "postprocessed_samples.jsonl"
    if not meta_path.exists() or not samples_path.exists():
        return None, None

    with open(meta_path) as f:
        instruction = json.load(f)["language_instruction"]

    samples = []
    with open(samples_path) as f:
        for line in f:
            samples.append(json.loads(line))
    samples.sort(key=lambda s: s["sample_index"])
    image_names = [f"img/{s['image']}" for s in samples]

    return instruction, image_names


def _load_poses(poses_path: Path) -> tuple[list[str], np.ndarray]:
    """Per-frame image path and robot pose [x, y, yaw] (world frame) from poses.jsonl."""
    image_names, poses = [], []
    with open(poses_path) as f:
        for line in f:
            row = json.loads(line)
            _, x, y, yaw = row["pose"]  # [timestamp, x, y, yaw]
            image_names.append(f"img/{row['image']}")
            poses.append([x, y, yaw])
    return image_names, np.array(poses, dtype=np.float64)


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


def process_episode(episode_dir: Path, out_dir: Path, episode_id: str | None = None) -> tuple[int, np.ndarray]:
    """Write samples for one episode. Returns (samples written, per-frame step distances in metres).

    episode_id prefixes every filename (default: folder name) and must be unique across the dataset.
    """
    episode_id = episode_id or episode_dir.name
    no_steps = np.empty(0)

    instruction, image_names = _load_episode_meta(episode_dir)
    if instruction is None:
        print(f"  [skip] no manifest or metadata found in {episode_dir.name}")
        return 0, no_steps

    poses_path = episode_dir / "poses.jsonl"
    if not poses_path.exists():
        print(f"  [skip] poses.jsonl not found in {episode_dir.name}")
        return 0, no_steps

    pose_image_names, poses = _load_poses(poses_path)
    if pose_image_names != image_names:
        print(f"  [skip] poses.jsonl images do not match postprocessed_samples order in {episode_dir.name}")
        return 0, no_steps

    saved = 0
    for k in range(len(poses) - NUM_WAYPOINTS):
        lt = random.randint(0, min(k, MAX_DELAY_FRAMES))
        j  = k - lt

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

        torch.save(sample, out_dir / f"{episode_id}__j{j:04d}_k{k:04d}_lt{lt}.pt")
        saved += 1

    steps = np.linalg.norm(np.diff(poses[:, :2], axis=0), axis=1)
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
