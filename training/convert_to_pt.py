"""
Convert raw episode directories into per-sample .pt files for finetune.py.

Each anchor frame j gets one .pt file.  The delay between p_image (frame j)
and c_image (frame k ≈ t_j + delay) is sampled uniformly from DELAYS.
"""

import json
import random
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TVF
from PIL import Image
from torchvision import transforms

DELAYS = [0.2, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]

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


def find_nearest_frame(timestamps: np.ndarray, target_t: float) -> int:
    """Return index of the frame closest to target_t, or -1 if target_t is past episode end."""
    if target_t > timestamps[-1]:
        return -1
    return int(np.argmin(np.abs(timestamps - target_t)))


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


def _load_ego_actions(poses_path: Path) -> list:
    """
    Load per-frame ego-centric action chunks from poses.jsonl.

    Returns a list of length N where each entry is either:
        np.ndarray (8, 4)  [x_m, y_m, cos(yaw), sin(yaw)] in robot ego frame
        None               if the frame has no valid action_chunk
    """
    result = []
    with open(poses_path) as f:
        for line in f:
            frame = json.loads(line)
            try:
                rp = frame["action_chunk"]["relative_poses"]
                if len(rp) != 8:
                    result.append(None)
                    continue
                waypoints = np.array(
                    [[x, y, np.cos(yaw), np.sin(yaw)] for x, y, yaw in rp],
                    dtype=np.float32,
                )
                result.append(waypoints)
            except (KeyError, TypeError, ValueError):
                result.append(None)
    return result


def process_episode(episode_dir: Path, out_dir: Path) -> int:
    instruction, image_names = _load_episode_meta(episode_dir)
    if instruction is None:
        print(f"  [skip] no manifest or metadata found in {episode_dir.name}")
        return 0

    poses_path = episode_dir / "poses.jsonl"
    ts_path    = episode_dir / "timestamps.npy"

    if not poses_path.exists():
        print(f"  [skip] poses.jsonl not found in {episode_dir.name}")
        return 0

    timestamps = np.load(ts_path)      # (N,)
    N          = len(timestamps)

    if len(image_names) != N:
        print(f"  [skip] image count {len(image_names)} != timestamps {N} in {episode_dir.name}")
        return 0

    # Ego-centric actions from poses.jsonl: (N,) list of (8,4) arrays or None
    actions = _load_ego_actions(poses_path)
    if len(actions) != N:
        print(f"  [skip] poses count {len(actions)} != timestamps {N} in {episode_dir.name}")
        return 0

    saved      = 0
    episode_id = episode_dir.name

    for j in range(N):
        delay    = random.choice(DELAYS)
        target_t = timestamps[j] + delay
        k        = find_nearest_frame(timestamps, target_t)

        if k == -1:
            k = find_nearest_frame(timestamps, timestamps[j] + delay / 2)
            if k == -1:
                break

        if actions[k] is None:
            continue  # skip frames with missing action_chunk

        try:
            p_pil = Image.open(episode_dir / image_names[j]).convert("RGB")
            c_pil = Image.open(episode_dir / image_names[k]).convert("RGB")
        except (OSError, SyntaxError):
            continue  # skip samples with unreadable images

        sample = {
            "instruction":  instruction,
            "pixel_values": apply_transform(p_pil),              # (6, 224, 224) fused DinoV2+SigLIP
            "c_image":      _to_tensor_96(c_pil),                # (3, 96, 96)  fresh frame for Edge_adapter
            "p_image":      _to_tensor_96(p_pil),                # (3, 96, 96)  stale frame for Edge_adapter
            "actions":      torch.from_numpy(actions[k]).float(), # (8, 4) ego-centric [x_m, y_m, cosθ, sinθ]
        }

        delay_tag = f"{int(delay * 10):03d}"
        fname = f"{episode_id}__j{j:04d}_k{k:04d}_d{delay_tag}.pt"
        torch.save(sample, out_dir / fname)
        saved += 1

    return saved


# ── Configure paths here before running ──────────────────────────────────────
EPISODES_DIR = Path("./")         # directory whose subdirs are episode folders
OUT_DIR      = Path("./pt_data")  # where to write .pt files
SEED         = 42
# ─────────────────────────────────────────────────────────────────────────────


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    episode_dirs = sorted(d for d in EPISODES_DIR.iterdir() if d.is_dir())
    if not episode_dirs:
        raise RuntimeError(f"No subdirectories found in {EPISODES_DIR}")

    total = 0
    for i, ep_dir in enumerate(episode_dirs):
        print(f"[{i+1}/{len(episode_dirs)}] {ep_dir.name}")
        n = process_episode(ep_dir, OUT_DIR)
        print(f"  → {n} samples")
        total += n

    print(f"\nDone — {total} samples written to {OUT_DIR}")


if __name__ == "__main__":
    main()
