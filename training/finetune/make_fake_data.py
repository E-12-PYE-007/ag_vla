"""
Generates 30 pre-batched .pt files for pipeline testing.

Run:
    PROJECT_ID=pd98 python make_fake_data.py

Output: /data/gpfs/projects/${PROJECT_ID}/fake_data/batch_NNN.pt
        (falls back to ./fake_data/ if PROJECT_ID is not set)
"""

import os
from pathlib import Path
import torch

BATCH_SIZE  = 4
SEQ_LEN     = 50   # prompt + action tokens
NUM_BATCHES = 30

project_id = os.environ.get("PROJECT_ID", "")
if project_id:
    DATA_DIR = Path(f"/data/gpfs/projects/{project_id}/fake_data")
else:
    DATA_DIR = Path(__file__).parent / "fake_data"

DATA_DIR.mkdir(parents=True, exist_ok=True)

for i in range(NUM_BATCHES):
    # Alternate modality: 7 = language-only, 6 = image-only
    # lan_bool in run_forward_pass is True for 7 and 8
    goal_mask = torch.tensor([7, 6, 7, 6])

    batch = {
        "input_ids":            torch.randint(0, 32000, (BATCH_SIZE, SEQ_LEN)),
        "attention_mask":       torch.ones(BATCH_SIZE, SEQ_LEN, dtype=torch.bool),
        "attention_mask_label": torch.ones(BATCH_SIZE, SEQ_LEN, dtype=torch.bool),
        "labels":               torch.randint(0, 32000, (BATCH_SIZE, SEQ_LEN)),
        "pixel_values":         torch.randn(BATCH_SIZE, 6, 224, 224),  # 2 images × 3 channels
        "goal_pose":            torch.randn(BATCH_SIZE, 4),             # (x, y, cosθ, sinθ)
        "obj_pose_norm":        torch.randn(BATCH_SIZE, 2),             # normalised goal (x, y)
        "actions":              torch.randn(BATCH_SIZE, 8, 4),          # (len_traj_pred=8, 4)
        "c_image":              torch.rand(BATCH_SIZE, 3, 96, 96),      # [0,1] before _IMG_NORM
        "p_image":              torch.rand(BATCH_SIZE, 3, 96, 96),
        "goal_mask_select":     goal_mask,
    }

    path = DATA_DIR / f"batch_{i:03d}.pt"
    torch.save(batch, path)
    print(f"Saved {path}")

print(f"\nDone — {NUM_BATCHES} batches in {DATA_DIR}")
