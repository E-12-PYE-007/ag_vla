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

# Action chunk dimensions (must match AsyncVLA constants)
NUM_ACTIONS_CHUNK = 8
ACTION_DIM        = 4
N_ACTION          = NUM_ACTIONS_CHUNK * ACTION_DIM  # 32 action tokens per sample
N_LANG            = SEQ_LEN - N_ACTION              # 18 language tokens per sample

project_id = os.environ.get("PROJECT_ID", "")
if project_id:
    DATA_DIR = Path(f"/data/gpfs/projects/{project_id}/fake_data")
else:
    DATA_DIR = Path(__file__).parent / "fake_data"

DATA_DIR.mkdir(parents=True, exist_ok=True)

for i in range(NUM_BATCHES):
    # Action token IDs must be > ACTION_TOKEN_BEGIN_IDX (31743) — both
    # get_current_action_mask and get_next_actions_mask check token_id > 31743.
    action_token_ids = torch.randint(31744, 32000, (BATCH_SIZE, N_ACTION))

    # Language token IDs kept below 31744 so they are never mistaken for action tokens
    input_ids = torch.cat([
        torch.randint(1, 31744, (BATCH_SIZE, N_LANG)),
        action_token_ids,
    ], dim=1)  # (B, SEQ_LEN)

    # labels: IGNORE_INDEX (-100) for language positions, real token IDs for action positions.
    # This ensures _process_action_masks() finds exactly N_ACTION tokens per sample,
    # keeping the per-sample count uniform so the model's reshape([B, -1, D]) succeeds.
    labels = torch.full((BATCH_SIZE, SEQ_LEN), -100, dtype=torch.long)
    labels[:, N_LANG:] = action_token_ids

    # attention_mask_label: True only at action token positions
    attention_mask_label = torch.zeros(BATCH_SIZE, SEQ_LEN, dtype=torch.bool)
    attention_mask_label[:, N_LANG:] = True

    batch = {
        "input_ids":            input_ids,
        "attention_mask":       torch.ones(BATCH_SIZE, SEQ_LEN, dtype=torch.bool),
        "attention_mask_label": attention_mask_label,
        "labels":               labels,
        "pixel_values":         torch.randn(BATCH_SIZE, 6, 224, 224),  # 2 images × 3 channels
        "obj_pose_norm":        torch.randn(BATCH_SIZE, 2),             # normalised goal (x, y)
        "goal_pose":            torch.zeros(BATCH_SIZE, 4),             # dummy — masked for modality 7, values irrelevant
        "actions":              torch.randn(BATCH_SIZE, 8, 4),          # (len_traj_pred=8, 4)
        "c_image":              torch.rand(BATCH_SIZE, 3, 96, 96),      # [0,1] before _IMG_NORM
        "p_image":              torch.rand(BATCH_SIZE, 3, 96, 96),
        "goal_mask_select":     torch.full((BATCH_SIZE,), 7),           # all language-conditioned
    }

    path = DATA_DIR / f"batch_{i:03d}.pt"
    torch.save(batch, path)
    print(f"Saved {path}")

print(f"\nDone — {NUM_BATCHES} batches in {DATA_DIR}")
