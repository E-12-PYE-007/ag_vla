# AG-VLA Action Head Training

This branch contains the data preparation and action-head training pipeline for
predicting local robot-frame waypoint chunks from frozen OmniVLA action-token
embeddings.

The intended model path is:

```text
image + navigation prompt
  -> frozen OmniVLA navigation backbone
  -> raw_action_embeddings [T, 32, 4096]
  -> trainable projector + action head
  -> target_waypoints [T, 8, 3]
```

`target_waypoints` are metric local-frame labels:

```text
[x_forward_m, y_left_m, yaw_ccw_rad]
```

## Main Documents

```text
docs/DATASETS.md
  Dataset justification, processing choices, expected raw data, and conversion
  commands for SCAND, RECON, HuRoN, GO Stanford 2, and BotanicGarden.

docs/VLA_EMBEDDINGS.md
  How frozen OmniVLA embeddings are extracted on RCP.

docs/RCP_TO_SPARTAN_TRAINING.md
  How embedded data moves from RCP to Spartan, and how to train/evaluate with
  Slurm.

scripts/README.md
  Script folder layout.
```

## Repository Layout

```text
src/flow_head/
  Core Python modules: dataset loading, converters, models, projector, metrics,
  and split helpers.

scripts/data_processing/
  Download, inspect, convert, validate, and index dataset files.

scripts/embeddings/
  Frozen VLA embedding extraction and embedding manifests.

scripts/training/
  Train/evaluate/sample heads, make fixed splits, and submit Spartan jobs.

docs/
  Handoff documentation for datasets, embeddings, and remote training.
```

## Data Policy

Do not commit raw bags, downloaded archives, extracted images, `.npz` data,
VLA checkpoints, action-head checkpoints, or evaluation outputs.

Local/external-drive data has usually lived under:

```text
D:\Capstone
```

RCP data has usually lived under:

```text
~/capstone_data/processed_mixed
```

Spartan training data should live under project storage:

```text
/data/gpfs/projects/<projectID>/ag_vla/data/processed_mixed
```

## Processed Trajectory Format

Each converted trajectory directory contains:

```text
images/
trajectory.npz
metadata.json
```

Required `trajectory.npz` keys:

```text
image_paths        [T]
times              [T]
position           [T, 2]
yaw                [T]
velocity           [T, 2]      [forward_speed, yaw_rate]
target_waypoints   [T, 8, 3]
dataset_name       scalar string
trajectory_name    scalar string
```

After RCP embedding extraction, each trajectory should also have:

```text
trajectory_with_embeddings.npz
raw_action_embeddings [T, 32, 4096]
```

## Current Processed Dataset

The current `D:\Capstone\processed_mixed\mixed_index.json` contains:

```text
632 trajectories
97,057 total samples

go_stanford_2: 19,390 samples, 49 trajectories, 20.0%
huron:         32,537 samples, 111 trajectories, 33.5%
recon:         25,028 samples, 448 trajectories, 25.8%
scand:         20,102 samples, 24 trajectories, 20.7%
```

This is the dataset state to copy to RCP for VLA embedding extraction.

## Typical Workflow

1. Convert public navigation datasets into `processed_mixed/<dataset>/<traj>/`.
2. Build/check the mixed index and validate waypoint statistics.
3. Extract `raw_action_embeddings` on RCP with frozen OmniVLA.
4. Copy embedded `processed_mixed` to Spartan project storage.
5. Build a fixed trajectory-level train/val/test split.
6. Train the MLP and flow-matching heads with a jointly trained projector.
7. Evaluate on the held-out test split and inspect prediction plots.

Start with:

```text
docs/DATASETS.md
scripts/data_processing/README.md
docs/RCP_TO_SPARTAN_TRAINING.md
scripts/training/README.md
```
