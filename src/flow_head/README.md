# `flow_head` Source Package

This package contains the reusable Python code behind the AG-VLA action-head
pipeline. The command-line scripts in `scripts/` are thin wrappers around these
modules.

The package covers three main jobs:

```text
1. Convert raw navigation data into trajectory.npz files.
2. Load embedded trajectories for training/evaluation.
3. Define and evaluate waypoint action heads.
```

## Training Data Contract

Most modules assume the shared processed trajectory format:

```text
trajectory.npz
  image_paths          [T]
  times                [T]
  position             [T, 2]
  yaw                  [T]
  velocity             [T, 2]
  target_waypoints     [T, 8, 3]
  dataset_name
  trajectory_name
```

After RCP embedding extraction, training normally reads:

```text
trajectory_with_embeddings.npz
  raw_action_embeddings [T, 32, 4096]
  target_waypoints      [T, 8, 3]
  robot_state or velocity
```

Waypoint convention everywhere is:

```text
[x_forward_m, y_left_m, yaw_ccw_rad]
```

## Core Training Modules

### `dataset.py`

Defines `TrajectoryEmbeddingDataset`, the dataset used by MLP/flow training and
evaluation scripts.

Responsibilities:

- find `trajectory_with_embeddings.npz` or `trajectory.npz` files;
- load one timestep at a time;
- keep embeddings, robot state, labels, and metadata aligned;
- support single-file, directory, or mixed-root training inputs;
- apply train-set normalization statistics.

This is the main bridge between processed data and PyTorch.

### `asyncvla_projector.py`

Implements the AsyncVLA-style action-token projector:

```text
raw_action_embeddings [B, 32, 4096]
  -> Proj_Actiontokens
  -> projected context tokens [B, 8, 1024]
```

Training scripts use this when called with:

```text
--use-asyncvla-projector --train-projector
```

That means the projector is trained jointly with the action head instead of
using AsyncVLA's released projector checkpoint.

### `mlp_waypoint_head.py`

Defines the deterministic MLP baseline:

```text
MLPWaypointHead
MLPWaypointHeadConfig
```

It flattens context tokens, optionally appends robot state/modality ID, and
predicts the full waypoint chunk directly:

```text
[B, context_tokens, context_dim] -> [B, 8, 3]
```

This is the simplest supervised baseline.

### `flow_waypoint_head.py`

Defines the current flow-matching waypoint head:

```text
FlowWaypointHead
FlowWaypointHeadConfig
```

It learns a vector field in waypoint space. During sampling it starts from noise
and iteratively flows toward a waypoint chunk conditioned on the VLA context and
robot state.

It also defines `clamp_waypoints`, which constrains sampled predictions to safe
physical limits after unnormalization.

### `model.py`

Contains an older/alternate flow-matching implementation:

```text
FlowMatchingWaypointHead
FlowMatchingWaypointHeadConfig
```

Most current training scripts use `flow_waypoint_head.py`; keep this module only
for compatibility with earlier experiments unless a script explicitly imports
it.

### `evaluation.py`

Shared metric and output helpers:

- L1
- RMSE
- ADE
- FDE
- yaw MAE
- final-yaw MAE
- `metrics.json`
- `predictions.npz`

Used by `scripts/training/evaluate_mlp_head.py` and
`scripts/training/evaluate_flow_head.py`.

### `splits.py`

Helpers for fixed train/val/test splits.

Splits are trajectory-level, not frame-level, to prevent leakage from adjacent
frames of the same trajectory.

## Dataset Conversion Modules

### `public_nav_conversion.py`

Shared conversion utilities used by multiple public datasets.

Includes:

- yaw wrapping and quaternion/rotation helpers;
- nearest timestamp matching;
- local waypoint chunk generation;
- distance-spaced waypoint chunk generation;
- velocity estimation from pose;
- trajectory validation;
- image linking/copying;
- `save_processed_trajectory`.

This is the common toolbox for dataset converters.

### `scand_conversion.py`

SCAND Jackal ROS bag support.

Includes:

- pure-Python ROS bag topic inspection using `rosbags`;
- compressed/raw image decoding;
- odometry extraction;
- image/odom synchronization;
- future waypoint chunk generation;
- writing `trajectory.npz`, `metadata.json`, and extracted images.

Used by:

```text
scripts/data_processing/inspect_scand_bag.py
scripts/data_processing/convert_scand_bag.py
scripts/data_processing/batch_convert_scand.py
```

### `recon_loader.py`

RECON HDF5 loader helpers.

The conversion script uses this to inspect and load HDF5 arrays before applying
sample-level filtering and waypoint conversion.

Used by:

```text
scripts/data_processing/convert_recon.py
scripts/data_processing/select_recon_subset.py
```

### `huron_conversion.py`

HuRoN/SACSoN ROS bag conversion support.

Important project-specific choice:

```text
velocity is derived from pose deltas, not raw twist fields
```

because the tested HuRoN twist values were not plausible for training.

Used by:

```text
scripts/data_processing/convert_huron_bag.py
```

### `botanic_garden_conversion.py`

BotanicGarden conversion support for either ROS bag data or file-based
timestamped images plus TUM-style trajectories.

Important caveat:

BotanicGarden ground truth may be in the Velodyne frame, so frame transforms
must be verified before training labels are trusted.

Used by:

```text
scripts/data_processing/convert_botanic_garden.py
```

### `waypoint_labels.py`

Small waypoint-label helper functions for converting future global poses into
local robot-frame waypoint chunks.

This module is useful when implementing another converter and wanting to reuse
the same label convention.

## Import Examples

Load training data:

```python
from flow_head.dataset import TrajectoryEmbeddingDataset

dataset = TrajectoryEmbeddingDataset("/path/to/processed_mixed")
sample = dataset[0]
```

Create the MLP head:

```python
from flow_head.mlp_waypoint_head import MLPWaypointHead, MLPWaypointHeadConfig

model = MLPWaypointHead(MLPWaypointHeadConfig(context_dim=1024))
```

Create the flow head:

```python
from flow_head.flow_waypoint_head import FlowWaypointHead, FlowWaypointHeadConfig

model = FlowWaypointHead(FlowWaypointHeadConfig(context_dim=1024))
```

Use the projector:

```python
from flow_head.asyncvla_projector import Proj_Actiontokens

projector = Proj_Actiontokens()
```

## Where To Add New Code

- New dataset converter helpers: add a dedicated `*_conversion.py` module here
  and a command-line wrapper under `scripts/data_processing/`.
- New action-head architecture: add a new model module here and a training
  wrapper under `scripts/training/`.
- New shared metric: add it to `evaluation.py`.
- New split behavior: add it to `splits.py`.

Keep scripts thin. Put reusable logic in this package.
