# Simulation Data Generation

This folder is for generating our own simulated expert navigation trajectories
from scene YAML files. It is separate from `scripts/data_processing`, which is
for converting real public datasets such as SCAND, RECON, HuRoN, GO Stanford 2,
and BotanicGarden.

The current generator creates expert fenceline trajectories from YAML scene
configs containing:

```text
rover_pose
fences
obstacles
assets
```

Example input:

```text
examples/fence01_seed43_roverstart_right.yaml
```

## Script

```text
generate_fenceline_expert_trajectories.py
```

The script builds a 2D planning problem from the YAML scene:

- the rover start pose comes from `rover_pose`;
- fence wires and posts become obstacles;
- plants, logs, boulders, and other objects become circular obstacles using
  their configured asset bounding boxes;
- an A* grid planner finds a collision-free path to the selected fenceline goal;
- the path is resampled into smooth metric poses;
- future local-frame waypoint chunks are generated for training.

## Output Format

The output matches the same trajectory format used by the real datasets:

```text
<out-root>/<yaml-stem>/<trajectory-id>/
  trajectory.npz
  metadata.json
```

The `.npz` contains:

```text
image_paths          [T] synthetic sim:// identifiers
times                [T]
position             [T, 2]
yaw                  [T]
velocity             [T, 2]
robot_state          [T, 5]  [x, y, yaw, forward_speed, yaw_rate]
target_waypoints     [T, 8, 3]
waypoints            [T, 8, 3]
full_position        [N, 2]
full_yaw             [N]
full_velocity        [N, 2]
dataset_name
trajectory_id
trajectory_name
source_yaml
```

`target_waypoints` uses the same convention as the rest of the project:

```text
[x_forward_m, y_left_m, yaw_ccw_rad]
```

The generated `image_paths` are `sim://...` placeholders. They do not point to
real rendered camera images. Use this data for waypoint/action-head logic only
unless a later rendering step is added and aligned to these poses.

## Basic Command

From the repository root:

```powershell
python .\scripts\simulation_data_generation\generate_fenceline_expert_trajectories.py `
  --input .\scripts\simulation_data_generation\examples\fence01_seed43_roverstart_right.yaml `
  --out-root "D:\Capstone\processed_mixed\fenceline_sim" `
  --rover-radius 0.22 `
  --safety-margin 0.08 `
  --plot `
  --overwrite
```

This writes:

```text
D:\Capstone\processed_mixed\fenceline_sim\fence01_seed43_roverstart_right\
  fence01_seed43_roverstart_right_traj000\
    trajectory.npz
    metadata.json
  scene_trajectories.png
index.json
```

## Generate Multiple Expert Variants

Use `--num-trajectories` to produce multiple valid paths through the same scene.
The planner adds variation so paths do not all collapse to the same route.

```powershell
python .\scripts\simulation_data_generation\generate_fenceline_expert_trajectories.py `
  --input "D:\Capstone\sim_configs\fenceline" `
  --out-root "D:\Capstone\processed_mixed\fenceline_sim" `
  --num-trajectories 5 `
  --rover-radius 0.22 `
  --safety-margin 0.08 `
  --plot `
  --overwrite
```

`--input` can be either one YAML file or a directory containing `.yaml`/`.yml`
files.

## Important Arguments

```text
--input                 YAML file or directory of YAML files
--out-root              output root for generated trajectories
--num-trajectories      number of expert paths per YAML scene
--dataset-name          saved dataset name, default fenceline_sim
--goal-x, --goal-y      manually override the goal
--goal-end              auto, start, or end of the fence
--side                  auto, left, or right side of the fence
--fence-offset          desired lateral offset from the fence
--rover-radius          rover collision radius in metres
--safety-margin         additional obstacle clearance
--grid-resolution       A* grid resolution in metres
--path-step             output path resampling step in metres
--speed                 assumed traversal speed for timestamps
--horizon               waypoint horizon, default 8
--target-spacing-m      distance spacing between future waypoints
--plot                  save scene_trajectories.png
--overwrite             replace existing generated output
```

## Validation

After generation, inspect the produced trajectory with the normal checker:

```powershell
python .\scripts\data_processing\check_processed_trajectory.py `
  --npz "D:\Capstone\processed_mixed\fenceline_sim\fence01_seed43_roverstart_right\fence01_seed43_roverstart_right_traj000\trajectory.npz"
```

Plot a waypoint chunk:

```powershell
python .\scripts\data_processing\plot_waypoint_chunk.py `
  --npz "D:\Capstone\processed_mixed\fenceline_sim\fence01_seed43_roverstart_right\fence01_seed43_roverstart_right_traj000\trajectory.npz" `
  --index 0 `
  --save "D:\Capstone\processed_mixed\fenceline_sim\fence01_seed43_roverstart_right\waypoint_debug_0.png"
```

## Training Note

These trajectories do not currently have real images, so they cannot be passed
through the frozen VLA image backbone in the same way as SCAND/RECON/HuRoN.
They are useful for testing waypoint-label generation, action-head behavior, or
future simulation rendering workflows. If rendered camera images are later
generated for the same poses, they can be integrated into the normal embedding
and training pipeline.
