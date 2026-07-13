# Data Processing Scripts

These scripts turn raw public navigation data into the shared AG-VLA trajectory
format:

```text
processed_mixed/<dataset>/<trajectory>/
  images/
  trajectory.npz
  metadata.json
```

The important label is:

```text
target_waypoints [T, 8, 3]
```

where each waypoint is `[x_forward_m, y_left_m, yaw_ccw_rad]` in the robot's
local frame.

This folder is for real public navigation datasets. Simulation scene YAML files,
such as fenceline scene configs, should live in a separate simulation-data
generation folder once the full generator script is added.

## SCAND

Inspect one bag:

```powershell
python .\scripts\data_processing\inspect_scand_bag.py `
  --bag "D:\Capstone\A_Jackal_AHG_Library_Thu_Nov_4_16.bag" `
  --processed-root "D:\Capstone\processed_mixed\scand"
```

Batch convert all SCAND Jackal bags:

```powershell
python .\scripts\data_processing\batch_convert_scand.py `
  --bags-dir "D:\Capstone" `
  --out-root "D:\Capstone\processed_mixed\scand" `
  --horizon 8 `
  --waypoint-dt 0.5 `
  --image-stride 5 `
  --sync-threshold 0.10 `
  --max-final-distance 10.0
```

## RECON

Select a subset:

```powershell
python .\scripts\data_processing\select_recon_subset.py `
  --input-root "D:\Capstone\raw_recon\recon_release" `
  --out-json "D:\Capstone\configs\recon_clean_10h.json" `
  --target-hours 10
```

Convert selected RECON HDF5 trajectories:

```powershell
python .\scripts\data_processing\convert_recon.py `
  --input-root "D:\Capstone\raw_recon\recon_release" `
  --selection-json "D:\Capstone\configs\recon_clean_10h.json" `
  --out-root "D:\Capstone\processed_mixed\recon"
```

## HuRoN

Download a balanced subset from WSL:

```bash
cd /mnt/c/Users/miahv/Documents/Capstone_Project/ag_vla

python3 scripts/data_processing/download_huron_balanced_subset_wsl.py \
  --out-root /mnt/d/Capstone/vla_datasets/huron \
  --mode balanced \
  --max-bags-per-folder 10 \
  --max-total-gb 25 \
  --min-bag-mb 30
```

Convert one validated bag:

```powershell
python .\scripts\data_processing\convert_huron_bag.py `
  --bag "D:\Capstone\vla_datasets\huron\raw\Feb-15-2023-cory1\00000000.bag" `
  --out-dir "D:\Capstone\processed_mixed\huron\Feb-15-2023-cory1_00000000" `
  --image-topic "/fisheye_image/compressed" `
  --odom-topic "/odometry" `
  --image-stride 5 `
  --max-pose-time-error 0.20
```

Batch convert downloaded HuRoN bags:

```powershell
python .\scripts\data_processing\batch_convert_huron.py `
  --raw-root "D:\Capstone\vla_datasets\huron\raw" `
  --out-root "D:\Capstone\processed_mixed\huron" `
  --image-topic "/fisheye_image/compressed" `
  --odom-topic "/odometry" `
  --image-stride 5
```

Some downloaded bags may contain images but no useful odometry motion. Those
bags are skipped and recorded in `batch_summary.json`.

Current processed result:

```text
D:\Capstone\processed_mixed\huron
111 trajectories in the mixed index
32,537 samples
6 failed bags with no valid motion/waypoint samples
```

## GO Stanford 2

Inspect the extracted archive:

```powershell
python .\scripts\data_processing\inspect_go_stanford.py `
  --root "D:\Capstone\vla_datasets\go_stanford_2\raw\gs2_withres" `
  --out-json "D:\Capstone\vla_datasets\go_stanford_2\inspection_summary.json"
```

Convert non-flipped sequences:

```powershell
python .\scripts\data_processing\convert_go_stanford.py `
  --root "D:\Capstone\vla_datasets\go_stanford_2\raw\gs2_withres" `
  --out-root "D:\Capstone\processed_mixed\go_stanford_2" `
  --side auto `
  --frame-dt 0.2 `
  --sampling-mode distance `
  --target-spacing-m 0.25 `
  --image-stride 5
```

The GO Stanford 2 converter integrates per-frame result pickles as assumed
`[linear_velocity, angular_velocity]` commands. This is useful for a first
training pass, but the exact frame rate and command semantics should be treated
as dataset assumptions.

Current processed result:

```text
D:\Capstone\processed_mixed\go_stanford_2
49 processed non-flipped trajectories
19,390 samples
25 failed sequences due to incomplete extracted image/result folders
```

## BotanicGarden

BotanicGarden is deferred until a pilot bag is downloaded and frame transforms
are checked.

Inspect a bag:

```powershell
python .\scripts\data_processing\inspect_botanic_bag.py `
  --bag "D:\Capstone\vla_datasets\botanic_garden\raw\1018-00\<file>.bag" `
  --out-json "D:\Capstone\vla_datasets\botanic_garden\extracted\1018-00\metadata\bag_summary.json"
```

Convert only after inspection:

```powershell
python .\scripts\data_processing\convert_botanic_garden.py `
  --bag "D:\Capstone\vla_datasets\botanic_garden\raw\1018-00\<file>.bag" `
  --out-dir "D:\Capstone\processed_mixed\botanic_garden\1018-00" `
  --image-topic "/dalsa_rgb/left/image_raw" `
  --pose-topic "/gt_poses" `
  --image-stride 10 `
  --max-pose-time-error 0.05
```

## Validation and Indexing

Check one converted trajectory:

```powershell
python .\scripts\data_processing\check_processed_trajectory.py `
  --npz "D:\Capstone\processed_mixed\scand\<trajectory>\trajectory.npz"
```

Build the mixed index:

```powershell
python .\scripts\data_processing\build_mixed_dataset_index.py `
  --processed-root "D:\Capstone\processed_mixed" `
  --out "D:\Capstone\processed_mixed\mixed_index.json"
```

Summarize the index:

```powershell
python .\scripts\data_processing\summarize_mixed_index.py `
  --index "D:\Capstone\processed_mixed\mixed_index.json"
```

Current mixed index:

```text
632 trajectories
97,057 total samples

go_stanford_2: 19,390 samples, 49 trajectories, 20.0%
huron:         32,537 samples, 111 trajectories, 33.5%
recon:         25,028 samples, 448 trajectories, 25.8%
scand:         20,102 samples, 24 trajectories, 20.7%
```

Plot waypoint labels:

```powershell
python .\scripts\data_processing\plot_waypoint_chunk.py `
  --npz "D:\Capstone\processed_mixed\scand\<trajectory>\trajectory.npz" `
  --index 20 `
  --save "D:\Capstone\processed_mixed\scand\<trajectory>\waypoint_debug_20.png"
```
