# Dataset Preparation Notes

This document explains why each dataset is included, what raw information is
used, and how it is converted into the shared training format.

All datasets are converted to:

```text
processed_mixed/<dataset>/<trajectory>/
  images/
  trajectory.npz
  metadata.json
```

The key supervised label is:

```text
target_waypoints [T, 8, 3]
```

where each waypoint is:

```text
[x_forward_m, y_left_m, yaw_ccw_rad]
```

The horizon is normally 8 waypoints. The conversion scripts express future
poses in the current robot local frame, so the action head can learn a common
metric command representation across datasets.

## Current Mixed Dataset Summary

After adding GO Stanford 2 and HuRoN, the current mixed index is:

```text
D:\Capstone\processed_mixed\mixed_index.json

632 trajectories
97,057 total samples

go_stanford_2: 19,390 samples, 49 trajectories, 20.0%
huron:         32,537 samples, 111 trajectories, 33.5%
recon:         25,028 samples, 448 trajectories, 25.8%
scand:         20,102 samples, 24 trajectories, 20.7%
```

## SCAND

### Why Use It

SCAND provides real Jackal robot navigation bags with synchronized camera and
odometry streams. It is a strong base dataset because the robot platform is
close to the target navigation setting, the ROS bag structure is reliable, and
the trajectories contain real indoor/outdoor motion rather than only synthetic
or curated examples.

### Raw Data Used

The current plan uses all available SCAND Jackal bags downloaded to the external
drive. Each bag is inspected because the image topic can differ across files.

Typical useful topics:

```text
/left/image_color/compressed
/jackal_velocity_controller/odom
```

### Processing

For each image timestamp, the converter finds a nearby odometry pose, samples
future odometry poses at fixed waypoint intervals, transforms those future
poses into the current robot frame, and writes:

```text
image_paths
times
position
yaw
velocity
target_waypoints
dataset_name = scand
trajectory_name
```

Settings used for the first validated conversion:

```text
horizon:            8
waypoint_dt:        0.5 s
image_stride:       5
sync_threshold:     0.10 s
max_final_distance: 10.0 m
```

`max_final_distance` is intentionally higher than the default because some
SCAND Jackal files move close to 8 m over the 4 s horizon.

### Commands

Inspect:

```powershell
python .\scripts\data_processing\inspect_scand_bag.py `
  --bag "D:\Capstone\A_Jackal_AHG_Library_Thu_Nov_4_16.bag" `
  --processed-root "D:\Capstone\processed_mixed\scand"
```

Batch convert:

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

Validate:

```powershell
python .\scripts\data_processing\check_processed_trajectory.py `
  --npz "D:\Capstone\processed_mixed\scand\<trajectory>\trajectory.npz"
```

## RECON

### Why Use It

RECON adds broad outdoor and semi-structured navigation diversity: sidewalks,
parks, parking lots, grass, and noisy real-world robot behavior. This helps
avoid a SCAND-only model that overfits to one robot collection style.

### Raw Data Used

RECON is stored as HDF5 trajectories. The project currently uses a filtered
subset rather than blindly training on every frame.

### Processing

Initial whole-trajectory rejection was too aggressive because many RECON files
contain at least one collision or failure event. The converter was changed to
filter at the sample/window level instead:

```text
convert trajectory
drop samples near physical/stuck/flipped collision events where labels are bad
keep valid samples from the rest of the file
```

This preserves useful RECON diversity while avoiding bad supervision around
stuck, flipped, or physically invalid motion.

The converter writes short but valid waypoint chunks. Short trajectories are
acceptable because each image plus future waypoint chunk is one supervised
sample. Splits must still be trajectory-level to avoid leakage.

### Commands

Select:

```powershell
python .\scripts\data_processing\select_recon_subset.py `
  --input-root "D:\Capstone\raw_recon\recon_release" `
  --out-json "D:\Capstone\configs\recon_clean_10h.json" `
  --target-hours 10
```

Convert:

```powershell
python .\scripts\data_processing\convert_recon.py `
  --input-root "D:\Capstone\raw_recon\recon_release" `
  --selection-json "D:\Capstone\configs\recon_clean_10h.json" `
  --out-root "D:\Capstone\processed_mixed\recon"
```

## HuRoN / SACSoN

### Why Use It

HuRoN adds indoor social-navigation and policy-collected robot bags. It is
smaller and easier to iterate with than BotanicGarden, and it introduces data
from standard and interaction-loss collection conditions.

### Raw Data Used

Official source:

```text
https://rail.eecs.berkeley.edu/datasets/huron/
```

The balanced downloader selects a capped subset across multiple date/policy
folders instead of downloading the full release. This reduces imbalance against
SCAND/RECON while still adding meaningful diversity.

The pilot found:

```text
Feb-15-2023-cory1/00000000.bag                 usable
Feb-16-2023-cory1-intloss/00000000.bag         images but zero odometry motion
Feb-16-2023-cory1-intloss/00000001.bag         usable replacement
```

Typical topics:

```text
/fisheye_image/compressed
/odometry
```

### Processing

The HuRoN converter uses fisheye compressed images and odometry poses. Raw
twist values in the tested bags were not plausible for training, so velocity is
derived from pose deltas:

```text
forward_speed = local delta translation / dt
yaw_rate      = wrapped yaw delta / dt
```

Future poses are transformed to current local robot coordinates to produce the
same `[x_forward_m, y_left_m, yaw_ccw_rad]` waypoint labels as SCAND and RECON.

### Processed Result

HuRoN was batch converted from:

```text
D:\Capstone\vla_datasets\huron\raw
```

to:

```text
D:\Capstone\processed_mixed\huron
```

Current result:

```text
111 trajectories in the mixed index
32,537 samples
6 bags failed because they produced no valid motion/waypoint samples
```

Known failed bags:

```text
Feb-16-2023-cory1-intloss/00000000.bag
Feb-16-2023-cory1-intloss/00000003.bag
Feb-16-2023-cory1-intloss/00000005.bag
Feb-16-2023-cory1-intloss/00000006.bag
Feb-16-2023-cory1-intloss/00000009.bag
Feb-16-2023-cory1-intloss/00000010.bag
```

### Commands

Download balanced subset from WSL:

```bash
cd /mnt/c/Users/miahv/Documents/Capstone_Project/ag_vla

python3 scripts/data_processing/download_huron_balanced_subset_wsl.py \
  --out-root /mnt/d/Capstone/vla_datasets/huron \
  --mode balanced \
  --max-bags-per-folder 10 \
  --max-total-gb 25 \
  --min-bag-mb 30
```

Inspect:

```powershell
python .\scripts\data_processing\inspect_huron_bag.py `
  --bag "D:\Capstone\vla_datasets\huron\raw\Feb-15-2023-cory1\00000000.bag" `
  --out-json "D:\Capstone\vla_datasets\huron\extracted\Feb-15-2023-cory1\metadata\bag_summary.json"
```

Convert:

```powershell
python .\scripts\data_processing\convert_huron_bag.py `
  --bag "D:\Capstone\vla_datasets\huron\raw\Feb-15-2023-cory1\00000000.bag" `
  --out-dir "D:\Capstone\processed_mixed\huron\Feb-15-2023-cory1_00000000" `
  --image-topic "/fisheye_image/compressed" `
  --odom-topic "/odometry" `
  --image-stride 5 `
  --max-pose-time-error 0.20
```

## GO Stanford 2

### Why Use It

GO Stanford 2 can add large-scale fisheye visual navigation diversity across
buildings. It may also become valuable if later matched to LeLaN-style language
instructions.

### Raw Data Used

The downloaded archive was:

```text
D:\Capstone\gs2_withres.tar.xz
```

It was extracted to:

```text
D:\Capstone\vla_datasets\go_stanford_2\raw\gs2_withres
```

The extracted structure contains matching sequence list files:

```text
dataset_L_<building>pre_<sequence>.txt
dataset_R_<building>pre_<sequence>.txt
dataset_refres_<building>pre_<sequence>.txt
img_L_<building>pre/
img_R_<building>pre/
res_<building>pre/
```

The `dataset_L/R` files list image paths. The `dataset_refres` files list
per-frame pickle files under `res_*`. Each result pickle contains a 2-value
array that is treated as:

```text
[linear_velocity, angular_velocity]
```

Flipped sequences marked with `F` are excluded for the first training pass.

### Processing

GO Stanford 2 is not ROS-bag based and does not provide direct metric poses in
the extracted files. The converter therefore:

1. selects non-flipped sequences;
2. pairs an image list with the matching `dataset_refres` list;
3. loads each result pickle as assumed `[v, omega]`;
4. integrates those commands with `frame_dt = 0.2 s`;
5. generates local distance-spaced waypoint chunks;
6. saves the shared `trajectory.npz` format.

Important caveat:

```text
GO Stanford 2 poses are integrated from velocity-command pickles.
The exact frame rate and command semantics are assumptions in this first
converter pass.
```

### Processed Result

Current result:

```text
49 processed non-flipped trajectories
19,390 samples
25 non-flipped sequences failed because the extracted archive was incomplete
for those result/image folders
```

Examples of failures:

```text
missing res_10pre/*.pickle
missing res_14pre/*.pickle
missing some img_R_13pre / img_L_20pre / img_L_8pre folders
```

The successful trajectories are included in:

```text
D:\Capstone\processed_mixed\go_stanford_2
```

### Commands

Inspect:

```powershell
python .\scripts\data_processing\inspect_go_stanford.py `
  --root "D:\Capstone\vla_datasets\go_stanford_2\raw\gs2_withres" `
  --out-json "D:\Capstone\vla_datasets\go_stanford_2\inspection_summary.json"
```

Convert:

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

## BotanicGarden

### Why Use It

BotanicGarden would add natural outdoor unstructured navigation, calibrated RGB
stereo, LiDAR, and high-quality ground truth. It is a good future diversity
source, but the files are large, so it is deferred behind HuRoN and GO Stanford
2.

### Raw Data Used

Official repo:

```text
https://github.com/robot-pesg/BotanicGarden
```

Pilot target:

```text
1018-00 VLIO bag
```

Known issue: even the smallest pilot is about 13 GB.

Likely topics:

```text
/dalsa_rgb/left/image_raw
/dalsa_rgb/right/image_raw
/gt_poses
```

### Processing

The converter can read ROS bag image and pose topics and produce the shared
trajectory format. However, BotanicGarden ground truth tracks the Velodyne VLP16
frame. Before training, the transform chain from Velodyne to robot base/camera
must be verified so labels are not produced in the wrong frame.

### Commands

Inspect:

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

## Mixed Dataset Build

After all selected datasets are converted:

```powershell
python .\scripts\data_processing\build_mixed_dataset_index.py `
  --processed-root "D:\Capstone\processed_mixed" `
  --out "D:\Capstone\processed_mixed\mixed_index.json"

python .\scripts\data_processing\summarize_mixed_index.py `
  --index "D:\Capstone\processed_mixed\mixed_index.json"
```

After embeddings exist, build a fixed trajectory-level split:

```powershell
python .\scripts\training\build_train_val_test_split.py `
  --data-root "D:\Capstone\processed_mixed" `
  --out "D:\Capstone\processed_mixed\splits\train_val_test.json" `
  --train-fraction 0.70 `
  --val-fraction 0.15 `
  --test-fraction 0.15 `
  --seed 7
```
