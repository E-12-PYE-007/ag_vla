# Training Scripts

These scripts consume embedded trajectories and train/evaluate action heads.

The expected input is:

```text
processed_mixed/<dataset>/<trajectory>/trajectory_with_embeddings.npz
```

with:

```text
raw_action_embeddings [T, 32, 4096]
target_waypoints      [T, 8, 3]
robot_state           [T, 2]
```

Because the embeddings are raw OmniVLA action-token states, training should use:

```text
--use-asyncvla-projector --train-projector
```

This trains the projector jointly with the MLP or flow-matching head.

## Build Fixed Split

```bash
python scripts/training/build_train_val_test_split.py \
  --data-root /path/to/processed_mixed \
  --out /path/to/processed_mixed/splits/train_val_test.json \
  --train-fraction 0.70 \
  --val-fraction 0.15 \
  --test-fraction 0.15 \
  --seed 7
```

Splits are trajectory-level, not frame-level.

## Smoke Test

```bash
EMB=/path/to/trajectory_with_embeddings.npz

python scripts/training/train_mlp_head.py \
  --data "$EMB" \
  --output-dir /tmp/agvla_mlp_smoke \
  --use-asyncvla-projector \
  --train-projector \
  --epochs 1 \
  --batch-size 1 \
  --num-workers 0 \
  --val-fraction 0

python scripts/training/train_flow_head.py \
  --data "$EMB" \
  --output-dir /tmp/agvla_flow_smoke \
  --use-asyncvla-projector \
  --train-projector \
  --epochs 1 \
  --batch-size 1 \
  --num-workers 0 \
  --val-fraction 0
```

## Train Full Dataset

```bash
python scripts/training/train_mlp_head.py \
  --data /path/to/processed_mixed \
  --split-json /path/to/processed_mixed/splits/train_val_test.json \
  --output-dir checkpoints/mlp_head_raw_projector \
  --use-asyncvla-projector \
  --train-projector \
  --default-modality-id 7.0

python scripts/training/train_flow_head.py \
  --data /path/to/processed_mixed \
  --split-json /path/to/processed_mixed/splits/train_val_test.json \
  --output-dir checkpoints/flow_head_raw_projector \
  --use-asyncvla-projector \
  --train-projector \
  --default-modality-id 7.0
```

Each output directory contains:

```text
best.pt
last.pt
config.json
normalization_stats.json
```

## Evaluate

```bash
python scripts/training/evaluate_mlp_head.py \
  --checkpoint checkpoints/mlp_head_raw_projector/best.pt \
  --data /path/to/processed_mixed \
  --split-json /path/to/processed_mixed/splits/train_val_test.json \
  --split test \
  --output-dir eval/mlp_head_test

python scripts/training/evaluate_flow_head.py \
  --checkpoint checkpoints/flow_head_raw_projector/best.pt \
  --data /path/to/processed_mixed \
  --split-json /path/to/processed_mixed/splits/train_val_test.json \
  --split test \
  --num-steps 20 \
  --output-dir eval/flow_head_test
```

Metrics include L1, RMSE, ADE, FDE, yaw MAE, and final-yaw MAE.

Plot predictions:

```bash
python scripts/training/plot_head_predictions.py \
  --predictions eval/flow_head_test/predictions.npz \
  --out-dir eval/flow_head_test/plots \
  --num-plots 32
```

## Spartan

Submit from the repo root on Spartan:

```bash
sbatch --export=ALL,PROJECT_ROOT=/data/gpfs/projects/<projectID>/ag_vla \
  scripts/training/spartan_build_split.slurm

sbatch --export=ALL,PROJECT_ROOT=/data/gpfs/projects/<projectID>/ag_vla \
  scripts/training/spartan_train_mlp.slurm

sbatch --export=ALL,PROJECT_ROOT=/data/gpfs/projects/<projectID>/ag_vla \
  scripts/training/spartan_train_flow.slurm

sbatch --export=ALL,PROJECT_ROOT=/data/gpfs/projects/<projectID>/ag_vla \
  scripts/training/spartan_evaluate_heads.slurm
```

See `docs/RCP_TO_SPARTAN_TRAINING.md` for the full RCP-to-Spartan workflow.
