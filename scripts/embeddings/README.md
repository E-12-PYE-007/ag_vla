# Embedding Scripts

These scripts run a frozen OmniVLA/AsyncVLA-style backbone and save embeddings
beside each processed trajectory.

The current preferred checkpoint is:

```text
NHirose/omnivla-original-balance
```

The extractor is run on RCP

## Output

For each trajectory:

trajectory.npz
trajectory_with_embeddings.npz

The current training path uses:

raw_action_embeddings [T, 32, 4096]

Projected `action_embeddings [T, 8, 1024]` are intentionally not required,
because the projector is trained jointly with our action head.

## Smoke Test One Trajectory

```bash
ONE_NPZ=$(find "$HOME/capstone_data/processed_mixed" -name trajectory.npz | head -n 1)

python scripts/embeddings/extract_vla_embeddings.py \
  --npz "$ONE_NPZ" \
  --asyncvla-root "$HOME/asyncvla-test/AsyncVLA" \
  --vla-path "$HOME/asyncvla-test/AsyncVLA/omnivla-original-balance" \
  --raw-only \
  --max-samples 1 \
  --batch-size 1 \
  --overwrite
```

Expected keys:

```
raw_action_embeddings [1, 32, 4096]
target_waypoints      [1, 8, 3]
robot_state           [1, 2]
```

## Extract Full Processed Root on RCP

```bash
cd ~/action_head

python scripts/embeddings/extract_vla_embeddings.py \
  --processed-root "$HOME/capstone_data/processed_mixed" \
  --asyncvla-root "$HOME/asyncvla-test/AsyncVLA" \
  --vla-path "$HOME/asyncvla-test/AsyncVLA/omnivla-original-balance" \
  --raw-only \
  --prompt "navigate forward safely" \
  --batch-size 1
```

Use one consistent prompt for unlabeled navigation datasets. If GO Stanford 2 or
LeLaN language labels are later matched reliably, extract those samples with
their real instruction.

## Manifest Utility

Create a list of trajectories missing embeddings:

```bash
python scripts/embeddings/make_embedding_manifest.py \
  --processed-root "$HOME/capstone_data/processed_mixed" \
  --out "$HOME/capstone_data/missing_embeddings.txt" \
  --missing-only
```

If extraction is complete, this should report `0 trajectories`.
