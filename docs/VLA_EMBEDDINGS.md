# AsyncVLA Embedding Extraction

The AsyncVLA repository has been placed under:

```text
external/AsyncVLA-main/
```

The project-local extractor is:

```text
scripts/embeddings/extract_vla_embeddings.py
```

It reads processed SCAND/RECON trajectories and writes a training-ready NPZ with
AsyncVLA projected action embeddings:

```text
processed_mixed/<dataset>/<trajectory>/
    trajectory.npz
    trajectory_with_embeddings.npz
    embedding_metadata.json
```

The output includes:

```python
projected_actions  # [T, 8, 1024]
action_embeddings  # alias of projected_actions
target_waypoints   # [T, 8, 3]
robot_state        # [T, 2], copied from velocity
velocity           # [T, 2]
image_paths        # [T]
```

## Dependencies

Install the normal conversion requirements first:

```bash
python -m pip install -r requirements.txt
```

Then install AsyncVLA following `external/AsyncVLA-main/SETUP.md`. AsyncVLA uses
a custom Transformers fork and several navigation-model dependencies, so keep
that environment separate from the lightweight dataset-conversion environment if
needed.

The AsyncVLA README also says the release checkpoint should be placed in or
near the AsyncVLA directory:

```bash
git clone https://huggingface.co/NHirose/AsyncVLA_release
```

You need the VLA model directory and the action projector checkpoint, usually:

```text
AsyncVLA_release/
    action_proj--750000_checkpoint.pt
```

If your release contains a different projector step, pass
`--action-proj-checkpoint` explicitly.

By default, the extractor matches `scripts/embeddings/run_vla.py` for image handling:

```text
--num-images-in-input 1
--image-copies 2
```

That means each saved dataset image is duplicated into two `pixel_values` slots,
while the VLA patch offset is computed the same way as the live VLA-only script.
Keep this default unless you intentionally change the AsyncVLA modality setup.

## Extract One Trajectory

```bash
python scripts/embeddings/extract_vla_embeddings.py \
    --npz processed_mixed/scand/example/trajectory.npz \
    --asyncvla-root external/AsyncVLA-main \
    --vla-path external/AsyncVLA-main/AsyncVLA_release \
    --prompt "Continue safe navigation while avoiding obstacles." \
    --batch-size 1
```

## Extract All Processed SCAND/RECON Trajectories

```bash
python scripts/embeddings/extract_vla_embeddings.py \
    --processed-root processed_mixed \
    --asyncvla-root external/AsyncVLA-main \
    --vla-path external/AsyncVLA-main/AsyncVLA_release \
    --prompt "Continue safe navigation while avoiding obstacles." \
    --batch-size 1
```

If your projector checkpoint is not named with the default `--resume-step`:

```bash
python scripts/embeddings/extract_vla_embeddings.py \
    --processed-root processed_mixed \
    --asyncvla-root external/AsyncVLA-main \
    --vla-path E:/models/AsyncVLA_release \
    --action-proj-checkpoint E:/models/AsyncVLA_release/action_proj--845000_checkpoint.pt
```

## Training After Extraction

Point the flow-head trainer at files containing embeddings:

```bash
python scripts/training/train_flow_head.py \
    --data processed_mixed/scand/example/trajectory_with_embeddings.npz \
    --output-dir checkpoints/fenceline_flow_head \
    --epochs 100 \
    --batch-size 64
```

Waypoint normalization is enabled by default during training. The sampler
unnormalizes before applying physical metre/radian clamps.
