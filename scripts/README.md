# Scripts

Scripts are grouped by the stage of the pipeline they belong to.

```text
scripts/data_processing/
  Download, inspect, convert, validate, and index public navigation datasets.

scripts/embeddings/
  Run the frozen VLA backbone and save raw action-token embeddings.

scripts/training/
  Build splits, train heads, evaluate checkpoints, plot predictions, and submit
  Spartan jobs.

scripts/simulation_data_generation/
  Generate our own simulated expert trajectories from fenceline scene YAML
  files.
```

Run scripts from the repository root so relative imports and paths behave
consistently:

```bash
cd /path/to/ag_vla
python scripts/data_processing/check_processed_trajectory.py --help
python scripts/embeddings/extract_vla_embeddings.py --help
python scripts/training/train_mlp_head.py --help
python scripts/simulation_data_generation/generate_fenceline_expert_trajectories.py --help
```

The data-processing scripts produce `trajectory.npz`. The embedding scripts add
`trajectory_with_embeddings.npz`. The training scripts consume the embedded
files.
