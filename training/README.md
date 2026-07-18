# AG-VLA Finetuning
## Description
The finetuning script performs end-to-end training for the full AG-VLA architecture (OmniVLA -> action projector -> action head) so that the VLA backbone is better aligned with the action projector and action head. It uses custom made navigation data for the AION robot in an agricultural environment. 

The backbone (OmniVLA) is fine-tuned using LoRA - a parameter-efficient method that freezes the original model weights and trains small adapter matrices instead, significantly reducing memory and compute requirements. The base model weights are downloaded automatically from HuggingFace Hub at training time.

Training is designed to run on [Spartan](https://dashboard.hpc.unimelb.edu.au/), the University of Melbourne HPC cluster, on A100 GPU nodes.

## Getting Started
### Dependencies
- Python 3.10.4
- PyTorch 2.2.0
- FlashAttention 2.5.5
- CUDA 12.1.1
- Linux (Spartan HPC)
- [AsyncVLA](https://asyncvla.github.io/)
- See `AsyncVLA/requirements.txt` for the full Python dependency list

### Data
Datasets (~1 TB total) are stored on [Mediaflux](https://rcs-knowledge-hub.atlassian.net/wiki/spaces/KB/pages/5472333/Mediaflux), UoM's research data storage platform. At job start, the dataset is staged from Mediaflux to the node's local NVMe `/tmp` during training.

The storage layout across systems is:

| What | Where |
|---|---|
| Raw datasets | Mediaflux |
| Final checkpoints | Mediaflux (uploaded at end of job) |
| Active checkpoints (during training) | Spartan project storage (`/data/gpfs/projects/<project-id>/out/`) |
| W&B offline run data | Spartan project storage (`/data/gpfs/projects/<project-id>/wandb/`) |
| HuggingFace model cache | Spartan project storage (`/data/gpfs/projects/<project-id>/.cache/huggingface/`) |
| Dataset during training | `/tmp` (staged from Mediaflux at job start) |

The base OmniVLA model (~15 GB) is downloaded from HuggingFace Hub automatically on first run.

## Installing
Run the setup script once on a Spartan login node before submitting any jobs. It will:
1. Create a Python virtualenv at `/data/gpfs/projects/<project-id>/venvs/training/finetune/`
2. Installed pinned PyTorch 2.2.0 (CUDA 12.1)
3. Install AsyncVLA and visualnav-transformer as editable packages
4. Compile and install FlashAttention 2.5.5
5. Verify that all paths, imports and submodules are in place

```bash
bash training/setup_spartan.sh
```

## Training config
All training configuration is defined via dataclasses at the top of `finetune.py`. 

### `PathConfig`

| Field | Default | Description |
|---|---|---|
| `data_dir` | `./data` | Path to training dataset |
| `out_dir` | `./out/finetune` | Directory for checkpoints and logs |

### `LoraAdapterConfig`

| Field | Default | Description |
|---|---|---|
| `rank` | `128` | Rank of the LoRA adapter matrices — higher = more expressive but more memory |
| `alpha` | `16` | LoRA scaling factor applied during weight updates |
| `dropout` | `0.0` | Dropout probability on adapter layers |
| `initial_weights` | `gaussian` | Initialisation strategy for the LoRA A matrix |
| `use_dora` | `False` | Use DoRA (weight-decomposed LoRA) instead of standard LoRA |

### `TrainingConfig`

| Field | Default | Description |
|---|---|---|
| `batch_size` | `4` | Samples processed per forward pass |
| `learning_rate` | `1e-4` | Initial learning rate for AdamW |
| `max_steps` | `50,000` | Total gradient update steps |
| `grad_accumulation_steps` | `1` | Forward passes accumulated before each weight update |
| `save_freq` | `5,000` | Steps between checkpoint saves |
| `log_freq` | `100` | Steps between training log entries |
| `eval_freq` | `500` | Steps between validation runs |
| `num_steps_before_decay` | `30,000` | Step at which the learning rate scheduler applies decay |
| `gamma` | `0.1` | Multiplicative decay factor applied at `num_steps_before_decay` |
| `num_workers` | `4` | Parallel CPU processes for data loading |

## Executing program
Once setup has passed, submit the training job from the repo root:
```bash
sbatch training/e-12-pye-007_ag-vla-finetune.slurm
```

Monitor the job:
```bash
squeue --me                     # check job status
tail -f slurm-<job-id>.out      # stream stdout logs
tail -f slurm-<job-id>.err      # stream stderr logs
```

Checkpoints are saved to `out/finetune/step-XXXXXXX/` at the frequency set by `save_freq`.

## Monitoring results
Training metrics are logged to [Weights & Biases](https://wandb.ai) under the `aion-r6-vla-training` project. Each run is named `name=f"r{_lora_adapter.rank}_a{_lora_adapter.alpha}_dora{int(_lora_adapter.use_dora)}_lr{_train_params.learning_rate}_bs{_train_params.batch_size}"`.

### W&B offline mode
Spartan compute nodes cannot reach `wandb.ai` directly, so W&B runs in offline mode during training (`WANDB_MODE=offline`). Run data is saved locally to Spartan project storage and must be synced to W&B manually from the login node after the job finishes:

```bash
# Run from the Spartan login node after the job completes
wandb sync /data/gpfs/projects/<project-id>/wandb/run-<id>
```

## Acknowledgements
- [OmniVLA](https://omnivla-nav.github.io/) — backbone vision-language-action model
- [AsyncVLA](https://github.com/lyamatomato/AsyncVLA) — fork providing model architecture and training utilities
- [Spartan HPC](https://dashboard.hpc.unimelb.edu.au/) — University of Melbourne high-performance computing cluster