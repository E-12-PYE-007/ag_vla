# AG-VLA (Agriculture VLA)
## Description
A custom VLA fine-tuned for navigation in an agricultural environment. The backbone VLA is OmniVLA (read more about OmniVLA [here](https://omnivla-nav.github.io/)). The VLA is a dual-architecture, where the backbone is targeted to be run on a remote GPU device and the action head on a Jetson Orin Nano.                                  
Although the VLA is generalisable across different wheeled robots, it is fine-tuned with the AION robot.   

## Repository layout

```
ag_vla/
    ├── AsyncVLA/                                   ← git submodule (lyamatomato/AsyncVLA fork)
    │   ├── prismatic/                              ← model architecture code
    │   ├── experiments/                            ← openvla utilities
    │   ├── visualnav-transformer/                  ← editable install dependency (inside AsyncVLA)
    │   └── ...
    │
    ├── training/
    │   ├── finetune/
    │   │   ├── finetune.py
    │   │   ├── setup_spartan.sh                    ← one-time Spartan environment setup
    │   │   ├── ag-vla-finetune-test.slurm          ← SLURM job: smoke test (20 steps)
    │   │   ├── ag-vla-finetune-single.slurm        ← SLURM job: single training run
    │   │   └── ag-vla-finetune-sweep.slurm         ← SLURM job array: hyperparameter sweep
    │   └── README.md
    │
    └── inference/                                  ← inference scripts
```

## Storage layout

### Spartan project storage

Files except datasets live under `/data/gpfs/projects/<project-id>/` on Spartan (not in this repo).

```
/data/gpfs/projects/<project-id>/
    ├── ag_vla/
    │   └── AsyncVLA/                               ← git submodule clone
    │
    ├── out/                                        ← created at training time
    │   └── finetune/
    │       └── step-0010000/
    │           ├── lora_adapter/
    │           ├── action_projector--10000_checkpoint.pt
    │           └── action_head--10000_checkpoint.pt
    │
    ├── wandb/                                      ← W&B offline run data (sync after job)
    │   └── run-<id>/
    │
    ├── venvs/
    │   └── training/
    │       └── finetune/                           ← virtualenv for finetuning
    │
    └── .cache/
        └── huggingface/                            ← HF model cache (~15 GB for OmniVLA)
```

### Mediaflux

<!-- TODO: fill in Mediaflux namespace and folder structure -->

### Compute node `/tmp`

Datasets are staged from Mediaflux to `/tmp` at job start.
