## Repository layout

```
ag_vla/
    ├── AsyncVLA/                                   ← git submodule (lyamatomato/AsyncVLA fork)
    │   ├── prismatic/                              ← model architecture code
    │   ├── experiments/                            ← openvla utilities
    │   └── ...
    │
    ├── training/                                  
    │   ├── finetune.py                             
    │   ├── log.py                                  
    │   ├── requirements.txt                        ← dependency notes (see setup_spartan.sh)
    │   ├── setup_spartan.sh                        ← one-time Spartan environment setup
    │   └── submit_spartan.slurm                    ← SLURM job submission script
    │
    └── inference/                                  ← inference scripts
```

## Spartan project storage layout

All large files live under `/data/gpfs/projects/<project-id>/` on Spartan (not in this repo).

```
/data/gpfs/projects/<project-id>/
    ├── ag_vla/
    │   ├── AsyncVLA/                               ← git submodule clone
    │   └── visualnav-transformer/                  ← editable install dependency
    │
    ├── weights/
    │   └── omnivla_release/                        ← OmniVLA head checkpoints
    │       └── proprio_projector--285000_checkpoint.pt
    │
    ├── data/
    │   └── fenceline/                              ← training dataset (~1000 samples)
    │
    ├── out/                                        ← created at training time
    │   └── finetune/
    │       ├── training_logs/
    │       │   └── train_20260709_142301.log
    │       └── step-0010000/
    │           ├── lora_adapter/
    │           ├── tokenizer.json
    │           ├── preprocessor_config.json
    │           ├── pose_projector--10000_checkpoint.pt
    │           ├── action_proj--10000_checkpoint.pt
    │           ├── action_head--10000_checkpoint.pt
    │           └── eval/
    │               ├── eval_20260709_190012.log
    │               ├── metrics.json
    │               └── trajectories/
    │
    ├── venvs/
    │   └── training/
    │       └── finetune/                           ← virtualenv (PyTorch, transformers, etc.)
    │
    └── .cache/
        └── huggingface/                            ← HF model cache (~15 GB for OmniVLA)
```
