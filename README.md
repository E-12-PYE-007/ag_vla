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
    │   ├── finetune.py                             
    │   ├── log.py                                  
    │   ├── setup_spartan.sh                        ← one-time Spartan environment setup
    │   └── e-12-pye-007_ag-vla-finetune.slurm     ← SLURM job submission script
    │
    └── inference/                                  ← inference scripts
```

## Spartan project storage layout

All large files live under `/data/gpfs/projects/<project-id>/` on Spartan (not in this repo).

```
/data/gpfs/projects/<project-id>/
    ├── ag_vla/
    │   └── AsyncVLA/                               ← git submodule clone (visualnav-transformer is inside)
    │
    ├── data/
    │   └── <name_of_dataset>/                              ← training dataset
    │
    ├── out/                                        ← created at training time
    │   └── finetune/
    │       ├── training_logs/
    │       │   └── train_20260709_142301.log
    │       └── step-0010000/
    │           ├── lora_adapter/
    │           ├── tokenizer.json
    │           ├── preprocessor_config.json
    │           ├── action_projector--10000_checkpoint.pt
    │           ├── action_head--10000_checkpoint.pt
    │           └── eval/
    │               ├── eval_20260709_190012.log
    │               ├── metrics.json
    │               └── trajectories/
    │
    ├── venvs/
    │   └── training/
    │       └── finetune/                           ← virtualenv for finetuning
    │
    └── .cache/
        └── huggingface/                            ← HF model cache (~15 GB for OmniVLA)
```
