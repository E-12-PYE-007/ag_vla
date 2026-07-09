```
ag_vla/
    ├── AsyncVLA/                                   ← git submodule (pinned commit)
    │   ├── prismatic/                              ← model architecture code
    │   ├── experiments/                            ← openvla utilities
    │   └── ...
    │
    ├── training/                                   ← training scripts
    │   ├── dataset.py
    │   ├── log.py
    │   ├── finetune.py
    │   ├── eval.py
    │   ├── environment.yml
    │   └── submit_spartan.sh   
    │
    ├── inference/                                  ← inference scripts
    │
    ├── weights/                                    ← relevant checkpoints
    │   ├── proprio_projector--285000_checkpoint.pt
    │   ├── action_proj--750000_checkpoint.pt
    │   └── shead--750000_checkpoint.pt
    │
    ├── config_nav/
    │   └── dataset_config.yaml                     ← nav architecture config
    │
    ├── data/
    │   ├── fenceline/                              ← dataset
    │   └── roadside/
    │
    └── out/                                        ← created at training time
        └── finetune/
        │   ├── training_logs/  
        │   │   └── train_20260709_142301.log
        │   └── step-0050000/
        │       ├── lora_adapter/
        │       ├── tokenizer.json
        │       ├── preprocessor_config.json
        │       ├── pose_projector--50000_checkpoint.pt
        │       ├── action_proj--50000_checkpoint.pt
        │       ├── shead--50000_checkpoint.pt
        │       └── eval/
        │           ├── eval_20260709_190012.log
        │           ├── metrics.json
        │           └── trajectories/
        └── actionhead/
```