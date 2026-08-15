import os
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torchvision import transforms

import draccus
import wandb
from huggingface_hub import snapshot_download
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModelForVision2Seq,
    AutoProcessor,
)
from transformers.modeling_outputs import CausalLMOutputWithPast

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction_MMNv1
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.small_head import Edge_adapter, Proj_Actiontokens
from prismatic.training.train_utils import get_current_action_mask, get_next_actions_mask
from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK

os.environ["TOKENIZERS_PARALLELISM"] = "false"

ASYNCVLA_MODEL_ID = "NHirose/AsyncVLA_release"
ASYNCVLA_STEP     = 750_000
DEVICE            = "cuda"

WANDB_PROJECT = "vla-finetune-testing"
WANDB_ENTITY  = "e-12-pye-007-capstone-baddies"

# Edge_adapter architecture (from AsyncVLA config_nav/dataset_config.yaml)
EDGE_OBS_ENCODING_SIZE = 1024
EDGE_MHA_HEADS         = 4
EDGE_MHA_LAYERS        = 4
EDGE_MHA_FF_DIM_FACTOR = 4

# ImageNet normalisation applied to c_image / p_image before Edge_adapter
_IMG_NORM = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


@dataclass
class PathConfig:
    out_dir:      str = "./out/finetune"
    asyncvla_dir: str = ""  # defaults to HF snapshot of ASYNCVLA_MODEL_ID; set to override
    data_dir:     str = ""                    # defaults to $PROJECT_DIR/fake_data or ./fake_data if unset

@dataclass
class LoraAdapterConfig:
    rank:            int   = 32
    lora_alpha:      int   = 16   # convention: min(rank, 16)
    dropout:         float = 0.0
    initial_weights: str   = "gaussian"
    use_dora:        bool  = False

@dataclass
class TrainingConfig:
    batch_size:              int   = 4
    learning_rate:           float = 5e-4
    max_steps:               int   = 50_000
    grad_accumulation_steps: int   = 1
    save_freq:               int   = 5_000
    log_freq:                int   = 100
    eval_freq:               int   = 500
    num_steps_before_decay:  int   = 30_000
    gamma:                   float = 0.1
    num_workers:             int   = 8

@dataclass
class Config:
    paths: PathConfig        = field(default_factory=PathConfig)
    lora:  LoraAdapterConfig = field(default_factory=LoraAdapterConfig)
    train: TrainingConfig    = field(default_factory=TrainingConfig)


def delta_to_pose(delta: torch.Tensor) -> torch.Tensor:
    """delta: (N, T, 4) → pose: (N, T, 4)"""
    dx = delta[..., 0]
    dy = delta[..., 1]
    dtheta = torch.atan2(delta[..., 3], delta[..., 2])
    x, y, theta = dx[:, 0], dy[:, 0], dtheta[:, 0]
    poses = [torch.stack([x, y, torch.cos(theta), torch.sin(theta)], dim=-1)]
    for t in range(1, delta.shape[1]):
        ct, st = torch.cos(theta), torch.sin(theta)
        x = x + ct * dx[:, t] - st * dy[:, t]
        y = y + st * dx[:, t] + ct * dy[:, t]
        theta = theta + dtheta[:, t]
        poses.append(torch.stack([x, y, torch.cos(theta), torch.sin(theta)], dim=-1))
    return torch.stack(poses, dim=1)


def pose_to_delta(pose: torch.Tensor) -> torch.Tensor:
    """pose: (N, T, 4) → delta: (N, T, 4)"""
    x = pose[..., 0]
    y = pose[..., 1]
    theta = torch.atan2(pose[..., 3], pose[..., 2])
    delta_list = [pose[:, 0, :]]
    for t in range(1, pose.shape[1]):
        dx = x[:, t] - x[:, t - 1]
        dy = y[:, t] - y[:, t - 1]
        dtheta = theta[:, t] - theta[:, t - 1]
        ct, st = torch.cos(theta[:, t - 1]), torch.sin(theta[:, t - 1])
        delta_list.append(torch.stack(
            [ct * dx + st * dy, -st * dx + ct * dy, torch.cos(dtheta), torch.sin(dtheta)], dim=-1
        ))
    return torch.stack(delta_list, dim=1)


def load_asyncvla_for_finetune() -> Tuple:
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction_MMNv1)

    processor = AutoProcessor.from_pretrained(ASYNCVLA_MODEL_ID, trust_remote_code=True)

    vla = AutoModelForVision2Seq.from_pretrained(
        ASYNCVLA_MODEL_ID,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(DEVICE)

    target_modules = [name for name, m in vla.named_modules() if isinstance(m, nn.Linear)]
    lora_config = LoraConfig(
        r=_lora_adapter.rank,
        lora_alpha=min(_lora_adapter.rank, _lora_adapter.lora_alpha),  # convention: min(rank, 16)
        lora_dropout=_lora_adapter.dropout,
        target_modules=target_modules,
        init_lora_weights=_lora_adapter.initial_weights,
        use_dora=_lora_adapter.use_dora,
    )
    vla = get_peft_model(vla, lora_config)
    vla.print_trainable_parameters()

    return vla, processor


def _load_support_module(module_class, name: str, **kwargs) -> nn.Module:
    module = module_class(**kwargs)

    ckpt_path = Path(_paths.asyncvla_dir) / f"{name}--{ASYNCVLA_STEP}_checkpoint.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"[{name}] checkpoint not found: {ckpt_path}")

    sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    missing, unexpected = module.load_state_dict(sd, strict=False)
    if missing:
        print(f"[WARN] [{name}] missing keys: {missing}")
    if unexpected:
        print(f"[WARN] [{name}] unexpected keys: {unexpected}")
    print(f"[{name}] loaded from {ckpt_path}")

    return module.to(torch.bfloat16).to(DEVICE)


def load_support_modules(vla) -> Tuple:
    llm_dim = vla.base_model.model.llm_dim

    shead = _load_support_module(
        Edge_adapter,
        "shead",
        obs_encoding_size=EDGE_OBS_ENCODING_SIZE,
        mha_num_attention_heads=EDGE_MHA_HEADS,
        mha_num_attention_layers=EDGE_MHA_LAYERS,
        mha_ff_dim_factor=EDGE_MHA_FF_DIM_FACTOR,
    )

    action_proj = _load_support_module(
        Proj_Actiontokens,
        "action_proj",
        input_dim=llm_dim,
        hidden_dim=llm_dim,
        action_dim=1024,
    )

    return shead, action_proj


class BatchDataset(torch.utils.data.Dataset):
    """Each item is one pre-batched .pt file; the DataLoader unwraps it directly."""
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        return torch.load(self.paths[idx], weights_only=True)


def load_dataset(data_dir: str) -> Tuple[DataLoader, DataLoader]:
    files = sorted(Path(data_dir).glob("batch_*.pt"))
    if not files:
        raise FileNotFoundError(f"No batch_*.pt files found in {data_dir}")
    train_files, val_files = files[:25], files[25:]

    def collate(items):
        return items[0]  # each .pt is already a full batch

    train_loader = DataLoader(BatchDataset(train_files), batch_size=1, shuffle=True,  collate_fn=collate, num_workers=0)
    val_loader   = DataLoader(BatchDataset(val_files),   batch_size=1, shuffle=False, collate_fn=collate, num_workers=0)
    return train_loader, val_loader


def run_forward_pass(
    vla,
    action_proj,
    shead,
    batch: dict,
    num_patches: int,
    no_grad: bool = False,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    batch_size = batch["input_ids"].shape[0]
    modality_id = batch["goal_mask_select"].to(torch.bfloat16).to(DEVICE)

    img_cur  = _IMG_NORM(batch["c_image"]).to(DEVICE).to(torch.bfloat16)
    img_past = _IMG_NORM(batch["p_image"]).to(DEVICE).to(torch.bfloat16)

    ground_truth_actions = batch["actions"].to(DEVICE).to(torch.bfloat16)
    pose_goal = batch["obj_pose_norm"].to(DEVICE).to(torch.bfloat16)

    vla_context = torch.no_grad() if no_grad else torch.enable_grad()
    with vla_context, torch.autocast(DEVICE, dtype=torch.bfloat16):
        output: CausalLMOutputWithPast = vla(
            input_ids=batch["input_ids"].to(DEVICE),
            attention_mask=batch["attention_mask"].to(DEVICE),
            attention_mask_label=batch["attention_mask_label"].to(DEVICE),
            pixel_values=batch["pixel_values"].to(torch.bfloat16).to(DEVICE),
            modality_id=modality_id,
            labels=batch["labels"].to(DEVICE),
            output_hidden_states=True,
            use_film=False,
        )

    gt_ids = batch["labels"][:, 1:].to(DEVICE)
    curr_mask = get_current_action_mask(gt_ids)
    next_mask = get_next_actions_mask(gt_ids)

    last_hidden = output.hidden_states[-1]
    text_hidden = last_hidden[:, num_patches:-1]
    action_hidden = (
        text_hidden[curr_mask | next_mask]
        .reshape(batch_size, NUM_ACTIONS_CHUNK * ACTION_DIM, -1)
        .to(torch.bfloat16)
    )

    # action_proj and shead are frozen — run without gradient
    with torch.no_grad():
        projected_actions = action_proj.predict_action(action_hidden.detach(), modality_id)
        predicted_dactions = shead(img_cur, img_past, projected_actions)

    predicted_actions = delta_to_pose(predicted_dactions)
    action_ref  = ground_truth_actions
    daction_ref = pose_to_delta(action_ref)

    # lan_bool: language-conditioned samples (modality 7 or 8) use obj_pose loss
    lan_bool = ((batch["goal_mask_select"] == 7) | (batch["goal_mask_select"] == 8)).to(DEVICE)

    origin = torch.zeros(batch_size, 1, 4, device=DEVICE, dtype=torch.bfloat16)
    origin[:, :, 2] = torch.cos(torch.tensor(0.0))
    origin[:, :, 3] = torch.sin(torch.tensor(0.0))
    smooth_ref = torch.cat([origin, predicted_actions[:, :-1]], dim=1)

    non_lan = ~lan_bool
    loss = torch.zeros(1, device=DEVICE, dtype=torch.bfloat16)
    mse_action = mse_delta = mse_obj = 0.0

    if non_lan.any():
        mse_action = nn.MSELoss()(action_ref[non_lan], predicted_actions[non_lan])
        mse_delta  = nn.MSELoss()(daction_ref[non_lan], predicted_dactions[non_lan])
        loss = loss + 0.5 * mse_action + 0.5 * 15.0 * mse_delta
        mse_action, mse_delta = mse_action.item(), mse_delta.item()

    if lan_bool.any():
        mse_obj = nn.MSELoss()(pose_goal[lan_bool], predicted_actions[:, -1, :2][lan_bool])
        loss    = loss + 0.1 * mse_obj
        mse_obj = mse_obj.item()

    mse_smooth = nn.MSELoss()(smooth_ref, predicted_actions)
    loss       = loss + 0.1 * mse_smooth
    loss       = loss.squeeze()

    metrics = {
        "loss":       loss.item(),
        "mse_action": mse_action,
        "mse_delta":  mse_delta,
        "mse_obj":    mse_obj,
        "mse_smooth": mse_smooth.item(),
    }
    return loss, metrics


def save_checkpoint(step: int, vla) -> None:
    ckpt_dir = Path(_paths.out_dir) / f"step-{step:07d}"
    adapter_dir = ckpt_dir / "lora_adapter"
    os.makedirs(adapter_dir, exist_ok=True)

    # Only save LoRA adapter — shead/action_proj/pose_projector are frozen (use AsyncVLA originals)
    vla.save_pretrained(adapter_dir)
    print(f"Checkpoint saved → {ckpt_dir}")


def validate(
    vla,
    action_proj,
    shead,
    val_loader: DataLoader,
    num_patches: int,
) -> Dict[str, float]:
    vla.eval()

    all_metrics: Dict[str, float] = {}
    count = 0

    for batch in val_loader:
        _, metrics = run_forward_pass(
            vla, action_proj, shead, batch, num_patches, no_grad=True,
        )
        for k, v in metrics.items():
            all_metrics[k] = all_metrics.get(k, 0.0) + v
        count += 1

    vla.train()
    return {k: v / max(count, 1) for k, v in all_metrics.items()}


def setup_wandb():
    return wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        name=(
            f"r{_lora_adapter.rank}"
            f"_a{_lora_adapter.lora_alpha}"
            f"_dora{int(_lora_adapter.use_dora)}"
            f"_lr{_train_params.learning_rate}"
            f"_bs{_train_params.batch_size}"
        ),
        config={
            "model":                  ASYNCVLA_MODEL_ID,
            "asyncvla_step":          ASYNCVLA_STEP,
            "lora_rank":              _lora_adapter.rank,
            "lora_alpha":             _lora_adapter.lora_alpha,
            "lora_dropout":           _lora_adapter.dropout,
            "use_dora":               _lora_adapter.use_dora,
            "learning_rate":          _train_params.learning_rate,
            "batch_size":             _train_params.batch_size,
            "max_steps":              _train_params.max_steps,
            "grad_accumulation":      _train_params.grad_accumulation_steps,
            "num_steps_before_decay": _train_params.num_steps_before_decay,
        },
    )


@draccus.wrap()
def main(cfg: Config) -> None:
    global _paths, _lora_adapter, _train_params
    _paths        = cfg.paths
    _lora_adapter = cfg.lora
    _train_params = cfg.train

    if not _paths.asyncvla_dir:
        _paths.asyncvla_dir = snapshot_download(ASYNCVLA_MODEL_ID)
        print(f"AsyncVLA snapshot: {_paths.asyncvla_dir}")

    os.makedirs(_paths.out_dir, exist_ok=True)
    print(f"Run started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Paths: {_paths}")
    print(f"Adapter: {_lora_adapter}")
    print(f"Training: {_train_params}")

    vla, _ = load_asyncvla_for_finetune()
    vla.train()

    shead, action_proj = load_support_modules(vla)
    shead.eval()
    action_proj.eval()

    trainable = [p for p in vla.parameters() if p.requires_grad]
    print(f"Total trainable params: {sum(p.numel() for p in trainable):,}")
    optimiser = AdamW(trainable, lr=_train_params.learning_rate)
    scheduler = MultiStepLR(optimiser, milestones=[_train_params.num_steps_before_decay], gamma=_train_params.gamma)

    data_dir = _paths.data_dir or (
        f"{os.environ['PROJECT_DIR']}/ag_vla/training/finetune/fake_data"
        if os.environ.get("PROJECT_DIR") else
        str(Path(__file__).parent / "fake_data")
    )
    train_loader, val_loader = load_dataset(data_dir)
    print(f"Dataset: {len(train_loader)} train batches, {len(val_loader)} val batches from {data_dir}")

    num_patches = (
        vla.base_model.model.vision_backbone.get_num_patches()
        * vla.base_model.model.vision_backbone.get_num_images_in_input()
    ) + 1

    wandb_run = setup_wandb()

    metrics_queues = {k: deque(maxlen=_train_params.grad_accumulation_steps)
                      for k in ("loss", "mse_action", "mse_delta", "mse_obj", "mse_smooth")}
    optimiser.zero_grad()
    step = 0

    while True:
        for batch in train_loader:
            loss, metrics = run_forward_pass(
                vla, action_proj, shead, batch, num_patches, no_grad=False,
            )
            (loss / _train_params.grad_accumulation_steps).backward()

            for k, v in metrics.items():
                metrics_queues[k].append(v)

            if (step + 1) % _train_params.grad_accumulation_steps == 0:
                optimiser.step()
                scheduler.step()
                optimiser.zero_grad()

            weight_update_step = step // _train_params.grad_accumulation_steps
            metrics_avg = {k: sum(d) / len(d) for k, d in metrics_queues.items() if d}

            if weight_update_step % _train_params.log_freq == 0:
                lr = scheduler.get_last_lr()[0]
                print(
                    f"[step {weight_update_step:>6}/{_train_params.max_steps}]  "
                    + "  ".join(f"{k}={v:.4f}" for k, v in metrics_avg.items())
                    + f"  lr={lr:.2e}"
                )
            wandb.log(
                {"train/" + k: v for k, v in metrics_avg.items()}
                | {"lr": scheduler.get_last_lr()[0]},
                step=weight_update_step,
            )

            if weight_update_step > 0 and weight_update_step % _train_params.save_freq == 0:
                save_checkpoint(weight_update_step, vla)

            if val_loader is not None and weight_update_step > 0 and weight_update_step % _train_params.eval_freq == 0:
                val_metrics = validate(vla, action_proj, shead, val_loader, num_patches)
                print(
                    f"[val   {weight_update_step:>6}/{_train_params.max_steps}]  "
                    + "  ".join(f"{k}={v:.4f}" for k, v in val_metrics.items())
                )
                wandb.log({"val/" + k: v for k, v in val_metrics.items()}, step=weight_update_step)

            step += 1

            if weight_update_step >= _train_params.max_steps:
                save_checkpoint(weight_update_step, vla)
                wandb_run.finish()
                return


if __name__ == "__main__":
    main()
