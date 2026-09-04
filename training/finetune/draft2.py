import contextlib
import os
import random
import zipfile
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, Dataset, DistributedSampler
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
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.models.projectors import ProprioProjector
from prismatic.models.small_head import Edge_adapter, Proj_Actiontokens
from prismatic.training.train_utils import get_current_action_mask, get_next_actions_mask
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import ACTION_DIM, IGNORE_INDEX, NUM_ACTIONS_CHUNK, POSE_DIM

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _valid_pt(path: Path) -> bool:
    try:
        with zipfile.ZipFile(path) as zf:
            zf.testzip()
        return True
    except (zipfile.BadZipFile, OSError):
        return False

ASYNCVLA_MODEL_ID = "NHirose/AsyncVLA_release"
ASYNCVLA_STEP     = 750_000

WANDB_PROJECT = "aion-r6-vla-training"
WANDB_ENTITY  = "e-12-pye-007-capstone-baddies"

EDGE_OBS_ENCODING_SIZE = 1024
EDGE_MHA_HEADS         = 4
EDGE_MHA_LAYERS        = 4
EDGE_MHA_FF_DIM_FACTOR = 4

_IMG_NORM = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


@dataclass
class PathConfig:
    out_dir:      str = "./out/finetune"
    asyncvla_dir: str = ""  # defaults to HF snapshot of ASYNCVLA_MODEL_ID; set to override
    data_dir:     str = ""  # directory of per-sample .pt files; defaults to $PROJECT_DIR/data


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
    max_steps:               int   = 10_000
    grad_accumulation_steps: int   = 4
    save_freq:               int   = 2_000
    log_freq:                int   = 50
    eval_freq:               int   = 200
    num_steps_before_decay:  int   = 7_000
    gamma:                   float = 0.1
    num_workers:             int   = 4
    val_split:               float = 0.2


@dataclass
class DataConfig:
    max_samples: int = 0  # 0 = use all samples


@dataclass
class Config:
    paths: PathConfig        = field(default_factory=PathConfig)
    lora:  LoraAdapterConfig = field(default_factory=LoraAdapterConfig)
    train: TrainingConfig    = field(default_factory=TrainingConfig)
    data:  DataConfig        = field(default_factory=DataConfig)


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


def setup_distributed() -> Tuple[int, int, int]:
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def load_asyncvla_for_finetune(device: str) -> Tuple:
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction_MMNv1)

    processor = AutoProcessor.from_pretrained(ASYNCVLA_MODEL_ID, trust_remote_code=True)

    vla = AutoModelForVision2Seq.from_pretrained(
        ASYNCVLA_MODEL_ID,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(device)

    # LoRA only on LLM layers — vision backbone (DinoV2+SigLIP) is frozen separately
    target_modules = [
        name for name, m in vla.named_modules()
        if isinstance(m, nn.Linear) and "vision_backbone" not in name
    ]
    lora_config = LoraConfig(
        r=_lora_adapter.rank,
        lora_alpha=min(_lora_adapter.rank, _lora_adapter.lora_alpha),
        lora_dropout=_lora_adapter.dropout,
        target_modules=target_modules,
        init_lora_weights=_lora_adapter.initial_weights,
        use_dora=_lora_adapter.use_dora,
    )
    vla = get_peft_model(vla, lora_config)

    # Freeze vision backbone (DinoV2 + SigLIP)
    vla.base_model.model.vision_backbone.requires_grad_(False)

    if _rank == 0:
        vla.print_trainable_parameters()

    return vla, processor


def _load_support_module(module_class, name: str, device: str, **kwargs) -> nn.Module:
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
    if _rank == 0:
        print(f"[{name}] loaded from {ckpt_path}")

    return module.to(torch.bfloat16).to(device)


def load_support_modules(vla_base, device: str) -> Tuple:
    llm_dim = vla_base.base_model.model.llm_dim

    shead = _load_support_module(
        Edge_adapter,
        "shead",
        device,
        obs_encoding_size=EDGE_OBS_ENCODING_SIZE,
        mha_num_attention_heads=EDGE_MHA_HEADS,
        mha_num_attention_layers=EDGE_MHA_LAYERS,
        mha_ff_dim_factor=EDGE_MHA_FF_DIM_FACTOR,
    )

    action_proj = _load_support_module(
        Proj_Actiontokens,
        "action_proj",
        device,
        input_dim=llm_dim,
        hidden_dim=llm_dim,
        action_dim=1024,
    )

    # pose_projector: fresh init (no checkpoint) — output is masked for modality 7 so weights don't matter
    pose_projector = ProprioProjector(llm_dim=llm_dim, proprio_dim=POSE_DIM)
    pose_projector = pose_projector.to(torch.bfloat16).to(device)

    return shead, action_proj, pose_projector


class SampleDataset(Dataset):
    """
    Each .pt file is one sample:
        pixel_values  : (6, 224, 224)  — 2 camera images × 3 channels, VLA-normalised
        actions       : (8, 4)         — ground-truth trajectory (x, y, cosθ, sinθ)
        c_image       : (3, 96, 96)    — current image [0,1] for Edge_adapter
        p_image       : (3, 96, 96)    — past image [0,1] for Edge_adapter
        instruction   : str            — language navigation goal
    """
    def __init__(self, paths: List[Path]):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        return torch.load(self.paths[idx], weights_only=False)


def make_collate_fn(processor):
    action_tokenizer = ActionTokenizer(processor.tokenizer)
    tokenizer        = processor.tokenizer
    max_len          = tokenizer.model_max_length
    pad_id           = tokenizer.pad_token_id

    def collate_fn(samples: List[dict]) -> dict:
        input_ids_list: List[torch.Tensor] = []
        labels_list:    List[torch.Tensor] = []

        for s in samples:
            actions_np = s["actions"].numpy()  # (8, 4)

            # Build action chunk string: one call per timestep
            action_chunk_str = "".join(
                action_tokenizer(actions_np[t]) for t in range(len(actions_np))
            )

            # Full VLA prompt: instruction + action tokens (same format as sys2.py)
            prompt_builder = PurePromptBuilder("openvla")
            prompt_builder.add_turn("human", s["instruction"])
            prompt_builder.add_turn("gpt",   action_chunk_str)

            full_ids = torch.tensor(
                tokenizer(
                    prompt_builder.get_prompt(),
                    add_special_tokens=True,
                    truncation=True,
                    max_length=max_len,
                ).input_ids,
                dtype=torch.long,
            )

            # Labels: IGNORE_INDEX everywhere except the action token positions
            n_action = len(tokenizer(action_chunk_str, add_special_tokens=False).input_ids)
            labels = torch.full_like(full_ids, IGNORE_INDEX)
            if n_action > 0 and n_action <= len(full_ids):
                labels[-n_action:] = full_ids[-n_action:]

            input_ids_list.append(full_ids)
            labels_list.append(labels)

        def _pad(seqs: List[torch.Tensor], pad_val: int) -> torch.Tensor:
            out = torch.full((len(seqs), max_len), pad_val, dtype=torch.long)
            for i, seq in enumerate(seqs):
                l = min(len(seq), max_len)
                out[i, :l] = seq[:l]
            return out

        input_ids = _pad(input_ids_list, pad_id)
        labels    = _pad(labels_list,    IGNORE_INDEX)

        return {
            "input_ids":            input_ids,
            "attention_mask":       input_ids.ne(pad_id),
            "attention_mask_label": labels.ne(IGNORE_INDEX),
            "labels":               labels,
            "pixel_values":         torch.stack([s["pixel_values"] for s in samples]),
            "goal_pose":            torch.zeros(len(samples), POSE_DIM),  # dummy — masked for modality 7
            "actions":              torch.stack([s["actions"] for s in samples]),
            "c_image":              torch.stack([s["c_image"] for s in samples]),
            "p_image":              torch.stack([s["p_image"] for s in samples]),
            "goal_mask_select":     torch.full((len(samples),), 7),
        }
    return collate_fn


def load_dataset(data_dir: str, processor, rank: int, world_size: int) -> Tuple:
    files = sorted(Path(data_dir).glob("*.pt"))
    if not files:
        raise FileNotFoundError(f"No .pt sample files found in {data_dir}")

    files = [f for f in files if _valid_pt(f)]
    if not files:
        raise RuntimeError(f"No valid .pt files found in {data_dir} (all corrupted?)")

    if _data_params.max_samples > 0:
        files = files[:_data_params.max_samples]

    rng = random.Random(42)
    rng.shuffle(files)

    n_val       = max(1, int(len(files) * _train_params.val_split))
    train_files = files[n_val:]
    val_files   = files[:n_val]

    collate_fn = make_collate_fn(processor)

    train_sampler = DistributedSampler(SampleDataset(train_files), num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler   = DistributedSampler(SampleDataset(val_files),   num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(
        SampleDataset(train_files),
        batch_size=_train_params.batch_size,
        sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=_train_params.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        SampleDataset(val_files),
        batch_size=_train_params.batch_size,
        sampler=val_sampler,
        collate_fn=collate_fn,
        num_workers=_train_params.num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, train_sampler


def run_forward_pass(
    vla,
    action_proj,
    shead,
    pose_projector,
    batch: dict,
    num_patches: int,
    device: str,
    no_grad: bool = False,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    batch_size  = batch["input_ids"].shape[0]
    modality_id = batch["goal_mask_select"].to(torch.bfloat16).to(device)

    img_cur  = _IMG_NORM(batch["c_image"]).to(device).to(torch.bfloat16)
    img_past = _IMG_NORM(batch["p_image"]).to(device).to(torch.bfloat16)

    ground_truth_actions = batch["actions"].to(device).to(torch.bfloat16)

    vla_context = torch.no_grad() if no_grad else torch.enable_grad()
    with vla_context, torch.autocast("cuda", dtype=torch.bfloat16):
        output: CausalLMOutputWithPast = vla(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            attention_mask_label=batch["attention_mask_label"].to(device),
            pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device),
            modality_id=modality_id,
            labels=batch["labels"].to(device),
            output_hidden_states=True,
            use_film=False,
            proprio=batch["goal_pose"].to(torch.bfloat16).to(device),
            proprio_projector=pose_projector,
        )

    gt_ids    = batch["labels"][:, 1:].to(device)
    curr_mask = get_current_action_mask(gt_ids)
    next_mask = get_next_actions_mask(gt_ids)

    last_hidden = output.hidden_states[-1]
    text_hidden = last_hidden[:, num_patches:-1]
    action_hidden = (
        text_hidden[curr_mask | next_mask]
        .reshape(batch_size, NUM_ACTIONS_CHUNK * ACTION_DIM, -1)
        .to(torch.bfloat16)
    )

    projected_actions  = action_proj.predict_action(action_hidden, modality_id)
    predicted_dactions = shead(img_cur, img_past, projected_actions)

    predicted_actions = delta_to_pose(predicted_dactions)

    origin = torch.zeros(batch_size, 1, 4, device=device, dtype=torch.bfloat16)
    origin[:, :, 2] = torch.cos(torch.tensor(0.0))
    origin[:, :, 3] = torch.sin(torch.tensor(0.0))
    smooth_ref = torch.cat([origin, predicted_actions[:, :-1]], dim=1)

    action_ref  = ground_truth_actions
    daction_ref = pose_to_delta(action_ref)

    mse_action = nn.MSELoss()(action_ref, predicted_actions)
    mse_delta  = nn.MSELoss()(daction_ref, predicted_dactions)
    mse_smooth = nn.MSELoss()(smooth_ref, predicted_actions)

    loss = 0.5 * mse_action + 7.5 * mse_delta + 0.1 * mse_smooth

    metrics = {
        "loss":       loss.item(),
        "mse_action": mse_action.item(),
        "mse_delta":  mse_delta.item(),
        "mse_smooth": mse_smooth.item(),
    }
    return loss, metrics


def save_checkpoint(step: int, vla) -> None:
    if _rank != 0:
        return
    ckpt_dir    = Path(_paths.out_dir) / f"step-{step:07d}"
    adapter_dir = ckpt_dir / "lora_adapter"
    os.makedirs(adapter_dir, exist_ok=True)
    vla.module.save_pretrained(adapter_dir)
    print(f"Checkpoint saved → {ckpt_dir}")


def validate(vla, action_proj, shead, pose_projector, val_loader: DataLoader, num_patches: int, device: str) -> Dict[str, float]:
    vla.eval()

    totals: Dict[str, float] = {}
    count = 0

    for batch in val_loader:
        _, metrics = run_forward_pass(vla, action_proj, shead, pose_projector, batch, num_patches, device, no_grad=True)
        for k, v in metrics.items():
            totals[k] = totals.get(k, 0.0) + v
        count += 1

    count_t = torch.tensor(count, dtype=torch.float32, device=device)
    dist.all_reduce(count_t, op=dist.ReduceOp.SUM)
    for k in totals:
        t = torch.tensor(totals[k], dtype=torch.float32, device=device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        totals[k] = (t / count_t).item()

    vla.train()
    return totals


def setup_wandb(world_size: int):
    if _rank != 0:
        return None
    return wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        name=(
            f"r{_lora_adapter.rank}"
            f"_a{_lora_adapter.lora_alpha}"
            f"_dora{int(_lora_adapter.use_dora)}"
            f"_lr{_train_params.learning_rate}"
            f"_bs{_train_params.batch_size * world_size}"
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
            "effective_batch_size":   _train_params.batch_size * world_size,
            "max_steps":              _train_params.max_steps,
            "grad_accumulation":      _train_params.grad_accumulation_steps,
            "num_steps_before_decay": _train_params.num_steps_before_decay,
            "world_size":             world_size,
        },
    )


@draccus.wrap()
def main(cfg: Config) -> None:
    global _paths, _lora_adapter, _train_params, _data_params, _rank

    _rank, world_size, local_rank = setup_distributed()
    device = f"cuda:{local_rank}"

    _paths        = cfg.paths
    _lora_adapter = cfg.lora
    _train_params = cfg.train
    _data_params  = cfg.data

    if not _paths.asyncvla_dir:
        _paths.asyncvla_dir = snapshot_download(ASYNCVLA_MODEL_ID)
        if _rank == 0:
            print(f"AsyncVLA snapshot: {_paths.asyncvla_dir}")

    if _rank == 0:
        os.makedirs(_paths.out_dir, exist_ok=True)
        print(f"Run started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  world_size={world_size}")

    dist.barrier()

    vla, processor = load_asyncvla_for_finetune(device)
    vla = DDP(vla, device_ids=[local_rank], find_unused_parameters=True)
    vla.train()

    shead, action_proj, pose_projector = load_support_modules(vla.module, device)
    shead.eval()
    pose_projector.eval()
    shead.requires_grad_(False)
    pose_projector.requires_grad_(False)
    action_proj.train()

    if _rank == 0:
        trainable = [p for p in vla.parameters() if p.requires_grad]
        print(f"Total trainable params: {sum(p.numel() for p in trainable):,}")

    optimiser = AdamW(
        [p for p in vla.parameters() if p.requires_grad] + list(action_proj.parameters()),
        lr=_train_params.learning_rate,
    )
    scheduler = MultiStepLR(optimiser, milestones=[_train_params.num_steps_before_decay], gamma=_train_params.gamma)

    data_dir = _paths.data_dir or (
        f"{os.environ['PROJECT_DIR']}/data"
        if os.environ.get("PROJECT_DIR") else
        str(Path(__file__).parent / "data")
    )
    train_loader, val_loader, train_sampler = load_dataset(data_dir, processor, _rank, world_size)
    if _rank == 0:
        print(f"Dataset: {len(train_loader.dataset)} train, {len(val_loader.dataset)} val from {data_dir}")

    num_patches = (
        vla.module.base_model.model.vision_backbone.get_num_patches()
        * vla.module.base_model.model.vision_backbone.get_num_images_in_input()
    ) + 1  # +1 for goal-pose proprio token

    wandb_run = setup_wandb(world_size)

    metrics_queues = {k: deque(maxlen=_train_params.grad_accumulation_steps)
                      for k in ("loss", "mse_action", "mse_delta", "mse_smooth")}
    optimiser.zero_grad()
    step  = 0
    epoch = 0

    while True:
        train_sampler.set_epoch(epoch)
        for batch in train_loader:
            sync_grads = (step + 1) % _train_params.grad_accumulation_steps == 0
            ctx = vla.no_sync() if not sync_grads else contextlib.nullcontext()

            with ctx:
                loss, metrics = run_forward_pass(
                    vla, action_proj, shead, pose_projector, batch, num_patches, device, no_grad=False,
                )
                (loss / _train_params.grad_accumulation_steps).backward()

            for k, v in metrics.items():
                metrics_queues[k].append(v)

            if sync_grads:
                optimiser.step()
                scheduler.step()
                optimiser.zero_grad()

            weight_update_step = step // _train_params.grad_accumulation_steps
            metrics_avg = {k: sum(d) / len(d) for k, d in metrics_queues.items() if d}

            if _rank == 0:
                if weight_update_step % _train_params.log_freq == 0:
                    print(
                        f"[step {weight_update_step:>6}/{_train_params.max_steps}]  "
                        + "  ".join(f"{k}={v:.4f}" for k, v in metrics_avg.items())
                        + f"  lr={scheduler.get_last_lr()[0]:.2e}"
                    )
                wandb.log(
                    {"train/" + k: v for k, v in metrics_avg.items()}
                    | {"lr": scheduler.get_last_lr()[0]},
                    step=weight_update_step,
                )

            if weight_update_step > 0 and weight_update_step % _train_params.save_freq == 0:
                save_checkpoint(weight_update_step, vla)

            if weight_update_step > 0 and weight_update_step % _train_params.eval_freq == 0:
                val_metrics = validate(vla, action_proj, shead, pose_projector, val_loader, num_patches, device)
                if _rank == 0:
                    print(
                        f"[val   {weight_update_step:>6}/{_train_params.max_steps}]  "
                        + "  ".join(f"{k}={v:.4f}" for k, v in val_metrics.items())
                    )
                    wandb.log({"val/" + k: v for k, v in val_metrics.items()}, step=weight_update_step)

            step += 1

            if weight_update_step >= _train_params.max_steps:
                save_checkpoint(weight_update_step, vla)
                if _rank == 0 and wandb_run:
                    wandb_run.finish()
                dist.destroy_process_group()
                return

        epoch += 1


if __name__ == "__main__":
    main()
