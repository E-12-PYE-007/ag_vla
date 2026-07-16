import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModelForVision2Seq,
    AutoProcessor,
    CausalLMOutputWithPast,
)

# Parameter-Efficient Fine-Tuning - HF library that implements techniques like LoRA
from peft import LoraConfig, get_peft_model

# Weights and Biases - platform for tracking and logging training
import wandb

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction_MMNv1
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.small_head import Proj_Actiontokens
from prismatic.training.train_utils import get_current_action_mask, get_next_actions_mask
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK

from training import logger

# Disable HF's tokenizer library, which uses multiple threads to process text faster
# We want to use parallelism for loading image data instead in DataLoader
os.environ["TOKENIZERS_PARALLELISM"] = "false"

OMNIVLA_MODEL_ID = "NHirose/omnivla-original-balance"
OMNIVLA_STEP = 285_000
MODALITY_ID = 7
DEVICE = "cuda"

#TODO: Add wandb API key in .bashrc for Spartan
WANDB_PROJECT = "aion-r6-vla-training"
WANDB_ENTITY = "e-12-pye-007-capstone-baddies"

@dataclass
class PathConfig:
    data_dir: str = "./data"
    out_dir:  str = "./out/finetune"

#TODO: these might need to be customisable at runtime, also research
# Initial values are from AsyncVLA
@dataclass
class LoraAdapterConfig:
    rank:            int   = 128 # rank of adapter matrices
    alpha:           int   = 16 # scaling during update
    dropout:         float = 0.0
    initial_weights: str   = "gaussian" # initial value of A matrix 
    use_dora:        bool  = False

@dataclass
class TrainingConfig:
    batch_size:              int   = 4 # how many training samples are processed together in one forward pass
    learning_rate:           float = 1e-4 # 
    max_steps:               int   = 50_000 # total number of gradient update steps before training stops
    grad_accumulation_steps: int   = 1 # how many forward passes to accumulate before doing one weight update
    save_freq:               int   = 5_000 # how often the checkpoints get saved
    log_freq:                int   = 100 # frequency to log loss
    eval_freq:               int   = 500  # frequency of validation loop
    num_steps_before_decay:  int   = 30_000 # the step at which the scheduler drops the learning rate by gamma
    gamma:                   float = 0.1 # 
    num_workers:             int   = 4 # how many parallel CPU processes load and preprocess data while the GPU is training

# Define global variables
_paths = PathConfig()
_lora_adapter = LoraAdapterConfig()
_train_params = TrainingConfig()


def load_omnivla_for_finetune():
    # 1. Register and load OmniVLA

    # Register custom HF auto-classes
    #   AutoConfig, AutoImageProcessor etc. are the parent classes to construct a model
    #   Since OpenVLA is a custome model, the model is manually defined

    # Register OpenVLAConfig settings for using "openvla"
    #   OpenVLAConfig contains configs like LLM backbone, action normalisation statistics etc...
    AutoConfig.register("openvla", OpenVLAConfig)
    # Register the image processor which handles resizing, normalising, convesion to tensors
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    # Register the processor which combines the image and the tokenizer into one object
    #   This simply combines, so the image processor still needs to be defined
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    # Register OmniVLA
    #   "Vision2Seq" simply means it's a model that takes images + text as input and ouputs a sequence of tokens
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction_MMNv1)

    # Load OmniVLA processor from HF Hub
    processor = AutoProcessor.from_pretrained(OMNIVLA_MODEL_ID, trust_remote_code=True)

    # Load OmniVLA model weights from HB Hub
    vla = AutoModelForVision2Seq.from_pretrained(
        OMNIVLA_MODEL_ID,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(DEVICE)

    # 2. Prepare OmniVLA for LoRA finetuning
    
    # Ensure LoRA is only applied to layers that contains weights
    #   nn.Linear: performs y = x @ W + b
    #       e.g. For LLMs:
    #               language_model.layers.0.self_attn.q_proj -> nn.Linear
    #               language_model.layers.0.input_layernorm -> nn.LayerNorm
    target_modules = [path for path, module in vla.named_modules() if isinstance(module, nn.Linear)]
    lora_config = LoraConfig(
        r=_lora_adapter.rank,
        lora_alpha=_lora_adapter.rank,
        lora_dropout=_lora_adapter.dropout,
        target_modules=target_modules,
        init_lora_weights=_lora_adapter.initial_weights,
        use_dora=_lora_adapter.use_dora,
    )

    vla = get_peft_model(vla, lora_config)
    vla.print_trainable_parameters()

    return vla, processor

def load_support_modules(llm_dim: int) -> nn.Module:
    #TODO: action_dim depends on action_head architecture
    # Load AsyncVLA's action_projector with empty weights
    #   Compresses LLM hidden states
    action_projector = nn.Module()
    # action_projector = _load_module(
    #     Proj_Actiontokens,
    #     None,
    #     input_dim=llm_dim,
    #     hidden_dim=llm_dim,
    #     action_dim=
    # ).to(torch.bfloat16).to(DEVICE)

    return action_projector

def load_optimiser_scheduler(vla):
    trainable = (
        [p for p in vla.parameters() if p.requires_grad]
        #TODO: list all the parameters for action_projector and action_head
    )
    logger.info(f"Total trainable parameters:  {sum(p.numel() for p in trainable):,}")
    
    optimiser = AdamW(trainable, lr=_train_params.learning_rate)
    scheduler = MultiStepLR(optimiser, milestones=[_train_params.num_steps_before_decay], gamma=_train_params.gamma)

    return optimiser, scheduler

def load_dataset(processor):
    # Handler for converting between raw action numbers and token IDs, OmniVLA represents actions as special tokens
    action_tokeniser = ActionTokenizer(processor.tokenizer)
    #TODO: Load training and validation dataset

def setup_wandb():
    return wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        name=f"finetune+r{_lora_adapter.rank}",
        config={
            "model":                  OMNIVLA_MODEL_ID,
            "omnivla_step":           OMNIVLA_STEP,
            "lora_rank":              _lora_adapter.rank,
            "lora_dropout":           _lora_adapter.dropout,
            "use_dora":               _lora_adapter.use_dora,
            "learning_rate":          _train_params.learning_rate,
            "batch_size":             _train_params.batch_size,
            "max_steps":              _train_params.max_steps,
            "grad_accumulation":      _train_params.grad_accumulation_steps,
            "num_steps_before_decay": _train_params.num_steps_before_decay, 
        }
    )

def _load_module(
        module_class, 
        module_path: Optional[Path], 
        **kwargs # catch-all for any extra keyword arguments
    ) -> nn.Module:
    module = module_class(**kwargs)

    if not module_path.exists():
        msg = f"[{module_class.__name__}] checkpoint not found: {module_path}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    # Define state dict (PyTorch's name for a dict of all a module's weights)
    #   Key: name of parameter
    #   Value: tensor of numbers
    #   e.g. {
    #           "linear1.weight": tensor([[...]])
    #           "linear1.bias":   tensor([...])
    #        }
    sd = torch.load(module_path, map_location=DEVICE, weights_only=True)
    result = module.load_state_dict({k.replace("module.", ""): v for k, v in sd.items()}, strict=False)
    if result.missing_keys:
        logger.warn(f"[{module_class.__name__}] missing keys: {result.missing_keys}")
    if result.unexpected_keys:
        logger.warn(f"[{module_class.__name__}] unexpected keys: {result.unexpected_keys}")
    logger.info(f"[{module_class.__name__}] loaded from {module_path}")
    return module

def _get_ckpt_path(name: str) -> Path:
    return Path(_paths.omnivla_dir) / f"{name}--{OMNIVLA_STEP}_checkpoint.pt"

def _delta_to_pose(delta: torch.Tensor) -> torch.Tensor:
    dx = delta[..., 0]
    dy = delta[..., 1]
    dtheta = torch.atan2(delta[..., 3], delta[..., 2])
    
    x = dx[:, 0]
    y = dy[:, 0]
    theta = dtheta[:, 0]

    poses = []
    poses.append(torch.stack([x, y, torch.cos(theta), torch.sin(theta)], dim=-1))

    for t in range(1, delta.shape[1]):
        ct, st = torch.cos(theta), torch.sin(theta)
        x = x + ct * dx[:, t] - st * dy[:, t]
        y = y + st * dx[:, t] + ct * dy[:, t]
        theta = theta + dtheta[:, t]
        poses.append(torch.stack([x, y, torch.cos(theta), torch.sin(theta)], dim=-1))
    return torch.stack(poses, dim=-1)

def omnivla_forward_pass(
    vla,
    batch: dict,
    num_patches: int,
    no_grad: bool
) -> torch.Tensor:
    # Check whether this is a training or validation run
    context = torch.no_grad() if no_grad else torch.enable_grad()

    # CausalLMOutputWithPast: a dataclass defined by HF that bundles together everything the model outputs, this includes:
    #   hidden_states which records the embeddings at every layer of the LLM
    with context, torch.autocast(DEVICE, dtype=torch.bfloat16):
        #TODO: get these when data structure is defined
        output: CausalLMOutputWithPast = vla(
            # input_ids=
            # attention_mask=
            # attention_mask_label=
            # pixel_valus=
            modality_id=MODALITY_ID,
            # labels=
            output_hidden_states=True,
            # proprio=
            use_film=False,
        )

    # Obtain positions of current and next action tokens in hidden action state embedding
    gt_ids = batch["labels"][:, 1:].to(DEVICE) #TODO: check labels
    is_curr_action = get_current_action_mask(gt_ids)
    is_next_action = get_next_actions_mask(gt_ids)

    # Obtain just the last hidden states since that contains the embedding for hidden action states
    last_hidden_states = output.hidden_states[-1]
    # Extract just the action tokens (collapses to 2D matrix)
    hidden_action_states = last_hidden_states[:, num_patches:-1][is_curr_action | is_next_action]

    return hidden_action_states.reshape(_train_params.batch_size, NUM_ACTIONS_CHUNK * ACTION_DIM, -1).to(torch.bfloat16)

def action_head_forward_pass(
        hidden_action_states,
        action_projector: Proj_Actiontokens,
        action_head,
        batch: dict,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    
    #TODO: extract ground truths
    gt_waypoint_norm = Optional()

    # Convert hidden states to compact trajectory tokens
    projected_action_states = action_projector.predict_action(
        hidden_action_states, MODALITY_ID
    )

    #TODO: Run through action head
    predicted_deltas = action_head(projected_action_states)
    predicted_waypoints = _delta_to_pose(predicted_deltas)
    
    # Generate the waypoints that are shifted by one step forward
    # This is used to determine how smooth the path is (penalities are applied for sudden jumps in waypoints)
    origin = torch.zeros(_train_params.batch_size, 1, ACTION_DIM, device=DEVICE, dtype=torch.bfloat16)
    origin[:, :, 2] = torch.cos(torch.tensor(0.0))
    origin[:, :, 3] = torch.sin(torch.tensor(0.0))
    gt_waypoints_shifted = torch.cat([origin, predicted_waypoints[:, :-1]], dim=1)

    obj_loss = nn.MSELoss()(gt_waypoint_norm, predicted_waypoints[:, -1, :2])
    smooth_loss = nn.MSELoss()(gt_waypoints_shifted, predicted_waypoints)

    #TODO: verify these values
    loss = 0.1 * obj_loss + 0.1 * smooth_loss

    #TODO: check if we need other metrics
    metrics = {
        "loss": loss.item(),
        "mse_obj": obj_loss.item(),
        "mse_smooth": smooth_loss.item(),
    }
    return loss, metrics

def save_checkpoint(
    step: int,
    vla,
    processor,
    action_projector: nn.Module,
    action_head: nn.Module,
) -> None:
    ckpt_dir = Path(_paths.out_dir) / f"step-{step:07d}"
    adapter_dir = ckpt_dir / "lora_adapter"
    os.makedirs(adapter_dir, exist_ok=True)

    # save_pretrained(): a HF method that saves with metadata that lets you reload the model later using PeftModel.from_pretrained()
    #   you don't need to know the original architecture
    processor.save_pretrained(ckpt_dir)
    vla.save_pretrained(adapter_dir)

    # torch.save(): writes the weights dictionary with no metadata
    #   you need to instantiate the class and call load_state_dict() to fill in the weights
    torch.save(action_projector.state_dict(), ckpt_dir / f"action_projector--{step}_checkpoint.pt")
    torch.save(action_head.state_dict(), ckpt_dir / f"action_head--{step}_checkpoint.pt")
    logger.info(f"Checkpoint saved → {ckpt_dir}")

def validate(
    vla,
    action_projector: Proj_Actiontokens,
    action_head: nn.Module,
    val_set: DataLoader,
    num_patches: int,
) -> Dict[str, float]:
    vla.eval()
    action_head.eval()

    all_metrics: Dict[str, float] = {}
    count = 0
    for batch in val_set:
        hidden_action_states = omnivla_forward_pass(vla, batch, num_patches, no_grad=True)
        _, metrics = action_head_forward_pass(hidden_action_states, action_projector, action_head)
        # For every batch compute average metric values
        for k, v in metrics.items():
            if k not in all_metrics:
                all_metrics[k] = 0.0
            all_metrics[k] += v
        count += 1

    vla.train()
    action_projector.train()
    action_head.train()

    return {k: v / count for k, v in all_metrics.items()}

def main() -> None:
    # Ensure directory for outputs exist
    out_dir = Path(_paths.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    logger.setup(out_dir / "logs", "train.log")

    logger.info(f"Paths:   {_paths}")
    logger.info(f"Adapter: {_lora_adapter}")
    logger.info(f"Training parameters: {_train_params}")

    # Load OmniVLA with weights frozen and the LoRA adapter matrices
    vla, processor = load_omnivla_for_finetune()
    # Set the model to training mode which sets the following two things
    #   Dropout - randomly zeros out neruons during training to prevent overfitting. In eval mode it's disabled
    #   Batch normalisation - a techniques that normalises the values flowing through a layer during training to keep them in a stable range. In eval mode it uses data acquired during training
    vla.train()

     # Load supporting modules: action_projector (LLM hidden states → action space)
    llm_dim = vla.base_model.model.llm_dim
    action_projector = load_support_modules(llm_dim)

    #TODO: load action head
    action_head = Optional()

    optimiser, scheduler = load_optimiser_scheduler(vla)

    train_set, val_set = load_dataset(processor)
    
    wandb_run = setup_wandb()

    # num_patches = patches per image × num_images
    #   patch = grid of 16 x 16 pixels
    num_patches = (
        vla.base_model.model.vision_backbone.get_num_patches()
        * vla.base_model.vision_backbone.get_num_images_in_input()
    )

    # Training main loop
    step = 0
    # Queues for storing run metrics for per gradient accumulation cycle
    # Drops old data at the end of each gradient accumulation cycle
    metrics_queues = {k: deque(maxlen=_train_params.grad_accumulation_steps)
                    for k in ("loss", "mse_obj", "mse_smooth")}
    # Reset gradients 
    optimiser.zero_grad()

    while True:
        for batch in train_set:
            hidden_action_states = omnivla_forward_pass(vla, batch, num_patches, no_grad=False)
            loss, metrics = action_head_forward_pass(hidden_action_states, action_projector, action_head, batch)

            # Run back propagation and store it with each weight
            (loss / _train_params.grad_accumulation_steps).backward()

            for k, v in metrics.items():
                metrics_queues[k].append(v)

            # Update weights per grad accumulation cycle
            if step % _train_params.grad_accumulation_steps == 0:
                optimiser.step()
                scheduler.step()
                optimiser.zero_grad()

            # step: counts every forward pass
            # weight_update_step: counts every weight update, increments every grad accumulation cycle
            weight_update_step = step // _train_params.grad_accumulation_steps

            metrics_avg = {k: sum(d) / len(d) for k, d in metrics_queues.items() if d}

            if weight_update_step % _train_params.log_freq == 0:
                lr = scheduler.get_last_lr()[0]
                logger.info(
                    f"[step {weight_update_step:>6}/{_train_params.max_steps}]  "
                    + "  ".join(f"{k}={v:.4f}" for k, v in metrics_avg.items())
                    + f"  lr={lr:.2e}"
                )

            # Store metrics under train/ section
            wandb.log(
                {"train/" + k: v for k, v in metrics_avg.items()}
                | {"lr": scheduler.get_last_lr()[0]},
                step=weight_update_step,
            )

            if weight_update_step > 0 and weight_update_step % _train_params.save_freq == 0:
                save_checkpoint(
                    weight_update_step, 
                    vla, 
                    processor,
                    action_projector,
                    action_head
                )

            if weight_update_step > 0 and weight_update_step % _train_params.eval_freq == 0:
                val_metrics = validate(
                    vla,
                    action_projector,
                    action_head,
                    val_set,
                    num_patches
                )
                logger.info(
                    f"[val   {weight_update_step:>6}/{_train_params.max_steps}]  "
                    + "  ".join(f"{k}={v:.4f}" for k, v in val_metrics.items())
                )
                # Store metrics under /val section
                wandb.log({"val/" + k: v for k, v in val_metrics.items()}, step=weight_update_step)

            step += 1

            # Finish training
            if weight_update_step >= _train_params.max_steps:
                logger.info(f"Reached max_steps={_train_params.max_steps}, stopping.")
                save_checkpoint(
                    weight_update_step,
                    vla,
                    processor,
                    action_projector,
                    action_head
                )
                wandb_run.finish()
                break
    

if __name__ == "__main__":
    main()