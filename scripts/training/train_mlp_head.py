from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from flow_head.asyncvla_projector import Proj_Actiontokens, load_projector_state
from flow_head.dataset import NormalizationStats, TrajectoryEmbeddingDataset
from flow_head.mlp_waypoint_head import MLPWaypointHead, MLPWaypointHeadConfig
from flow_head.splits import dataset_indices_for_split, load_split_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a deterministic MLP waypoint head on VLA embeddings.")
    parser.add_argument("--data", "--data-dir", "--data_dir", dest="data", type=Path, required=True)
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", type=Path, default=Path("checkpoints/mlp_head"))
    parser.add_argument("--context-dim", type=int, default=1024)
    parser.add_argument("--num-context-tokens", type=int, default=8)
    parser.add_argument("--use-asyncvla-projector", action="store_true", help="Project raw [B, 32, 4096] tokens to [B, 8, 1024].")
    parser.add_argument("--projector-checkpoint", "--projector_checkpoint", dest="projector_checkpoint", type=Path, default=None, help="Optional AsyncVLA projector checkpoint.")
    parser.add_argument("--train-projector", action="store_true", help="Unfreeze projector. Required when training a projector from scratch.")
    parser.add_argument("--raw-context-dim", type=int, default=4096)
    parser.add_argument("--raw-action-dim", type=int, default=4)
    parser.add_argument("--robot-state-dim", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--waypoint-dim", type=int, default=3)
    parser.add_argument("--hidden-dim", type=int, default=4096)
    parser.add_argument("--num-blocks", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--no-modality-id", action="store_true")
    parser.add_argument("--default-modality-id", type=float, default=7.0)
    parser.add_argument("--loss-type", choices=["l1", "mse", "smooth_l1"], default="l1")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--split-json", type=Path, default=None)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--normalize-waypoints",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Train in normalized waypoint-label space. Use --no-normalize-waypoints to disable.",
    )
    parser.add_argument("--normalize-robot-state", action="store_true")
    parser.add_argument("--stats-path", type=Path, default=None)
    parser.add_argument("--cache-trajectories", action="store_true")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--max-forward-distance", type=float, default=3.0)
    parser.add_argument("--max-lateral-distance", type=float, default=1.5)
    parser.add_argument("--max-yaw-change", type=float, default=1.57)
    return parser.parse_args()


def collate_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    batch: dict[str, Any] = {}
    all_keys = set().union(*(sample.keys() for sample in samples))
    for key in sorted(all_keys):
        values = [sample.get(key) for sample in samples]
        if any(value is None for value in values):
            continue
        if all(torch.is_tensor(value) for value in values):
            batch[key] = torch.stack(values)
        else:
            batch[key] = values
    return batch


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def split_indices(length: int, val_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(length, generator=generator).tolist()
    val_size = int(round(length * val_fraction))
    if length > 1 and val_fraction > 0:
        val_size = max(1, min(val_size, length - 1))
    return perm[val_size:], perm[:val_size]


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value for key, value in batch.items()}


def compute_subset_stats(subset: Subset) -> NormalizationStats:
    waypoint_values = []
    robot_values = []
    for sample in subset:
        waypoint_values.append(sample["waypoints"].reshape(-1, sample["waypoints"].shape[-1]))
        if "robot_state" in sample:
            robot_values.append(sample["robot_state"].reshape(-1, sample["robot_state"].shape[-1]))

    waypoints = torch.cat(waypoint_values, dim=0).float()
    waypoint_mean = waypoints.mean(dim=0)
    waypoint_std = waypoints.std(dim=0).clamp_min(1e-4)

    robot_mean = None
    robot_std = None
    if robot_values:
        robot = torch.cat(robot_values, dim=0).float()
        robot_mean = robot.mean(dim=0)
        robot_std = robot.std(dim=0).clamp_min(1e-4)

    return NormalizationStats(
        waypoint_mean=waypoint_mean,
        waypoint_std=waypoint_std,
        robot_state_mean=robot_mean,
        robot_state_std=robot_std,
    )


def infer_dims(batch: dict[str, Any], args: argparse.Namespace) -> tuple[int, int, int]:
    if args.use_asyncvla_projector:
        if "raw_action_embeddings" not in batch:
            raise KeyError("--use-asyncvla-projector requires raw_action_embeddings/actions_hidden_states in the batch.")
        context_dim = args.context_dim
        num_context_tokens = args.num_context_tokens
    elif "action_embeddings" in batch:
        context = batch["action_embeddings"]
        if context.ndim != 3:
            raise ValueError(f"Expected action_embeddings [B, N, D], got {tuple(context.shape)}")
        context_dim = args.context_dim or int(context.shape[-1])
        num_context_tokens = args.num_context_tokens or int(context.shape[-2])
    else:
        raise ValueError(
            "Batch has raw_action_embeddings but no projected action_embeddings. "
            "Pass --use-asyncvla-projector or precompute projected action_embeddings [B, 8, 1024]."
        )
    robot_state_dim = args.robot_state_dim
    if robot_state_dim is None:
        robot_state_dim = int(batch["robot_state"].shape[-1]) if "robot_state" in batch else 0
    return context_dim, num_context_tokens, robot_state_dim


def projector_config_from_args(args: argparse.Namespace) -> dict[str, int]:
    return {
        "input_dim": args.raw_context_dim,
        "hidden_dim": args.raw_context_dim,
        "action_dim": args.context_dim,
        "num_actions_chunk": args.num_context_tokens,
        "token_action_dim": args.raw_action_dim,
    }


def batch_context(
    batch: dict[str, Any],
    projector: Proj_Actiontokens | None,
    device: torch.device,
    default_modality_id: float,
    allow_projector_grad: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if projector is None:
        context = batch.get("action_embeddings")
        if context is None:
            raise KeyError("No action_embeddings context available for the MLP head.")
        modality_id = batch.get("modality_id")
        if modality_id is None:
            modality_id = torch.full((context.shape[0],), default_modality_id, device=device)
        return context, modality_id

    if "raw_action_embeddings" not in batch:
        raise KeyError("--use-asyncvla-projector requires raw_action_embeddings/actions_hidden_states in the batch.")
    raw_context = batch["raw_action_embeddings"]
    modality_id = batch.get("modality_id")
    if modality_id is None:
        modality_id = torch.full((raw_context.shape[0],), default_modality_id, device=device)
    if allow_projector_grad:
        context = projector.predict_action(raw_context, modality_id)
    else:
        with torch.no_grad():
            context = projector.predict_action(raw_context, modality_id)
    return context, modality_id


def run_epoch(
    model: MLPWaypointHead,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    loss_type: str,
    default_modality_id: float,
    projector: Proj_Actiontokens | None = None,
    trainable_modules: list[torch.nn.Module] | None = None,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    projector_has_grad = projector is not None and any(param.requires_grad for param in projector.parameters())
    if projector is not None:
        projector.train(is_train and projector_has_grad)
    total_loss = 0.0
    total_l1 = 0.0
    total_rmse = 0.0
    total_items = 0

    for batch in tqdm(loader, desc="train" if is_train else "val", leave=False):
        batch = move_batch(batch, device)
        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            context, modality_id = batch_context(
                batch,
                projector=projector,
                device=device,
                default_modality_id=default_modality_id,
                allow_projector_grad=is_train and projector_has_grad,
            )
            loss, metrics = model.loss(
                target_waypoints=batch["waypoints"],
                context=context,
                robot_state=batch.get("robot_state"),
                modality_id=modality_id,
                loss_type=loss_type,
            )
            if is_train:
                loss.backward()
                clip_params = []
                for module in trainable_modules or [model]:
                    clip_params.extend(param for param in module.parameters() if param.requires_grad)
                torch.nn.utils.clip_grad_norm_(clip_params, max_norm=1.0)
                optimizer.step()

        batch_size = batch["waypoints"].shape[0]
        total_loss += float(metrics["loss"].detach().cpu()) * batch_size
        total_l1 += float(metrics["l1"].detach().cpu()) * batch_size
        total_rmse += float(metrics["rmse"].detach().cpu()) * batch_size
        total_items += batch_size

    return {
        "loss": total_loss / max(total_items, 1),
        "l1": total_l1 / max(total_items, 1),
        "rmse": total_rmse / max(total_items, 1),
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    base_dataset = TrajectoryEmbeddingDataset(
        args.data,
        horizon=args.horizon,
        waypoint_dim=args.waypoint_dim,
        normalize_waypoints=False,
        normalize_robot_state=False,
        cache_trajectories=args.cache_trajectories,
    )
    split_payload = load_split_file(args.split_json) if args.split_json is not None else None
    if split_payload is not None:
        train_indices = dataset_indices_for_split(base_dataset, split_payload, args.train_split)
        val_indices = dataset_indices_for_split(base_dataset, split_payload, args.val_split)
    else:
        train_indices, val_indices = split_indices(len(base_dataset), args.val_fraction, args.seed)

    stats = None
    if args.normalize_waypoints or args.normalize_robot_state:
        if args.stats_path is not None and args.stats_path.exists():
            stats = NormalizationStats.from_file(args.stats_path)
        else:
            stats = compute_subset_stats(Subset(base_dataset, train_indices))
            stats.save(args.output_dir / "normalization_stats.json")

    dataset = TrajectoryEmbeddingDataset(
        args.data,
        horizon=args.horizon,
        waypoint_dim=args.waypoint_dim,
        normalize_waypoints=args.normalize_waypoints,
        normalize_robot_state=args.normalize_robot_state,
        stats=stats,
        cache_trajectories=args.cache_trajectories,
    )
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices) if val_indices else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_samples,
    )
    first_batch = next(iter(train_loader))
    context_dim, num_context_tokens, robot_state_dim = infer_dims(first_batch, args)

    config = MLPWaypointHeadConfig(
        context_dim=context_dim,
        num_context_tokens=num_context_tokens,
        horizon=args.horizon,
        waypoint_dim=args.waypoint_dim,
        hidden_dim=args.hidden_dim,
        num_blocks=args.num_blocks,
        dropout=args.dropout,
        robot_state_dim=robot_state_dim,
        use_modality_id=not args.no_modality_id,
        max_forward_distance=args.max_forward_distance,
        max_lateral_distance=args.max_lateral_distance,
        max_yaw_change=args.max_yaw_change,
    )
    with (args.output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=2)

    device = choose_device(args.device)
    model = MLPWaypointHead(config).to(device)
    projector = None
    projector_config = None
    if args.use_asyncvla_projector:
        if args.projector_checkpoint is None and not args.train_projector:
            raise ValueError(
                "--use-asyncvla-projector without --projector-checkpoint creates a random projector. "
                "Pass --train-projector to train it from scratch, or provide --projector-checkpoint to freeze/load one."
            )
        projector_config = projector_config_from_args(args)
        projector = Proj_Actiontokens(**projector_config).to(device)
        if args.projector_checkpoint is not None:
            load_projector_state(projector, args.projector_checkpoint, strict=False)
        for param in projector.parameters():
            param.requires_grad = args.train_projector
        projector.train(args.train_projector)

    trainable_modules = [model]
    trainable_params = list(model.parameters())
    if projector is not None and args.train_projector:
        trainable_modules.append(projector)
        trainable_params.extend(projector.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate_samples,
        )

    print(
        f"Training MLPWaypointHead with context=[N={num_context_tokens}, D={context_dim}], "
        f"horizon={args.horizon}, waypoint_dim={args.waypoint_dim}, robot_state_dim={robot_state_dim}, "
        f"hidden_dim={args.hidden_dim}, loss={args.loss_type}, "
        f"use_projector={args.use_asyncvla_projector}, train_projector={args.train_projector}, device={device}"
    )

    best_val = float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            args.loss_type,
            default_modality_id=args.default_modality_id,
            projector=projector,
            trainable_modules=trainable_modules,
        )
        val_metrics = (
            run_epoch(
                model,
                val_loader,
                None,
                device,
                args.loss_type,
                default_modality_id=args.default_modality_id,
                projector=projector,
            )
            if val_loader is not None
            else {}
        )
        record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(record)
        with (args.output_dir / "metrics.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        text = (
            f"epoch={epoch:03d} train_loss={train_metrics['loss']:.6f} "
            f"train_l1={train_metrics['l1']:.6f} train_rmse={train_metrics['rmse']:.6f}"
        )
        if val_metrics:
            text += (
                f" val_loss={val_metrics['loss']:.6f} "
                f"val_l1={val_metrics['l1']:.6f} val_rmse={val_metrics['rmse']:.6f}"
            )
        print(text)

        checkpoint = {
            "model": model.state_dict(),
            "config": asdict(config),
            "epoch": epoch,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "normalization_stats": stats.to_dict() if stats is not None else None,
            "projector": projector.state_dict() if projector is not None else None,
            "projector_config": projector_config,
            "use_asyncvla_projector": args.use_asyncvla_projector,
            "train_projector": args.train_projector,
            "args": vars(args) | {"data": str(args.data), "output_dir": str(args.output_dir)},
            "split_json": str(args.split_json) if args.split_json is not None else None,
            "train_split": args.train_split,
            "val_split": args.val_split,
        }
        torch.save(checkpoint, args.output_dir / "last.pt")
        score = val_metrics.get("loss", train_metrics["loss"])
        if score < best_val:
            best_val = score
            torch.save(checkpoint, args.output_dir / "best.pt")


if __name__ == "__main__":
    main()
