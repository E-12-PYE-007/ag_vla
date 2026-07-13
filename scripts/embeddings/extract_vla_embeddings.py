from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract AsyncVLA projected and raw action-token embeddings for processed SCAND/RECON trajectories."
    )
    parser.add_argument("--processed-root", type=Path, default=Path("processed_mixed"))
    parser.add_argument("--npz", type=Path, action="append", help="Specific trajectory.npz file. Can be passed more than once.")
    parser.add_argument("--asyncvla-root", type=Path, default=ROOT / "external" / "AsyncVLA-main")
    parser.add_argument("--mbra-root", type=Path, default=None, help="Optional Learning-to-Drive-Anywhere-with-MBRA repo root.")
    parser.add_argument("--lerobot-root", type=Path, default=None, help="Optional lerobot repo root.")
    parser.add_argument("--vla-path", type=Path, required=True, help="AsyncVLA_release checkpoint/model directory.")
    parser.add_argument("--action-proj-checkpoint", type=Path, default=None)
    parser.add_argument("--resume-step", type=int, default=750000)
    parser.add_argument(
        "--raw-only",
        action="store_true",
        help="Save raw action-token embeddings only. Does not load or require an action_proj checkpoint.",
    )
    parser.add_argument("--prompt", default="Continue safe navigation while avoiding obstacles.")
    parser.add_argument("--output-name", default="trajectory_with_embeddings.npz")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-trajectories", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--modality-id", type=float, default=7.0, help="AsyncVLA language-only modality id.")
    parser.add_argument(
        "--dummy-action-mode",
        choices=["random", "constant"],
        default="random",
        help="Defaults to random to match scripts/embeddings/run_vla.py's np.random.rand(8, 4) dummy actions.",
    )
    parser.add_argument("--dummy-action-value", type=float, default=0.0)
    parser.add_argument("--dummy-action-seed", type=int, default=None)
    parser.add_argument(
        "--num-images-in-input",
        type=int,
        default=2,
        help="Value passed to vla.vision_backbone.set_num_images_in_input. Defaults to match scripts/embeddings/run_vla.py.",
    )
    parser.add_argument(
        "--image-copies",
        type=int,
        default=2,
        help="Number of duplicated image slots placed in pixel_values. Defaults to match scripts/embeddings/run_vla.py.",
    )
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--save-raw-action-embeddings",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save raw action-token hidden states as raw_action_embeddings. Enabled by default.",
    )
    parser.add_argument(
        "--save-raw-action-hidden-states",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def configure_import_paths(args: argparse.Namespace) -> None:
    paths = [args.asyncvla_root]
    if args.mbra_root is not None:
        paths.append(args.mbra_root / "train")
    if args.lerobot_root is not None:
        paths.append(args.lerobot_root)
    for path in reversed(paths):
        if path is not None:
            sys.path.insert(0, str(path))


def choose_device(name: str):
    import torch

    if name != "auto":
        return torch.device(name)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def remove_ddp_prefix(state_dict: dict[str, Any]) -> dict[str, Any]:
    return {key[7:] if key.startswith("module.") else key: value for key, value in state_dict.items()}


def default_action_proj_checkpoint(vla_path: Path, resume_step: int) -> Path:
    return vla_path / f"action_proj--{resume_step}_checkpoint.pt"


def register_asyncvla_hf_classes() -> None:
    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction_MMNv1
    from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
    from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction_MMNv1)


def load_asyncvla(args: argparse.Namespace):
    import torch
    from prismatic.vla.action_tokenizer import ActionTokenizer
    from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction_MMNv1
    from transformers import AutoProcessor

    register_asyncvla_hf_classes()
    device = choose_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()

    processor = AutoProcessor.from_pretrained(args.vla_path, trust_remote_code=True)
    # Use the MMN AsyncVLA class directly. Some environments dispatch the
    # checkpoint through AutoModelForVision2Seq to the base Prismatic class,
    # whose forward() does not accept modality_id.
    vla = OpenVLAForActionPrediction_MMNv1.from_pretrained(
        args.vla_path,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device)
    print(f"Loaded VLA class: {vla.__class__.__name__}")
    vla.vision_backbone.set_num_images_in_input(args.num_images_in_input)
    if device.type == "cuda":
        vla.to(dtype=torch.bfloat16, device=device)
    vla.eval()

    action_proj = None
    if not args.raw_only:
        try:
            # Preferred path: use the exact AsyncVLA projector class used by scripts/embeddings/run_vla.py.
            from prismatic.models.small_head import Proj_Actiontokens
        except ImportError:
            # Fallback only for environments where AsyncVLA's small_head import is unavailable.
            from flow_head.asyncvla_projector import Proj_Actiontokens

        action_proj = Proj_Actiontokens(
            input_dim=vla.llm_dim,
            hidden_dim=vla.llm_dim,
            action_dim=1024,
        ).to(device)
        if device.type == "cuda":
            action_proj = action_proj.to(torch.bfloat16)
        checkpoint = args.action_proj_checkpoint or default_action_proj_checkpoint(args.vla_path, args.resume_step)
        load_action_projector_like_run_vla(action_proj, checkpoint, device)
        action_proj.eval()

    num_patches = vla.vision_backbone.get_num_patches() * vla.vision_backbone.get_num_images_in_input()
    action_tokenizer = ActionTokenizer(processor.tokenizer)
    return vla, action_proj, processor, action_tokenizer, device, num_patches


def load_action_projector_like_run_vla(action_proj: Any, checkpoint_path: Path, device: Any) -> None:
    import torch
    from flow_head.asyncvla_projector import load_projector_state

    print(f"Loading checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device)
    if isinstance(state_dict, dict) and all(hasattr(value, "shape") for value in state_dict.values()):
        state_dict = remove_ddp_prefix(state_dict)
        load_result = action_proj.load_state_dict(state_dict, strict=False)
        print(f"action_proj load_state_dict missing_keys: {load_result.missing_keys}")
        print(f"action_proj load_state_dict unexpected_keys: {load_result.unexpected_keys}")
        return

    # Some checkpoints are wrapped in a top-level key. This fallback is the only non-run_vla
    # behavior here, and exists so the offline extractor can still resume from common checkpoint formats.
    load_projector_state(action_proj, checkpoint_path, strict=False)


def make_dummy_actions(mode: str, value: float) -> np.ndarray:
    from prismatic.vla.constants import NUM_ACTIONS_CHUNK, ACTION_DIM

    if mode == "random":
        # Match scripts/embeddings/run_vla.py: actions = np.random.rand(8, 4)
        return np.random.rand(NUM_ACTIONS_CHUNK, ACTION_DIM).astype(np.float32)
    return np.full((NUM_ACTIONS_CHUNK, ACTION_DIM), value, dtype=np.float32)


def make_prompt_batch(
    images: list[Image.Image],
    prompt: str,
    action_tokenizer: Any,
    processor: Any,
    dummy_action_mode: str,
    dummy_action_value: float,
    image_copies: int,
):
    import torch
    from prismatic.models.backbones.llm.prompting import PurePromptBuilder
    from torch.nn.utils.rnn import pad_sequence

    ignore_index = -100
    instances = []

    for image in images:
        dummy_actions = make_dummy_actions(dummy_action_mode, dummy_action_value)
        current_action = dummy_actions[0]
        future_actions = dummy_actions[1:]
        action_chunk_string = action_tokenizer(current_action) + "".join(action_tokenizer(future_actions))
        action_chunk_len = len(action_chunk_string)

        prompt_builder = PurePromptBuilder("openvla")
        prompt_builder.add_turn("human", f"What action should the robot take to {prompt}?")
        prompt_builder.add_turn("gpt", action_chunk_string)

        input_ids = torch.tensor(processor.tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids)
        labels = input_ids.clone()
        labels[: -(action_chunk_len + 1)] = ignore_index
        pixel_values_current = processor.image_processor.apply_transform(image)
        instances.append(
            {
                "input_ids": input_ids,
                "labels": labels,
                "pixel_values_current": pixel_values_current,
            }
        )

    input_ids = pad_sequence(
        [item["input_ids"] for item in instances],
        batch_first=True,
        padding_value=processor.tokenizer.pad_token_id,
    )
    labels = pad_sequence([item["labels"] for item in instances], batch_first=True, padding_value=ignore_index)
    input_ids = input_ids[:, : processor.tokenizer.model_max_length]
    labels = labels[:, : processor.tokenizer.model_max_length]
    attention_mask = input_ids.ne(processor.tokenizer.pad_token_id)

    stacked_images = torch.stack([item["pixel_values_current"] for item in instances])
    pixel_values = torch.cat([stacked_images] * image_copies, dim=1)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "pixel_values": pixel_values,
    }


def extract_batch(
    vla: Any,
    action_proj: Any | None,
    processor: Any,
    action_tokenizer: Any,
    device: Any,
    num_patches: int,
    images: list[Image.Image],
    prompt: str,
    modality_id_value: float,
    dummy_action_mode: str,
    dummy_action_value: float,
    image_copies: int,
) -> tuple[np.ndarray | None, np.ndarray]:
    import torch
    from prismatic.training.train_utils import get_current_action_mask, get_next_actions_mask
    from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK

    batch = make_prompt_batch(
        images,
        prompt,
        action_tokenizer,
        processor,
        dummy_action_mode,
        dummy_action_value,
        image_copies,
    )
    modality_id = torch.full((len(images),), modality_id_value, dtype=torch.float32, device=device)
    autocast_enabled = device.type == "cuda"

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
        output = vla(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            pixel_values=batch["pixel_values"].to(device=device, dtype=torch.bfloat16 if device.type == "cuda" else torch.float32),
            modality_id=modality_id.to(dtype=torch.bfloat16 if device.type == "cuda" else torch.float32),
            labels=batch["labels"].to(device),
            output_hidden_states=True,
            noisy_actions=None,
            noisy_action_projector=None,
            diffusion_timestep_embeddings=None,
            use_film=False,
        )

        ground_truth_token_ids = batch["labels"][:, 1:].to(device)
        current_action_mask = get_current_action_mask(ground_truth_token_ids)
        next_actions_mask = get_next_actions_mask(ground_truth_token_ids)
        text_hidden_states = output.hidden_states[-1][:, num_patches:-1]
        batch_size = batch["input_ids"].shape[0]
        actions_hidden_states = (
            text_hidden_states[current_action_mask | next_actions_mask]
            .reshape(batch_size, NUM_ACTIONS_CHUNK * ACTION_DIM, -1)
            .to(torch.bfloat16 if device.type == "cuda" else torch.float32)
        )
        projected_actions = None
        if action_proj is not None:
            projected_actions = action_proj.predict_action(
                actions_hidden_states.detach(),
                modality_id.to(dtype=torch.bfloat16 if device.type == "cuda" else torch.float32),
            )

    return (
        projected_actions.detach().to(torch.float32).cpu().numpy() if projected_actions is not None else None,
        actions_hidden_states.detach().to(torch.float32).cpu().numpy(),
    )


def find_trajectory_npzs(args: argparse.Namespace) -> list[Path]:
    if args.npz:
        files = [path.resolve() for path in args.npz]
    else:
        files = sorted(args.processed_root.glob("*/*/trajectory.npz"))
    if args.max_trajectories is not None:
        files = files[: args.max_trajectories]
    return files


def resolve_image_path(npz_path: Path, image_path: str) -> Path:
    path = Path(str(image_path))
    if path.is_absolute():
        return path
    return npz_path.parent / path


def copy_npz_payload(data: np.lib.npyio.NpzFile) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key in data.files:
        payload[key] = data[key]
    return payload


def extract_trajectory(npz_path: Path, args: argparse.Namespace, model_parts: tuple[Any, ...]) -> None:
    vla, action_proj, processor, action_tokenizer, device, num_patches = model_parts
    out_path = npz_path.parent / args.output_name
    if out_path.exists() and not args.overwrite:
        print(f"Skipping existing embeddings: {out_path}")
        return

    with np.load(npz_path, allow_pickle=True) as data:
        payload = copy_npz_payload(data)
        image_paths = [resolve_image_path(npz_path, item) for item in data["image_paths"]]
        if args.max_samples is not None:
            image_paths = image_paths[: args.max_samples]
            for key, value in list(payload.items()):
                if isinstance(value, np.ndarray) and len(value.shape) > 0 and value.shape[0] >= len(data["image_paths"]):
                    payload[key] = value[: args.max_samples]

    save_raw_action_embeddings = args.save_raw_action_embeddings or args.save_raw_action_hidden_states
    projected_chunks = []
    raw_chunks = []
    for start in range(0, len(image_paths), args.batch_size):
        batch_paths = image_paths[start : start + args.batch_size]
        images = [Image.open(path).convert("RGB") for path in batch_paths]
        projected, raw_hidden = extract_batch(
            vla=vla,
            action_proj=action_proj,
            processor=processor,
            action_tokenizer=action_tokenizer,
            device=device,
            num_patches=num_patches,
            images=images,
            prompt=args.prompt,
            modality_id_value=args.modality_id,
            dummy_action_mode=args.dummy_action_mode,
            dummy_action_value=args.dummy_action_value,
            image_copies=args.image_copies,
        )
        if projected is not None:
            projected_chunks.append(projected)
        if save_raw_action_embeddings:
            raw_chunks.append(raw_hidden)
        print(f"{npz_path.parent.name}: {min(start + len(batch_paths), len(image_paths))}/{len(image_paths)}")

    projected_actions = None
    if projected_chunks:
        projected_actions = np.concatenate(projected_chunks, axis=0).astype(np.float32)
        payload["projected_actions"] = projected_actions
        payload["action_embeddings"] = projected_actions
    if "velocity" in payload:
        payload["robot_state"] = np.asarray(payload["velocity"], dtype=np.float32)
    raw_action_embeddings = None
    if save_raw_action_embeddings:
        raw_action_embeddings = np.concatenate(raw_chunks, axis=0).astype(np.float32)
        payload["raw_action_embeddings"] = raw_action_embeddings
    payload["embedding_prompt"] = np.asarray(args.prompt)
    payload["embedding_modality_id"] = np.asarray(args.modality_id, dtype=np.float32)

    np.savez_compressed(out_path, **payload)
    metadata_path = npz_path.parent / "embedding_metadata.json"
    metadata = {
        "source_npz": str(npz_path),
        "output_npz": str(out_path),
        "prompt": args.prompt,
        "modality_id": args.modality_id,
        "dummy_action_mode": args.dummy_action_mode,
        "dummy_action_value": args.dummy_action_value,
        "num_images_in_input": args.num_images_in_input,
        "image_copies": args.image_copies,
        "raw_only": args.raw_only,
        "projected_actions_shape": list(projected_actions.shape) if projected_actions is not None else None,
        "raw_action_embeddings_shape": list(raw_action_embeddings.shape) if raw_action_embeddings is not None else None,
        "save_raw_action_embeddings": save_raw_action_embeddings,
        "asyncvla_root": str(args.asyncvla_root),
        "vla_path": str(args.vla_path),
        "action_proj_checkpoint": (
            None if args.raw_only else str(args.action_proj_checkpoint or default_action_proj_checkpoint(args.vla_path, args.resume_step))
        ),
    }
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved {out_path}")


def main() -> None:
    args = parse_args()
    if args.dummy_action_seed is not None:
        np.random.seed(args.dummy_action_seed)
    configure_import_paths(args)
    model_parts = load_asyncvla(args)
    npz_files = find_trajectory_npzs(args)
    if not npz_files:
        raise SystemExit("No trajectory.npz files found.")
    for npz_path in npz_files:
        extract_trajectory(npz_path, args, model_parts)


if __name__ == "__main__":
    main()
