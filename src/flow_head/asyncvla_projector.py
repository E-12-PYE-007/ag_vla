from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn


class MLPResNetBlock(nn.Module):
    """One MLP ResNet block with a residual connection, matching AsyncVLA."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.ffn(x) + x


class MLPResNetIDCat(nn.Module):
    """AsyncVLA projector MLP that concatenates scalar task/modality id."""

    def __init__(self, num_blocks: int, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim + 1, hidden_dim)
        self.relu = nn.ReLU()
        self.mlp_resnet_blocks = nn.ModuleList(
            [MLPResNetBlock(dim=hidden_dim) for _ in range(num_blocks)]
        )
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: Tensor, taskid: Tensor) -> Tensor:
        x = self.layer_norm1(x)
        if taskid.ndim == 0:
            taskid = taskid.expand(x.shape[0])
        if taskid.ndim != 1:
            taskid = taskid.reshape(x.shape[0])
        taskid = taskid.to(device=x.device, dtype=x.dtype)
        taskid_token = taskid[:, None, None].repeat(1, x.shape[1], 1)
        x = torch.cat((x, taskid_token), dim=2)
        x = self.relu(self.fc1(x))
        for block in self.mlp_resnet_blocks:
            x = block(x)
        return self.fc2(self.layer_norm2(x))


class Proj_Actiontokens(nn.Module):
    """
    Local copy of AsyncVLA's Proj_Actiontokens interface.

    AsyncVLA uses raw final-layer action-token hidden states:
        [B, NUM_ACTIONS_CHUNK * ACTION_DIM, input_dim]

    For navigation defaults:
        [B, 32, 4096] -> reshape [B, 8, 4 * 4096]
        concatenate taskid/modality_id -> [B, 8, 4 * 4096 + 1]
        project -> [B, 8, 1024]
    """

    def __init__(
        self,
        input_dim: int = 4096,
        hidden_dim: int = 4096,
        action_dim: int = 1024,
        num_actions_chunk: int = 8,
        token_action_dim: int = 4,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.num_actions_chunk = num_actions_chunk
        self.token_action_dim = token_action_dim
        self.input_dim = input_dim
        self.model = MLPResNetIDCat(
            num_blocks=2,
            input_dim=input_dim * token_action_dim,
            hidden_dim=hidden_dim,
            output_dim=action_dim,
        )

    def predict_action(self, actions_hidden_states: Tensor, taskid: Tensor) -> Tensor:
        batch_size = actions_hidden_states.shape[0]
        if actions_hidden_states.ndim == 4:
            actions_hidden_states = actions_hidden_states.reshape(
                batch_size,
                self.num_actions_chunk * self.token_action_dim,
                self.input_dim,
            )
        expected_shape = (self.num_actions_chunk * self.token_action_dim, self.input_dim)
        if tuple(actions_hidden_states.shape[1:]) != expected_shape:
            raise ValueError(
                f"Expected actions_hidden_states [B, {expected_shape[0]}, {expected_shape[1]}], "
                f"got {tuple(actions_hidden_states.shape)}"
            )
        rearranged_actions_hidden_states = actions_hidden_states.reshape(
            batch_size,
            self.num_actions_chunk,
            -1,
        )
        return self.model(rearranged_actions_hidden_states, taskid)


AsyncVLAActionProjector = Proj_Actiontokens


def load_projector_state(projector: nn.Module, checkpoint_path: str | Path, strict: bool = False) -> None:
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    print(f"[Projector] Loaded checkpoint file: {checkpoint_path}")
    print(
        "[Projector] Top-level checkpoint keys: "
        f"{list(checkpoint.keys()) if isinstance(checkpoint, dict) else type(checkpoint)}"
    )

    candidate_keys = ("action_proj", "projector", "action_projector", "state_dict", "model")
    raw_state = None

    if isinstance(checkpoint, dict):
        for key in candidate_keys:
            if key in checkpoint and isinstance(checkpoint[key], dict):
                raw_state = checkpoint[key]
                print(f"[Projector] Using checkpoint subkey: {key}")
                break
        if raw_state is None:
            raw_state = checkpoint
            print("[Projector] No known subkey found; using full checkpoint as state dict.")
    else:
        raise ValueError("Checkpoint must be a dict-like object.")

    own_keys = set(projector.state_dict().keys())
    state_dict = {}
    prefixes = ("module.", "action_proj.", "projector.", "action_projector.")
    for key, value in raw_state.items():
        clean_key = key
        for prefix in prefixes:
            if clean_key.startswith(prefix):
                clean_key = clean_key[len(prefix) :]
        if clean_key in own_keys:
            local_value = projector.state_dict()[clean_key]
            if tuple(local_value.shape) == tuple(value.shape):
                state_dict[clean_key] = value
            else:
                print(
                    f"[Projector] Shape mismatch for {clean_key}: "
                    f"checkpoint {tuple(value.shape)} vs local {tuple(local_value.shape)}"
                )

    missing_keys = [key for key in projector.state_dict().keys() if key not in state_dict]
    unexpected_keys = []
    for key in raw_state.keys():
        clean_key = key
        for prefix in prefixes:
            if clean_key.startswith(prefix):
                clean_key = clean_key[len(prefix) :]
        if clean_key not in own_keys:
            unexpected_keys.append(key)

    print(f"[Projector] Matched keys: {len(state_dict)} / {len(projector.state_dict())}")
    print(f"[Projector] Missing local keys: {missing_keys}")
    print(f"[Projector] Unexpected checkpoint keys not used: {unexpected_keys[:20]}")
    print(f"[Projector] Example loaded keys: {list(state_dict.keys())[:10]}")

    if len(state_dict) == 0:
        raise RuntimeError(
            "No projector weights were loaded. Check checkpoint path, key prefixes, "
            "and whether this is actually the AsyncVLA action projector checkpoint."
        )

    load_result = projector.load_state_dict(state_dict, strict=strict)
    print(f"[Projector] load_state_dict missing_keys: {load_result.missing_keys}")
    print(f"[Projector] load_state_dict unexpected_keys: {load_result.unexpected_keys}")
    print("[Projector] State loaded.")
