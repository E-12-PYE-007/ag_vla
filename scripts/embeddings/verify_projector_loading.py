from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from flow_head.asyncvla_projector import Proj_Actiontokens, load_projector_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify AsyncVLA projector checkpoint loading.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input-dim", type=int, default=4096)
    parser.add_argument("--hidden-dim", type=int, default=4096)
    parser.add_argument("--action-dim", type=int, default=1024)
    parser.add_argument("--num-actions-chunk", type=int, default=8)
    parser.add_argument("--token-action-dim", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--modality-id", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    projector = Proj_Actiontokens(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        action_dim=args.action_dim,
        num_actions_chunk=args.num_actions_chunk,
        token_action_dim=args.token_action_dim,
    )

    before_state = {
        key: value.detach().cpu().clone()
        for key, value in projector.state_dict().items()
    }

    load_projector_state(projector, args.checkpoint, strict=False)

    after_state = projector.state_dict()
    changed = []
    unchanged = []
    for key in before_state:
        after_value = after_state[key].detach().cpu()
        if torch.allclose(before_state[key], after_value):
            unchanged.append(key)
        else:
            changed.append(key)

    print("=" * 80)
    print("PARAMETER CHANGE CHECK")
    print("=" * 80)
    print(f"Changed params: {len(changed)}")
    print(f"Unchanged params: {len(unchanged)}")
    print("Changed examples:", changed[:10])
    print("Unchanged examples:", unchanged[:10])
    if len(changed) == 0:
        raise RuntimeError("No parameters changed after loading checkpoint.")

    projector.eval()
    for param in projector.parameters():
        param.requires_grad = False

    actions_hidden_states = torch.randn(
        args.batch_size,
        args.num_actions_chunk * args.token_action_dim,
        args.input_dim,
    )
    modality_id = torch.full((args.batch_size,), args.modality_id)

    with torch.no_grad():
        projected_actions = projector.predict_action(actions_hidden_states, modality_id)

    print("=" * 80)
    print("FORWARD PASS CHECK")
    print("=" * 80)
    print("actions_hidden_states:", tuple(actions_hidden_states.shape))
    print("projected_actions:", tuple(projected_actions.shape))
    print("projected_actions dtype:", projected_actions.dtype)
    print("projected_actions mean:", projected_actions.mean().item())
    print("projected_actions std:", projected_actions.std().item())
    print("projected_actions min:", projected_actions.min().item())
    print("projected_actions max:", projected_actions.max().item())

    expected_shape = (args.batch_size, args.num_actions_chunk, args.action_dim)
    if tuple(projected_actions.shape) != expected_shape:
        raise RuntimeError(
            f"Wrong projected_actions shape. Expected {expected_shape}, got {tuple(projected_actions.shape)}"
        )
    if not torch.isfinite(projected_actions).all():
        raise RuntimeError("Projected actions contain NaN or Inf.")

    print("=" * 80)
    print("FREEZE CHECK")
    print("=" * 80)
    print("Any projector parameter requires grad:", any(param.requires_grad for param in projector.parameters()))

    print("=" * 80)
    print("SUCCESS: projector checkpoint loads and forward pass works.")
    print("=" * 80)


if __name__ == "__main__":
    main()

