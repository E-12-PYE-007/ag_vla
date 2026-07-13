from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from flow_head.go_stanford_conversion import discover_sequences, write_sequence_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect extracted GO Stanford 2 list files.")
    parser.add_argument("--root", type=Path, required=True, help="Extracted gs2_withres directory.")
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--include-flipped", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sequences = discover_sequences(args.root)
    if not args.include_flipped:
        shown = [seq for seq in sequences if not seq.is_flipped]
    else:
        shown = sequences
    print(f"root: {args.root}")
    print(f"sequences: {len(sequences)}")
    print(f"non_flipped: {sum(1 for seq in sequences if not seq.is_flipped)}")
    print(f"flipped: {sum(1 for seq in sequences if seq.is_flipped)}")
    print("\nFirst sequences:")
    for seq in shown[:25]:
        print(
            f"  {seq.sequence_id}: "
            f"L={seq.num_left} R={seq.num_right} res={seq.num_results} flipped={seq.is_flipped}"
        )
    mismatched = [
        seq
        for seq in shown
        if seq.num_results == 0
        or (seq.num_left and seq.num_left != seq.num_results)
        or (seq.num_right and seq.num_right != seq.num_results)
    ]
    print(f"\nMismatched/nonempty warnings: {len(mismatched)}")
    for seq in mismatched[:20]:
        print(f"  {seq.sequence_id}: L={seq.num_left} R={seq.num_right} res={seq.num_results}")
    if args.out_json:
        write_sequence_summary(args.out_json, sequences)
        print(f"\nSaved {args.out_json}")


if __name__ == "__main__":
    main()
