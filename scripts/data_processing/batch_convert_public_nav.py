from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch convert public navigation datasets into processed_mixed format.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("Missing pyyaml. Install with: python3 -m pip install pyyaml") from exc
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def run_command(command: list[str]) -> tuple[bool, str]:
    print("\n" + " ".join(command))
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    return result.returncode == 0, result.stderr[-4000:]


def add_if_present(command: list[str], flag: str, cfg: dict[str, Any], key: str) -> None:
    if key in cfg and cfg[key] is not None:
        command.extend([flag, str(cfg[key])])


def scand_commands(name: str, cfg: dict[str, Any], overwrite: bool) -> list[list[str]]:
    bags_dir = Path(cfg.get("bags_dir", "bags"))
    pattern = cfg.get("pattern", "A_Jackal_*.bag")
    out_root = Path(cfg.get("out_root", f"processed_mixed/{name}"))
    bags = sorted((ROOT / bags_dir).glob(pattern) if not bags_dir.is_absolute() else bags_dir.glob(pattern))
    commands = []
    for bag in bags:
        out_dir = out_root / bag.stem
        if (ROOT / out_dir / "trajectory.npz").exists() and not overwrite:
            print(f"Skipping already processed bag: {out_dir}")
            continue
        command = [
            sys.executable,
            "scripts/data_processing/convert_scand_bag.py",
            "--bag",
            str(bag),
            "--image-topic",
            str(cfg["image_topic"]),
            "--odom-topic",
            str(cfg["odom_topic"]),
            "--out-dir",
            str(out_dir),
        ]
        for key in ["horizon", "waypoint_dt", "image_stride", "sync_threshold", "max_final_distance", "max_abs_yaw"]:
            add_if_present(command, "--" + key.replace("_", "-"), cfg, key)
        commands.append(command)
    return commands


def converter_command(script: str, cfg: dict[str, Any], overwrite: bool) -> list[str]:
    command = [sys.executable, script]
    add_if_present(command, "--input-root", cfg, "input_root")
    add_if_present(command, "--selection-json", cfg, "selection_json")
    add_if_present(command, "--out-root", cfg, "out_root")
    add_if_present(command, "--dataset-name", cfg, "dataset_name")
    for key in [
        "horizon",
        "waypoint_dt",
        "sampling_mode",
        "target_spacing_m",
        "distance_tolerance_m",
        "frame_stride",
        "frame_dt",
        "min_final_distance",
        "max_final_distance",
        "max_abs_yaw",
    ]:
        add_if_present(command, "--" + key.replace("_", "-"), cfg, key)
    if overwrite:
        command.append("--overwrite")
    return command


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    datasets = config.get("datasets", {})
    if not datasets:
        raise SystemExit(f"No datasets found in {args.config}")

    processed = []
    failed = []
    for name, cfg in datasets.items():
        dtype = cfg.get("type")
        try:
            if dtype == "rosbag":
                commands = scand_commands(name, cfg, args.overwrite)
            elif dtype == "recon":
                commands = [converter_command("scripts/data_processing/convert_recon.py", cfg | {"dataset_name": name}, args.overwrite)]
            else:
                raise ValueError(f"Unsupported dataset type for {name}: {dtype}. Supported types are rosbag and recon.")

            for command in commands:
                ok, error = run_command(command)
                if ok:
                    processed.append({"dataset": name, "command": command})
                else:
                    failed.append({"dataset": name, "command": command, "error": error})
        except Exception as exc:
            failed.append({"dataset": name, "error": str(exc)})
            print(f"FAILED {name}: {exc}")

    total_samples = 0
    out_root = ROOT / "processed_mixed"
    for npz in out_root.glob("*/*/trajectory.npz"):
        try:
            import numpy as np

            total_samples += int(len(np.load(npz, allow_pickle=True)["target_waypoints"]))
        except Exception:
            pass

    summary = {"processed": processed, "failed": failed, "total_samples": total_samples}
    out_root.mkdir(parents=True, exist_ok=True)
    summary_path = out_root / "batch_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {summary_path}")
    print(f"Processed commands: {len(processed)}")
    print(f"Failed commands: {len(failed)}")
    print(f"Total samples currently under processed_mixed: {total_samples}")


if __name__ == "__main__":
    main()
