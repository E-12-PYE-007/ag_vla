from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_split_file(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if "splits" not in payload:
        raise ValueError(f"{path} is missing top-level 'splits'.")
    return payload


def normalize_path(path: str | Path) -> str:
    return Path(path).as_posix()


def split_file_set(split_payload: dict[str, Any], split_name: str, data_root: str | Path | None = None) -> set[str]:
    splits = split_payload.get("splits", {})
    if split_name not in splits:
        raise ValueError(f"Split {split_name!r} not found. Available: {sorted(splits)}")
    data_root_path = Path(data_root).resolve() if data_root is not None else None
    values = set()
    for item in splits[split_name]:
        path = Path(item)
        values.add(normalize_path(path))
        if data_root_path is not None and not path.is_absolute():
            values.add(normalize_path((data_root_path / path).resolve()))
    return values


def dataset_indices_for_split(dataset: Any, split_payload: dict[str, Any], split_name: str) -> list[int]:
    allowed = split_file_set(split_payload, split_name, data_root=dataset.data_path)
    indices = []
    for sample_idx, (file_idx, _timestep) in enumerate(dataset.index):
        file_path = dataset.files[file_idx]
        candidates = {normalize_path(file_path), normalize_path(file_path.resolve())}
        try:
            candidates.add(normalize_path(file_path.relative_to(dataset.data_path)))
        except ValueError:
            pass
        if candidates & allowed:
            indices.append(sample_idx)
    if not indices:
        raise ValueError(f"Split {split_name!r} produced zero samples for dataset root {dataset.data_path}")
    return indices
