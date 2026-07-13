from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen


BASE_URL = "https://rail.eecs.berkeley.edu/datasets/huron"

BALANCED_FOLDERS = [
    "Dec-06-2022-bww8",
    "Dec-09-2022-bww8",
    "Feb-03-2023-bww8-intloss",
    "Feb-06-2023-bww8-intloss",
    "Feb-15-2023-bww1",
    "Feb-16-2023-bww1-intloss",
    "Feb-15-2023-cory1",
    "Feb-16-2023-cory1-intloss",
    "Feb-17-2023-bww2",
    "Feb-20-2023-bww2-intloss",
    "Feb-17-2023-soda3",
    "Feb-23-2023-soda3-intloss",
]

PILOT_FOLDERS = [
    "Feb-15-2023-cory1",
    "Feb-16-2023-cory1-intloss",
]


@dataclass
class BagItem:
    folder: str
    name: str
    url: str
    size_text: str
    size_bytes: int
    interaction_loss: bool
    local_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a balanced capped HuRoN subset from WSL.")
    parser.add_argument("--out-root", type=Path, default=Path("/mnt/d/Capstone/vla_datasets/huron"))
    parser.add_argument("--mode", choices=["balanced", "pilot", "full"], default="balanced")
    parser.add_argument("--max-bags-per-folder", type=int, default=10)
    parser.add_argument("--max-total-gb", type=float, default=25.0)
    parser.add_argument("--min-bag-mb", type=int, default=30)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def fetch_text(url: str) -> str:
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=60) as response:
        return response.read().decode("utf-8", errors="replace")


def size_to_bytes(text: str) -> int | None:
    match = re.match(r"^([0-9]+(?:\.[0-9]+)?)([KMG])$", text.strip())
    if not match:
        return None
    value = float(match.group(1))
    unit = match.group(2)
    scale = {"K": 1024, "M": 1024**2, "G": 1024**3}[unit]
    return int(value * scale)


def directory_folders() -> list[str]:
    html = fetch_text(BASE_URL + "/")
    folders = []
    for href in re.findall(r'href="([^"]+/)"', html):
        name = href.strip("/")
        if name and name != "..":
            folders.append(name)
    return sorted(set(folders))


def folder_bags(folder: str, out_root: Path, min_bag_mb: int, max_bags: int) -> list[BagItem]:
    url = f"{BASE_URL}/{folder}/"
    html = fetch_text(url)
    pattern = re.compile(
        r'href="([^"]+\.bag)"[^>]*>[^<]+</a>.*?'
        r'([0-9]{4}-[0-9]{2}-[0-9]{2}\s+[0-9]{2}:[0-9]{2}).*?'
        r'<td[^>]*>\s*([0-9.]+[KMG])\s*</td>',
        re.DOTALL,
    )
    bags: list[BagItem] = []
    for name, _date, size_text in pattern.findall(html):
        size_bytes = size_to_bytes(size_text)
        if size_bytes is None or size_bytes < min_bag_mb * 1024**2:
            continue
        bags.append(
            BagItem(
                folder=folder,
                name=name,
                url=f"{url}{name}",
                size_text=size_text,
                size_bytes=size_bytes,
                interaction_loss="intloss" in folder,
                local_path=str(out_root / "raw" / folder / name),
            )
        )
    bags.sort(key=lambda item: item.name)
    return bags[:max_bags] if max_bags > 0 else bags


def build_plan(args: argparse.Namespace) -> list[BagItem]:
    if args.mode == "full":
        folders = directory_folders()
    elif args.mode == "pilot":
        folders = PILOT_FOLDERS
    else:
        folders = BALANCED_FOLDERS

    print("Folders:")
    for folder in folders:
        print(f"  {folder}")

    candidates: list[BagItem] = []
    for folder in folders:
        print(f"Indexing {BASE_URL}/{folder}/")
        candidates.extend(folder_bags(folder, args.out_root, args.min_bag_mb, args.max_bags_per_folder))

    selected: list[BagItem] = []
    running = 0
    max_bytes = int(args.max_total_gb * 1024**3)
    for item in candidates:
        if running + item.size_bytes > max_bytes:
            print(f"Skipping due to cap: {item.folder}/{item.name}")
            continue
        selected.append(item)
        running += item.size_bytes
    return selected


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def looks_like_html(path: Path) -> bool:
    prefix = path.read_bytes()[:512].decode("utf-8", errors="ignore").lower()
    return any(token in prefix for token in ("<html", "<!doctype html", "login", "sign in", "not found", "forbidden"))


def download(item: BagItem) -> None:
    dest = Path(item.local_path)
    part = dest.with_name(dest.name + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and dest.stat().st_size == item.size_bytes and not looks_like_html(dest):
        print(f"Already complete: {dest}")
        return

    print(f"Downloading {item.folder}/{item.name} [{item.size_text}]")
    cmd = ["curl", "-L", "-C", "-", "-o", str(part), item.url]
    subprocess.run(cmd, check=True)
    if looks_like_html(part):
        raise RuntimeError(f"Downloaded content looks like HTML/error page: {part}")
    part.replace(dest)


def write_plan(args: argparse.Namespace, selected: list[BagItem]) -> None:
    manifests = args.out_root / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    rows = [asdict(item) for item in selected]

    csv_path = manifests / "huron-balanced-download-plan.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["folder"])
        writer.writeheader()
        writer.writerows(rows)

    total = sum(item.size_bytes for item in selected)
    json_path = manifests / "huron-balanced-download-plan.json"
    json_path.write_text(
        json.dumps(
            {
                "created_utc": now_utc(),
                "base_url": BASE_URL,
                "mode": args.mode,
                "max_bags_per_folder": args.max_bags_per_folder,
                "max_total_gb": args.max_total_gb,
                "min_bag_mb": args.min_bag_mb,
                "selected_count": len(selected),
                "selected_size_gb": round(total / 1024**3, 3),
                "bags": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Plan CSV: {csv_path}")
    print(f"Plan JSON: {json_path}")


def main() -> None:
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(args.out_root)
    free_gb = usage.free / 1024**3
    print(f"HuRoN output root: {args.out_root}")
    print(f"Free space: {free_gb:.2f} GB")
    if free_gb < args.max_total_gb + 5:
        raise SystemExit(f"Not enough free space for requested cap. Need at least {args.max_total_gb + 5:.1f} GB.")

    selected = build_plan(args)
    total_gb = sum(item.size_bytes for item in selected) / 1024**3
    print(f"\nSelected {len(selected)} bags, estimated size {total_gb:.2f} GB")
    write_plan(args, selected)

    if args.dry_run:
        print("Dry run only. No files downloaded.")
        return

    sha_path = args.out_root / "manifests" / "huron-balanced-sha256.txt"
    manifest_items = []
    for item in selected:
        download(item)
        dest = Path(item.local_path)
        digest = sha256(dest)
        with sha_path.open("a", encoding="ascii") as f:
            f.write(f"{digest}  {dest}\n")
        payload = asdict(item)
        payload["sha256"] = digest
        payload["actual_size_bytes"] = dest.stat().st_size
        manifest_items.append(payload)

    manifest_path = args.out_root / "manifests" / "huron-balanced-download-manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "created_utc": now_utc(),
                "base_url": BASE_URL,
                "mode": args.mode,
                "downloaded_count": len(manifest_items),
                "items": manifest_items,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nDone. Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
