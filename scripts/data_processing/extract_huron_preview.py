from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from flow_head.scand_conversion import decode_image_msg, load_records_from_bag, nearest_index, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract a small HuRoN image/odometry preview.")
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--image-topic", default="/fisheye_image/compressed")
    parser.add_argument("--odom-topic", default="/odometry")
    parser.add_argument("--num-frames", type=int, default=20)
    return parser.parse_args()


def contact_sheet(paths: list[Path], out_path: Path, thumb_width: int = 240) -> None:
    if not paths:
        raise ValueError("No preview images to place in contact sheet.")
    thumbs: list[Image.Image] = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        ratio = thumb_width / max(image.width, 1)
        thumb = image.resize((thumb_width, max(1, int(image.height * ratio))))
        thumbs.append(thumb)
    cols = min(5, len(thumbs))
    rows = int(np.ceil(len(thumbs) / cols))
    cell_h = max(img.height for img in thumbs) + 24
    sheet = Image.new("RGB", (cols * thumb_width, rows * cell_h), "white")
    draw = ImageDraw.Draw(sheet)
    for idx, thumb in enumerate(thumbs):
        x = (idx % cols) * thumb_width
        y = (idx // cols) * cell_h
        sheet.paste(thumb, (x, y))
        draw.text((x + 4, y + thumb.height + 4), str(idx), fill=(0, 0, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=95)


def main() -> None:
    args = parse_args()
    previews_dir = args.out_dir / "previews"
    metadata_dir = args.out_dir / "metadata"
    previews_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    image_records, odom_records, counts = load_records_from_bag(args.bag, args.image_topic, args.odom_topic)
    if not image_records:
        raise ValueError(f"No images found on {args.image_topic}")
    if not odom_records:
        raise ValueError(f"No odometry found on {args.odom_topic}")

    selected = np.linspace(0, len(image_records) - 1, min(args.num_frames, len(image_records)), dtype=int)
    odom_times = np.asarray([sample.time for sample in odom_records], dtype=np.float64)
    preview_paths: list[Path] = []
    rows: list[dict[str, object]] = []

    for preview_idx, image_idx in enumerate(selected):
        record = image_records[int(image_idx)]
        odom_idx, time_error = nearest_index(odom_times, record.time)
        odom = odom_records[odom_idx]

        image = decode_image_msg(record.msg, record.msgtype)
        filename = f"{args.bag.parent.name}__{args.bag.stem}__{int(record.time * 1e9):019d}.jpg"
        out_image = previews_dir / filename
        image.save(out_image, quality=95)
        preview_paths.append(out_image)

        rows.append(
            {
                "image_timestamp": f"{record.time:.9f}",
                "odometry_timestamp": f"{odom.time:.9f}",
                "time_difference_seconds": f"{time_error:.9f}",
                "position_x": float(odom.position[0]),
                "position_y": float(odom.position[1]),
                "position_z": "",
                "orientation_x": "",
                "orientation_y": "",
                "orientation_z": "",
                "orientation_w": "",
                "linear_x": float(odom.velocity[0]),
                "angular_z": float(odom.velocity[1]),
                "image_path": str(out_image),
            }
        )

    csv_path = metadata_dir / "preview_odometry.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    sheet_path = previews_dir / "contact_sheet.jpg"
    contact_sheet(preview_paths, sheet_path)

    write_json(
        metadata_dir / "preview_summary.json",
        {
            "bag": str(args.bag),
            "image_topic": args.image_topic,
            "odom_topic": args.odom_topic,
            "num_raw_images": counts["num_raw_images"],
            "num_raw_odom": counts["num_raw_odom"],
            "num_preview_frames": len(preview_paths),
            "contact_sheet": str(sheet_path),
            "preview_odometry_csv": str(csv_path),
        },
    )

    print(f"Saved {len(preview_paths)} preview frames to {previews_dir}")
    print(f"Saved contact sheet: {sheet_path}")
    print(f"Saved odometry CSV: {csv_path}")


if __name__ == "__main__":
    main()
