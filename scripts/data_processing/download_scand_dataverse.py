from __future__ import annotations

import argparse
import re
import time
from email.message import Message
from email.parser import Parser
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, quote, unquote, urlparse
from urllib.request import Request, urlopen


URLS = [
    "https://dataverse.tdl.org/file.xhtml?persistentId=doi:10.18738/T8/0PRYRH/S08P2U&version=5.2",
    "https://dataverse.tdl.org/file.xhtml?persistentId=doi:10.18738/T8/0PRYRH/3ETU91&version=5.2",
    "https://dataverse.tdl.org/file.xhtml?persistentId=doi:10.18738/T8/0PRYRH/TQVWDP&version=5.2",
    "https://dataverse.tdl.org/file.xhtml?persistentId=doi:10.18738/T8/0PRYRH/D0JHX3&version=5.2",
    "https://dataverse.tdl.org/file.xhtml?persistentId=doi:10.18738/T8/0PRYRH/CV6PQ9&version=5.2",
    "https://dataverse.tdl.org/file.xhtml?persistentId=doi:10.18738/T8/0PRYRH/CATZ5A&version=5.2",
    "https://dataverse.tdl.org/file.xhtml?persistentId=doi:10.18738/T8/0PRYRH/RGIJWJ&version=5.2",
    "https://dataverse.tdl.org/file.xhtml?persistentId=doi:10.18738/T8/0PRYRH/NHWZVE&version=5.2",
    "https://dataverse.tdl.org/file.xhtml?persistentId=doi:10.18738/T8/0PRYRH/DCF8QK&version=5.2",
    "https://dataverse.tdl.org/file.xhtml?persistentId=doi:10.18738/T8/0PRYRH/NHVC3W&version=5.2",
    "https://dataverse.tdl.org/file.xhtml?persistentId=doi:10.18738/T8/0PRYRH/Z8ORRZ&version=5.2",
    "https://dataverse.tdl.org/file.xhtml?persistentId=doi:10.18738/T8/0PRYRH/KRPNVZ&version=5.2",
    "https://dataverse.tdl.org/file.xhtml?persistentId=doi:10.18738/T8/0PRYRH/HJI5PF&version=5.2",
    "https://dataverse.tdl.org/file.xhtml?persistentId=doi:10.18738/T8/0PRYRH/T2LRFB&version=5.2",
    "https://dataverse.tdl.org/file.xhtml?persistentId=doi:10.18738/T8/0PRYRH/LFMRLZ&version=5.2",
    "https://dataverse.tdl.org/file.xhtml?persistentId=doi:10.18738/T8/0PRYRH/U4IPAE&version=5.2",
    "https://dataverse.tdl.org/file.xhtml?persistentId=doi:10.18738/T8/0PRYRH/N3IHJY&version=5.2",
    "https://dataverse.tdl.org/file.xhtml?persistentId=doi:10.18738/T8/0PRYRH/5B6UD9&version=5.2",
    "https://dataverse.tdl.org/file.xhtml?persistentId=doi:10.18738/T8/0PRYRH/94IOUW&version=5.2",
    "https://dataverse.tdl.org/file.xhtml?persistentId=doi:10.18738/T8/0PRYRH/GRJSQ9&version=5.2",
    "https://dataverse.tdl.org/file.xhtml?persistentId=doi:10.18738/T8/0PRYRH/I3UGFZ&version=5.2",
    "https://dataverse.tdl.org/file.xhtml?persistentId=doi:10.18738/T8/0PRYRH/CDZFF0&version=5.2",
    "https://dataverse.tdl.org/file.xhtml?persistentId=doi:10.18738/T8/0PRYRH/UODTZI&version=5.2",
    "https://dataverse.tdl.org/file.xhtml?persistentId=doi:10.18738/T8/0PRYRH/ZYR5SA&version=5.2",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download SCAND Dataverse files to an external drive.")
    parser.add_argument("--out-dir", type=Path, required=True, help="External-drive folder, for example E:\\SCAND")
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--chunk-mb", type=int, default=8)
    parser.add_argument("--sleep-sec", type=float, default=2.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def dataverse_download_urls(page_url: str) -> tuple[list[str], str]:
    parsed = urlparse(page_url)
    query = parse_qs(parsed.query)
    persistent_id = query["persistentId"][0]
    version = query.get("version", [""])[0]
    encoded_pid = quote(persistent_id, safe="")
    api_encoded = f"{parsed.scheme}://{parsed.netloc}/api/access/datafile/:persistentId?persistentId={encoded_pid}"
    api_plain = f"{parsed.scheme}://{parsed.netloc}/api/access/datafile/:persistentId?persistentId={persistent_id}"
    if version:
        api_encoded += f"&version={quote(version)}"
        api_plain += f"&version={quote(version)}"
    fallback_name = safe_filename(persistent_id.rsplit("/", 1)[-1]) + ".download"
    return [api_encoded, api_plain, page_url], fallback_name


def safe_filename(name: str) -> str:
    name = unquote(name).strip().replace("\\", "_").replace("/", "_")
    name = re.sub(r"[^A-Za-z0-9._ -]+", "_", name)
    return name or "dataverse_file"


def filename_from_content_disposition(header: str | None) -> str | None:
    if not header:
        return None
    msg: Message = Parser().parsestr(f"Content-Disposition: {header}\n")
    filename = msg.get_param("filename", header="content-disposition")
    if filename:
        if isinstance(filename, tuple):
            charset, _language, encoded = filename
            filename = encoded
            if isinstance(filename, bytes):
                filename = filename.decode(charset or "utf-8", errors="replace")
        return safe_filename(filename)
    match = re.search(r"filename\*=UTF-8''([^;]+)", header)
    if match:
        return safe_filename(match.group(1))
    return None


def open_request(url: str, start_byte: int = 0):
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/126.0 Safari/537.36"
        ),
        "Accept": "application/octet-stream,text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    if start_byte > 0:
        headers["Range"] = f"bytes={start_byte}-"
    return urlopen(Request(url, headers=headers), timeout=120)


def format_bytes(num_bytes: int | None) -> str:
    if num_bytes is None:
        return "unknown"
    value = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if value < 1024.0:
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} PB"


def choose_working_url_and_output_path(out_dir: Path, urls: list[str], fallback_name: str) -> tuple[str, Path, int | None]:
    errors = []
    for url in urls:
        try:
            with open_request(url) as response:
                filename = filename_from_content_disposition(response.headers.get("Content-Disposition")) or fallback_name
                total = response.headers.get("Content-Length")
                return url, out_dir / filename, int(total) if total is not None else None
        except HTTPError as exc:
            errors.append(f"{exc.code} {exc.reason}: {url}")
        except (URLError, TimeoutError, OSError) as exc:
            errors.append(f"{exc}: {url}")
    raise RuntimeError("Could not open Dataverse URL. Tried:\n  " + "\n  ".join(errors))


def download_one(page_url: str, out_dir: Path, args: argparse.Namespace) -> None:
    download_urls, fallback_name = dataverse_download_urls(page_url)
    download_url, output_path, total_size = choose_working_url_and_output_path(out_dir, download_urls, fallback_name)
    part_path = output_path.with_name(output_path.name + ".part")

    if output_path.exists() and not args.overwrite:
        if total_size is None or output_path.stat().st_size == total_size:
            print(f"SKIP {output_path.name} ({format_bytes(output_path.stat().st_size)})")
            return
        print(f"Existing file size differs; resuming via .part: {output_path.name}")
        output_path.rename(part_path)

    downloaded = part_path.stat().st_size if part_path.exists() else 0
    mode = "ab" if downloaded else "wb"
    chunk_size = max(1, args.chunk_mb) * 1024 * 1024

    print(f"GET  {output_path.name}")
    print(f"     size={format_bytes(total_size)} resume_from={format_bytes(downloaded)}")

    for attempt in range(1, args.retries + 1):
        try:
            with open_request(download_url, start_byte=downloaded) as response, part_path.open(mode) as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size:
                        pct = downloaded / total_size * 100.0
                        print(f"\r     {format_bytes(downloaded)} / {format_bytes(total_size)} ({pct:.1f}%)", end="")
                    else:
                        print(f"\r     {format_bytes(downloaded)}", end="")
                print()
            part_path.rename(output_path)
            print(f"DONE {output_path}")
            return
        except HTTPError as exc:
            if exc.code == 416 and part_path.exists():
                part_path.rename(output_path)
                print(f"DONE {output_path}")
                return
            print(f"\nAttempt {attempt}/{args.retries} failed: HTTP {exc.code} {exc.reason}")
        except (URLError, TimeoutError, OSError) as exc:
            print(f"\nAttempt {attempt}/{args.retries} failed: {exc}")

        downloaded = part_path.stat().st_size if part_path.exists() else 0
        mode = "ab" if downloaded else "wb"
        time.sleep(args.sleep_sec * attempt)

    raise RuntimeError(f"Failed after {args.retries} attempts: {page_url}")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {len(URLS)} files to {args.out_dir}")
    for index, url in enumerate(URLS, start=1):
        print(f"\n[{index}/{len(URLS)}]")
        download_one(url, args.out_dir, args)


if __name__ == "__main__":
    main()
