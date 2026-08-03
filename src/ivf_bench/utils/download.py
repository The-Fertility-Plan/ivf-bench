"""Resumable HTTP download with progress bar and MD5 verification."""
from __future__ import annotations

import hashlib
from pathlib import Path

import requests
from rich.console import Console
from tqdm import tqdm

console = Console()


def compute_md5(path: Path) -> str:
    """Compute MD5 hex digest of a file."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file(
    url: str,
    dest: Path,
    expected_md5: str | None = None,
    chunk_size: int = 8192,
) -> Path:
    """Download a file with resume support. Skips if already complete."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Skip if already downloaded and MD5 matches
    if dest.exists() and expected_md5:
        if compute_md5(dest) == expected_md5:
            console.print(f"[dim]Already downloaded and verified: {dest.name}[/dim]")
            return dest

    # Check existing partial file size for resume
    existing_size = dest.stat().st_size if dest.exists() else 0
    headers = {}
    if existing_size > 0:
        headers["Range"] = f"bytes={existing_size}-"
        console.print(f"[dim]Resuming from {existing_size} bytes...[/dim]")

    resp = requests.get(url, headers=headers, stream=True, timeout=30)

    # If server doesn't support Range, start over
    if resp.status_code == 200:
        existing_size = 0
        mode = "wb"
    elif resp.status_code == 206:
        mode = "ab"
    else:
        resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0)) + existing_size
    desc = dest.name

    with open(dest, mode) as f, tqdm(
        total=total,
        initial=existing_size,
        unit="B",
        unit_scale=True,
        desc=desc,
    ) as bar:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            bar.update(len(chunk))

    # Verify MD5 after download
    if expected_md5:
        actual = compute_md5(dest)
        if actual != expected_md5:
            raise ValueError(
                f"MD5 mismatch for {dest.name}: expected {expected_md5}, got {actual}"
            )
        console.print(f"[green]MD5 verified: {dest.name}[/green]")

    return dest
