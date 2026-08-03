"""PNG image validation using Pillow."""
from __future__ import annotations

from pathlib import Path

from PIL import Image


def validate_png(path: Path) -> bool:
    """Return True if path is a valid, loadable PNG."""
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def get_image_info(path: Path) -> dict:
    """Return width, height, and mode of an image."""
    with Image.open(path) as img:
        return {"width": img.width, "height": img.height, "mode": img.mode}
