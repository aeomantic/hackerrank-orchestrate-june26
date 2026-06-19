"""Image path parsing and encoding for the OpenAI Responses API."""

from __future__ import annotations

import base64
import io
import pillow_avif  # noqa: F401 -- registers AVIF decode support with Pillow; needed because not all platform Pillow wheels bundle it
from pathlib import Path

from PIL import Image


def parse_image_paths(image_paths_field: str) -> list[str]:
    """Split the semicolon-separated image_paths column into a list."""
    return [p.strip() for p in image_paths_field.split(";") if p.strip()]


def image_id_from_path(path: str) -> str:
    """'images/test/case_001/img_1.jpg' -> 'img_1'."""
    return Path(path).stem


def resolve_image_path(dataset_root: Path, relative_path: str) -> Path:
    """image_paths entries are relative to the dataset root (the same root
    that contains sample_claims.csv / claims.csv), e.g.
    'images/test/case_001/img_1.jpg'."""
    return dataset_root / relative_path


_SUPPORTED_MIME = {"JPEG": "image/jpeg", "PNG": "image/png", "GIF": "image/gif", "WEBP": "image/webp"}


def encode_image_data_url(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    with Image.open(path) as im:
        fmt = im.format
        if fmt in _SUPPORTED_MIME:
            data = path.read_bytes()
            mime = _SUPPORTED_MIME[fmt]
        else:
            buf = io.BytesIO()
            im.convert("RGB").save(buf, format="PNG")
            data = buf.getvalue()
            mime = "image/png"
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{b64}"


def load_images_for_call(dataset_root: Path, image_paths_field: str) -> list[dict]:
    """Returns [{'image_id': ..., 'data_url': ...}, ...] in submission order."""
    out = []
    for rel_path in parse_image_paths(image_paths_field):
        full_path = resolve_image_path(dataset_root, rel_path)
        out.append({
            "image_id": image_id_from_path(rel_path),
            "data_url": encode_image_data_url(full_path),
        })
    return out
