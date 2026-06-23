"""Small image helpers that avoid heavyweight dependencies."""

from __future__ import annotations

import base64
from io import BytesIO
import imghdr
import struct
from pathlib import Path


def image_size(path: Path) -> tuple[int, int]:
    """Return (width, height) for JPEG or PNG images."""

    path = Path(path)
    with path.open("rb") as fp:
        header = fp.read(32)
        if header.startswith(b"\x89PNG\r\n\x1a\n"):
            width, height = struct.unpack(">II", header[16:24])
            return int(width), int(height)

        if header[:2] != b"\xff\xd8":
            raise ValueError(f"Unsupported image format: {path}")

        fp.seek(2)
        while True:
            marker_start = fp.read(1)
            if not marker_start:
                break
            if marker_start != b"\xff":
                continue
            marker = fp.read(1)
            while marker == b"\xff":
                marker = fp.read(1)
            if marker in {b"\xd8", b"\xd9"}:
                continue
            length_bytes = fp.read(2)
            if len(length_bytes) != 2:
                break
            segment_length = struct.unpack(">H", length_bytes)[0]
            if marker in {
                b"\xc0",
                b"\xc1",
                b"\xc2",
                b"\xc3",
                b"\xc5",
                b"\xc6",
                b"\xc7",
                b"\xc9",
                b"\xca",
                b"\xcb",
                b"\xcd",
                b"\xce",
                b"\xcf",
            }:
                data = fp.read(5)
                if len(data) != 5:
                    break
                height, width = struct.unpack(">HH", data[1:5])
                return int(width), int(height)
            fp.seek(segment_length - 2, 1)

    raise ValueError(f"Could not read image size: {path}")


def image_to_data_uri(path: Path, *, max_side: int | None = None, jpeg_quality: int | None = None) -> str:
    path = Path(path)
    if max_side or jpeg_quality:
        try:
            from PIL import Image
        except ImportError as exc:  # pragma: no cover - Pillow is a project dependency.
            raise RuntimeError("Image compression requires Pillow") from exc

        with Image.open(path) as image:
            image = image.convert("RGB")
            if max_side and max(image.size) > max_side:
                image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
            buffer = BytesIO()
            image.save(buffer, format="JPEG", quality=jpeg_quality or 85, optimize=True)
            encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"

    kind = imghdr.what(path) or path.suffix.lstrip(".").lower() or "jpeg"
    mime = "image/jpeg" if kind in {"jpg", "jpeg"} else f"image/{kind}"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"
