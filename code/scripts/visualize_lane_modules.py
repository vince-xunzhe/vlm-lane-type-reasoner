#!/usr/bin/env python3
"""Visualize LaneCenterLine intermediate outputs on source images.

The script is intentionally self-contained so it can live beside the generated
debug artifacts in a remote inference result directory.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


PALETTE_BGR = [
    (0, 220, 255),
    (40, 220, 40),
    (255, 120, 0),
    (255, 0, 180),
    (80, 80, 255),
    (0, 160, 255),
    (180, 255, 0),
    (220, 160, 255),
    (0, 255, 180),
    (255, 255, 0),
]

ATTR_COLORS_BGR = {
    0: (80, 230, 80),      # normal
    1: (0, 210, 255),      # bus
    2: (255, 170, 0),      # bicycle
    3: (255, 0, 200),      # variable
    4: (0, 80, 255),       # tidal
}

ATTR_NAMES_EN = {
    0: "normal",
    1: "bus",
    2: "bicycle",
    3: "variable",
    4: "tidal",
}

OBJECT_LABEL_NAMES = {
    0: "bus_text_gong",
    1: "bus_text_jiao",
    2: "bus_related_time_restriction_sign",
    4: "variable_text_ke",
    5: "variable_text_bian",
    6: "mixed_lane_signal_candidate",
    9: "bicycle_sign",
    10: "bicycle_icon",
}

LABEL_FONT: ImageFont.ImageFont | None = None
GENERATED_OUTPUT_SUBDIRS = (
    "center_line_2d_overlay",
    "lanes_overlay",
    "objects_overlay",
    "lane_attr_overlay",
    "combined_overlay",
    "summary",
    "panels_contact_sheet",
)


@dataclass(frozen=True)
class Paths:
    base_dir: Path
    out_dir: Path
    center_dir: Path
    lanes_dir: Path
    objects_dir: Path
    panels_dir: Path
    attr_path: Path
    image_dirs: tuple[Path, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--las-dir", type=Path, default=None, help="LAS root directory; resolves <las>/<las>_r_LaneCenterLine.")
    parser.add_argument("--base-dir", type=Path, default=None, help="LaneCenterLine directory. Overrides --las-dir.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--attr-path", type=Path, default=None, help="LaneType output_lanes_attr.json to visualize.")
    parser.add_argument(
        "--instances-yaml",
        type=Path,
        default=None,
        help="YAML file listing frame stems to visualize. Missing file means all discovered frames.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Debug limit; 0 means all frames.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--frames", default="", help="Comma-separated frame stems, or a text file with one frame per line.")
    parser.add_argument("--no-panels", action="store_true")
    parser.add_argument("--panel-thumb-width", type=int, default=320)
    parser.add_argument("--panel-max", type=int, default=0, help="Max panel PNGs per frame; 0 means all.")
    parser.add_argument("--panel-page-size", type=int, default=40, help="Max panel PNGs per contact-sheet page.")
    parser.add_argument("--summary-width", type=int, default=640)
    return parser.parse_args()


def lane_centerline_dir_from_las_dir(las_dir: Path) -> Path:
    las_dir = las_dir.expanduser().resolve()
    return las_dir / f"{las_dir.name}_r_LaneCenterLine"


def resolve_base_dir(args: argparse.Namespace) -> Path:
    if args.base_dir is not None:
        return args.base_dir.expanduser().resolve()
    if args.las_dir is not None:
        return lane_centerline_dir_from_las_dir(args.las_dir)

    cwd = Path.cwd().resolve()
    if (cwd / "center_line_2d").exists() and (cwd / "lanes").exists():
        return cwd
    if cwd.name == "vis_debug" and (cwd.parent / "center_line_2d").exists():
        return cwd.parent
    raise SystemExit("Provide --las-dir or --base-dir, or run from a LaneCenterLine directory.")


def unique_paths(paths: Iterable[Path]) -> tuple[Path, ...]:
    seen: set[str] = set()
    unique: list[Path] = []
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return tuple(unique)


def build_paths(base_dir: Path, output_dir: Path | None, attr_path: Path | None = None) -> Paths:
    base_dir = base_dir.expanduser().resolve()
    out_dir = output_dir if output_dir is not None else base_dir / "vis_debug"
    las_dir = base_dir.parent
    las_name = las_dir.name
    return Paths(
        base_dir=base_dir,
        out_dir=out_dir.expanduser().resolve(),
        center_dir=base_dir / "center_line_2d" / "output",
        lanes_dir=base_dir / "lanes",
        objects_dir=base_dir / "objects" / "pred",
        panels_dir=base_dir / "inference" / "panels",
        attr_path=(attr_path.expanduser().resolve() if attr_path is not None else base_dir / "output" / "output_lanes_attr.json"),
        image_dirs=unique_paths(
            [
                las_dir / las_name / "Data" / "Img" / "Camera0",
                base_dir / "inference" / "output" / "input" / "images",
                base_dir / "inference" / "output" / "input",
                base_dir,
                las_dir,
            ]
        ),
    )


def ensure_dirs(paths: Paths) -> None:
    for rel in GENERATED_OUTPUT_SUBDIRS:
        (paths.out_dir / rel).mkdir(parents=True, exist_ok=True)


def output_frame_stem(path: Path) -> str:
    stem = path.stem
    if "_page" in stem:
        return stem.rsplit("_page", 1)[0]
    return stem


def prune_unselected_outputs(paths: Paths, selected_frames: list[str]) -> int:
    selected = set(selected_frames)
    removed = 0
    for rel in GENERATED_OUTPUT_SUBDIRS:
        output_dir = paths.out_dir / rel
        if not output_dir.exists():
            continue
        for pattern in ("*.jpg", "*.jpeg", "*.png"):
            for path in output_dir.glob(pattern):
                if output_frame_stem(path) in selected:
                    continue
                path.unlink()
                removed += 1
    return removed


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        print(f"[warn] failed to read json {path}: {exc}", file=sys.stderr)
        return default


def discover_frames(paths: Paths, limit: int = 0) -> list[str]:
    stems: set[str] = set()
    for p in paths.objects_dir.glob("*.json"):
        stems.add(p.stem)
    for p in paths.center_dir.glob("*.json"):
        stems.add(p.stem)
    for p in paths.lanes_dir.glob("*_segments.json"):
        stems.add(p.name[: -len("_segments.json")])
    frames = sorted(stems)
    if limit > 0:
        frames = frames[:limit]
    return frames


def filter_frames(frames: list[str], frame_arg: str) -> list[str]:
    if not frame_arg:
        return frames
    path = Path(frame_arg)
    if path.exists():
        requested = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        requested = [item.strip() for item in frame_arg.split(",") if item.strip()]
    requested_set = set(requested)
    return [frame for frame in frames if frame in requested_set]


def _clean_yaml_scalar(value: str) -> str:
    value = value.split("#", 1)[0].strip().rstrip(",")
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        value = value[1:-1]
    return value.strip()


def load_instances_yaml(path: Path) -> list[str] | None:
    """Load a small YAML frame list without requiring PyYAML at runtime.

    Supported shapes:

      instances:
        - frame_id

      frames:
        - frame_id

      - frame_id
    """

    if not path.exists():
        return None
    instances: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("-"):
            value = _clean_yaml_scalar(stripped[1:])
            if value:
                instances.append(Path(value).stem)
            continue
        if ":" not in stripped:
            continue
        key, raw_value = stripped.split(":", 1)
        key = key.strip()
        if key not in {"instances", "frames", "frame_ids"}:
            continue
        value = _clean_yaml_scalar(raw_value)
        if not value:
            continue
        if value.startswith("[") and value.endswith("]"):
            for item in value[1:-1].split(","):
                cleaned = _clean_yaml_scalar(item)
                if cleaned:
                    instances.append(Path(cleaned).stem)
        elif value not in {"|", ">"}:
            instances.append(Path(value).stem)
    return list(dict.fromkeys(instances))


def resolve_instances_yaml(args: argparse.Namespace, paths: Paths) -> Path:
    if args.instances_yaml is not None:
        return args.instances_yaml.expanduser().resolve()
    return paths.base_dir / "visualize_instances.yaml"


def apply_instance_selection(
    frames: list[str],
    instances_yaml: Path,
    frame_arg: str,
) -> tuple[list[str], list[str] | None, list[str]]:
    requested = load_instances_yaml(instances_yaml)
    if requested is None:
        return filter_frames(frames, frame_arg), None, []

    frame_set = set(frames)
    requested_set = set(requested)
    selected = [frame for frame in frames if frame in requested_set]
    missing = [frame for frame in requested if frame not in frame_set]
    return selected, requested, missing


def load_attrs(paths: Paths) -> dict[str, Any]:
    data = load_json(paths.attr_path, default={})
    return data if isinstance(data, dict) else {}


def get_font(size: int = 20) -> ImageFont.ImageFont:
    global LABEL_FONT
    if LABEL_FONT is not None:
        return LABEL_FONT
    candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                LABEL_FONT = ImageFont.truetype(path, size=size)
                return LABEL_FONT
            except Exception:
                pass
    LABEL_FONT = ImageFont.load_default()
    return LABEL_FONT


def cv_to_pil(image_bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))


def pil_to_cv(image: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)


def draw_text_box(
    image_bgr: np.ndarray,
    text: str,
    xy: tuple[int, int],
    fill_bgr: tuple[int, int, int] = (0, 0, 0),
    text_rgb: tuple[int, int, int] = (255, 255, 255),
    font_size: int = 20,
) -> None:
    if not text:
        return
    pil = cv_to_pil(image_bgr)
    draw = ImageDraw.Draw(pil)
    font = get_font(font_size)
    x, y = xy
    try:
        bbox = draw.multiline_textbbox((x, y), text, font=font, spacing=2)
    except AttributeError:
        bbox = draw.textbbox((x, y), text.replace("\n", " "), font=font)
    pad = 4
    rect = (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad)
    fill_rgb = (fill_bgr[2], fill_bgr[1], fill_bgr[0])
    draw.rectangle(rect, fill=fill_rgb)
    draw.multiline_text((x, y), text, font=font, fill=text_rgb, spacing=2)
    image_bgr[:, :, :] = pil_to_cv(pil)


def add_header(image_bgr: np.ndarray, title: str, extra: str = "") -> np.ndarray:
    out = image_bgr.copy()
    text = title if not extra else f"{title} | {extra}"
    draw_text_box(out, text, (16, 16), fill_bgr=(20, 20, 20), font_size=24)
    return out


def color_for_id(value: Any) -> tuple[int, int, int]:
    try:
        idx = int(value)
    except Exception:
        idx = abs(hash(str(value)))
    return PALETTE_BGR[idx % len(PALETTE_BGR)]


def object_label_name(label: Any) -> str:
    try:
        label_id = int(label)
    except Exception:
        return f"label_{label}"
    return OBJECT_LABEL_NAMES.get(label_id, f"label_{label_id}")


def points_to_np(points: Iterable[Any]) -> np.ndarray | None:
    parsed: list[tuple[int, int]] = []
    for pt in points:
        if isinstance(pt, dict):
            x = pt.get("x", pt.get("X", pt.get("u", pt.get("col"))))
            y = pt.get("y", pt.get("Y", pt.get("v", pt.get("row"))))
        elif isinstance(pt, (list, tuple)) and len(pt) >= 2:
            x, y = pt[0], pt[1]
        else:
            continue
        try:
            parsed.append((int(round(float(x))), int(round(float(y)))))
        except Exception:
            continue
    if len(parsed) < 2:
        return None
    return np.asarray(parsed, dtype=np.int32).reshape((-1, 1, 2))


def alpha_blend(base: np.ndarray, overlay: np.ndarray, alpha: float) -> np.ndarray:
    return cv2.addWeighted(overlay, alpha, base, 1.0 - alpha, 0)


def draw_center_lines(image_bgr: np.ndarray, center_data: dict[str, Any]) -> np.ndarray:
    out = image_bgr.copy()
    lanes = center_data.get("lane", []) if isinstance(center_data, dict) else []
    for idx, lane in enumerate(lanes):
        if not isinstance(lane, dict):
            continue
        lane_id = lane.get("id", idx)
        pts = points_to_np(lane.get("points", []))
        if pts is None:
            continue
        color = color_for_id(lane_id)
        cv2.polylines(out, [pts], isClosed=False, color=color, thickness=5, lineType=cv2.LINE_AA)
        for p in pts.reshape(-1, 2):
            cv2.circle(out, tuple(p), 5, color, -1, lineType=cv2.LINE_AA)
        label_pt = tuple(pts.reshape(-1, 2)[0])
        attr = lane.get("attribute", "")
        draw_text_box(out, f"CL id={lane_id} {attr}", (int(label_pt[0]) + 6, int(label_pt[1]) - 26), color, font_size=18)
    extra = ""
    if isinstance(center_data, dict):
        yrange = center_data.get("visible_geo_y_range")
        if yrange:
            extra = f"visible_y={yrange}"
    return add_header(out, "center_line_2d/output", extra)


def draw_lanes(image_bgr: np.ndarray, lane_data: dict[str, Any]) -> np.ndarray:
    out = image_bgr.copy()
    fill = out.copy()
    info_by_id: dict[str, dict[str, Any]] = {}
    if isinstance(lane_data, dict):
        for item in lane_data.get("segments_info", []):
            if isinstance(item, dict) and "id" in item:
                info_by_id[str(item["id"])] = item
    polylines_by_id = lane_data.get("polylines_by_id", {}) if isinstance(lane_data, dict) else {}
    if isinstance(polylines_by_id, dict):
        for seg_id, groups in polylines_by_id.items():
            color = color_for_id(seg_id)
            first_label_pt: tuple[int, int] | None = None
            for points in groups if isinstance(groups, list) else []:
                pts = points_to_np(points)
                if pts is None:
                    continue
                if len(pts) >= 3:
                    cv2.fillPoly(fill, [pts], color=color, lineType=cv2.LINE_AA)
                cv2.polylines(out, [pts], isClosed=False, color=color, thickness=4, lineType=cv2.LINE_AA)
                if first_label_pt is None:
                    first = pts.reshape(-1, 2)[0]
                    first_label_pt = (int(first[0]), int(first[1]))
            if first_label_pt is not None:
                info = info_by_id.get(str(seg_id), {})
                cat = info.get("category_id", "?")
                score = info.get("score")
                score_text = "" if score is None else f" {float(score):.2f}"
                draw_text_box(out, f"seg={seg_id} cat={cat}{score_text}", first_label_pt, color, font_size=17)
    out = alpha_blend(out, fill, 0.18)
    return add_header(out, "lanes/*_segments.json")


def draw_objects(image_bgr: np.ndarray, object_data: dict[str, Any]) -> np.ndarray:
    out = image_bgr.copy()
    labels = object_data.get("labels_info", []) if isinstance(object_data, dict) else []
    scores = object_data.get("scores_info", []) if isinstance(object_data, dict) else []
    boxes = object_data.get("bboxes_info", []) if isinstance(object_data, dict) else []
    for idx, box in enumerate(boxes):
        if not isinstance(box, (list, tuple)) or len(box) < 4:
            continue
        try:
            x1, y1, x2, y2 = [int(round(float(v))) for v in box[:4]]
        except Exception:
            continue
        label = labels[idx] if idx < len(labels) else "?"
        score = scores[idx] if idx < len(scores) else None
        color = color_for_id(label)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 4, lineType=cv2.LINE_AA)
        text = f"obj={label} {object_label_name(label)}"
        if score is not None:
            try:
                text += f"\nscore={float(score):.2f}"
            except Exception:
                text += f"\nscore={score}"
        draw_text_box(out, text, (x1 + 4, max(4, y1 - 52)), color, font_size=18)
    return add_header(out, "objects/pred", f"objects={len(boxes)}")


def get_lane_attr(attr_lane: dict[str, Any]) -> int | None:
    for key in ("Attr", "attr", "attribute"):
        if key in attr_lane:
            try:
                return int(attr_lane[key])
            except Exception:
                return None
    for points_key in ("points_utm", "Geo_points_utm"):
        points = attr_lane.get(points_key)
        if isinstance(points, list):
            for pt in points:
                if isinstance(pt, dict) and "Attr" in pt:
                    try:
                        return int(pt["Attr"])
                    except Exception:
                        return None
    return None


def draw_lane_attrs(
    image_bgr: np.ndarray,
    center_data: dict[str, Any],
    attr_data_for_frame: dict[str, Any],
) -> np.ndarray:
    out = image_bgr.copy()
    attr_lanes = {}
    if isinstance(attr_data_for_frame, dict):
        attr_lanes = attr_data_for_frame.get("lane", {}) or {}
    center_lanes = center_data.get("lane", []) if isinstance(center_data, dict) else []
    for idx, lane in enumerate(center_lanes):
        if not isinstance(lane, dict):
            continue
        lane_id = str(lane.get("id", idx))
        pts = points_to_np(lane.get("points", []))
        if pts is None:
            continue
        attr_lane = attr_lanes.get(lane_id, {}) if isinstance(attr_lanes, dict) else {}
        attr = get_lane_attr(attr_lane) if isinstance(attr_lane, dict) else None
        color = ATTR_COLORS_BGR.get(attr, color_for_id(lane_id))
        cv2.polylines(out, [pts], isClosed=False, color=color, thickness=7, lineType=cv2.LINE_AA)
        for p in pts.reshape(-1, 2):
            cv2.circle(out, tuple(p), 6, color, -1, lineType=cv2.LINE_AA)
        first = pts.reshape(-1, 2)[0]
        if attr is None:
            name = "unknown"
            attr_text = "?"
        else:
            name = ATTR_NAMES_EN.get(attr, str(attr))
            attr_text = str(attr)
        draw_text_box(out, f"lane={lane_id} Attr={attr_text} {name}", (int(first[0]) + 6, int(first[1]) - 26), color, font_size=18)
    legend = "Attr: 0 normal, 1 bus, 2 bicycle, 3 variable, 4 tidal"
    out = add_header(out, "output/output_lanes_attr.json", legend)
    return out


def draw_combined(
    image_bgr: np.ndarray,
    center_data: dict[str, Any],
    lane_data: dict[str, Any],
    object_data: dict[str, Any],
    attr_data_for_frame: dict[str, Any],
) -> np.ndarray:
    out = draw_lanes(image_bgr, lane_data)
    out = draw_objects(out, object_data)
    out = draw_lane_attrs(out, center_data, attr_data_for_frame)
    return add_header(out, "combined overlay", "lanes + objects + lane attrs")


def image_path_for_frame(paths: Paths, frame: str, object_data: dict[str, Any]) -> Path | None:
    if isinstance(object_data, dict):
        img_path = object_data.get("img_path")
        img_name = object_data.get("img_name") or f"{frame}.jpg"
    else:
        img_path = None
        img_name = f"{frame}.jpg"

    candidates: list[Path] = []
    image_names = unique_paths(
        [
            Path(str(img_name)),
            Path(f"{frame}.jpg"),
            Path(f"{frame}.jpeg"),
            Path(f"{frame}.png"),
        ]
    )
    for directory in paths.image_dirs:
        for name in image_names:
            candidates.append(directory / name.name)
    if img_path:
        candidates.append(Path(str(img_path)))

    for path in candidates:
        if path.exists() and path.is_file():
            return path
    return None


def read_image(path: Path | None) -> np.ndarray | None:
    if path is None:
        return None
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return img


def save_jpg(path: Path, image_bgr: np.ndarray, overwrite: bool = False) -> bool:
    if path.exists() and not overwrite:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        raise RuntimeError(f"failed to write {path}")
    return True


def resize_keep_aspect(image_bgr: np.ndarray, width: int) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    if w == width:
        return image_bgr.copy()
    scale = width / max(1, w)
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(image_bgr, (width, new_h), interpolation=cv2.INTER_AREA)


def make_summary(
    frame: str,
    images: list[tuple[str, np.ndarray]],
    width: int,
) -> np.ndarray:
    tiles: list[np.ndarray] = []
    for title, image in images:
        tile = resize_keep_aspect(image, width)
        tile = add_header(tile, title)
        tiles.append(tile)
    if not tiles:
        raise ValueError("no images for summary")
    cols = 3
    rows = int(math.ceil(len(tiles) / cols))
    tile_h = max(t.shape[0] for t in tiles)
    canvas = np.full((rows * tile_h, cols * width, 3), 245, dtype=np.uint8)
    for idx, tile in enumerate(tiles):
        row = idx // cols
        col = idx % cols
        h, w = tile.shape[:2]
        canvas[row * tile_h : row * tile_h + h, col * width : col * width + w] = tile
    draw_text_box(canvas, frame, (16, 16), fill_bgr=(0, 0, 0), font_size=28)
    return canvas


def collect_panel_files(paths: Paths, frame: str, panel_max: int) -> list[Path]:
    if not paths.panels_dir.exists():
        return []
    files: list[Path] = []
    for pass_dir in sorted(p for p in paths.panels_dir.iterdir() if p.is_dir()):
        frame_dir = pass_dir / frame
        if not frame_dir.exists():
            continue
        files.extend(sorted(frame_dir.glob("L*/*.png")))
    if panel_max > 0:
        files = files[:panel_max]
    return files


def panel_label(paths: Paths, panel_path: Path) -> str:
    try:
        rel = panel_path.relative_to(paths.panels_dir)
        parts = rel.parts
        if len(parts) >= 4:
            return f"{parts[0]} {parts[2]} {parts[3]}"
        return str(rel)
    except Exception:
        return panel_path.name


def make_panel_contact_sheet(
    paths: Paths,
    frame: str,
    panel_files: list[Path],
    thumb_width: int,
) -> Image.Image | None:
    if not panel_files:
        return None
    thumbs: list[Image.Image] = []
    label_h = 34
    for panel in panel_files:
        try:
            img = Image.open(panel).convert("RGB")
        except Exception as exc:
            print(f"[warn] failed to read panel {panel}: {exc}", file=sys.stderr)
            continue
        scale = thumb_width / max(1, img.width)
        thumb_height = max(1, int(round(img.height * scale)))
        img = img.resize((thumb_width, thumb_height), Image.Resampling.LANCZOS)
        tile = Image.new("RGB", (thumb_width, thumb_height + label_h), (245, 245, 245))
        tile.paste(img, (0, label_h))
        draw = ImageDraw.Draw(tile)
        font = get_font(18)
        label = panel_label(paths, panel)
        draw.rectangle((0, 0, thumb_width, label_h), fill=(20, 20, 20))
        draw.text((6, 6), label[:80], fill=(255, 255, 255), font=font)
        thumbs.append(tile)
    if not thumbs:
        return None
    cols = min(4, max(1, int(math.ceil(math.sqrt(len(thumbs))))))
    rows = int(math.ceil(len(thumbs) / cols))
    tile_h = max(t.height for t in thumbs)
    canvas = Image.new("RGB", (cols * thumb_width, rows * tile_h + 42), (235, 235, 235))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, canvas.width, 42), fill=(0, 0, 0))
    draw.text((10, 8), f"{frame} | inference/panels | panels={len(panel_files)}", fill=(255, 255, 255), font=get_font(22))
    for idx, thumb in enumerate(thumbs):
        row = idx // cols
        col = idx % cols
        canvas.paste(thumb, (col * thumb_width, 42 + row * tile_h))
    return canvas


def save_panel_sheet(path: Path, image: Image.Image, overwrite: bool = False) -> bool:
    if path.exists() and not overwrite:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, quality=88)
    return True


def process_frame(
    paths: Paths,
    frame: str,
    attrs: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    center_data = load_json(paths.center_dir / f"{frame}.json", default={})
    lane_data = load_json(paths.lanes_dir / f"{frame}_segments.json", default={})
    object_data = load_json(paths.objects_dir / f"{frame}.json", default={})
    attr_data = attrs.get(f"{frame}.jpg", {})

    img_path = image_path_for_frame(paths, frame, object_data)
    src = read_image(img_path)
    result: dict[str, Any] = {
        "frame": frame,
        "image_path": str(img_path) if img_path else None,
        "ok": False,
        "outputs": {},
        "panels": 0,
    }
    if src is None:
        result["error"] = "missing source image"
        return result

    center_img = draw_center_lines(src, center_data if isinstance(center_data, dict) else {})
    lanes_img = draw_lanes(src, lane_data if isinstance(lane_data, dict) else {})
    objects_img = draw_objects(src, object_data if isinstance(object_data, dict) else {})
    attrs_img = draw_lane_attrs(src, center_data if isinstance(center_data, dict) else {}, attr_data if isinstance(attr_data, dict) else {})
    combined_img = draw_combined(
        src,
        center_data if isinstance(center_data, dict) else {},
        lane_data if isinstance(lane_data, dict) else {},
        object_data if isinstance(object_data, dict) else {},
        attr_data if isinstance(attr_data, dict) else {},
    )
    summary_img = make_summary(
        frame,
        [
            ("source image", src),
            ("center_line_2d", center_img),
            ("lanes", lanes_img),
            ("objects", objects_img),
            ("lane_attr", attrs_img),
            ("combined", combined_img),
        ],
        args.summary_width,
    )

    out_map = {
        "center_line_2d_overlay": center_img,
        "lanes_overlay": lanes_img,
        "objects_overlay": objects_img,
        "lane_attr_overlay": attrs_img,
        "combined_overlay": combined_img,
        "summary": summary_img,
    }
    for rel, img in out_map.items():
        out_path = paths.out_dir / rel / f"{frame}.jpg"
        save_jpg(out_path, img, overwrite=args.overwrite)
        result["outputs"][rel] = str(out_path)

    if not args.no_panels:
        panel_files = collect_panel_files(paths, frame, args.panel_max)
        result["panels"] = len(panel_files)
        if panel_files:
            page_size = max(1, int(args.panel_page_size))
            page_paths: list[str] = []
            page_count = int(math.ceil(len(panel_files) / page_size))
            for page_idx in range(page_count):
                chunk = panel_files[page_idx * page_size : (page_idx + 1) * page_size]
                sheet = make_panel_contact_sheet(paths, frame, chunk, args.panel_thumb_width)
                if sheet is None:
                    continue
                if page_count == 1:
                    panel_path = paths.out_dir / "panels_contact_sheet" / f"{frame}.jpg"
                else:
                    panel_path = paths.out_dir / "panels_contact_sheet" / f"{frame}_page{page_idx + 1:03d}.jpg"
                save_panel_sheet(panel_path, sheet, overwrite=args.overwrite)
                page_paths.append(str(panel_path))
            if page_paths:
                result["outputs"]["panels_contact_sheet"] = page_paths[0]
                result["outputs"]["panels_contact_sheet_pages"] = page_paths

    result["ok"] = True
    return result


def write_readme(paths: Paths, args: argparse.Namespace, frames: list[str], results: list[dict[str, Any]]) -> None:
    ok_count = sum(1 for r in results if r.get("ok"))
    panel_count = sum(1 for r in results if r.get("outputs", {}).get("panels_contact_sheet"))
    lines = [
        "# LaneCenterLine Visual Debug",
        "",
        "Generated by `visualize_lane_modules.py`.",
        "",
        "## Inputs",
        "",
        f"- Base directory: `{paths.base_dir}`",
        f"- Center line 2D: `{paths.center_dir}`",
        f"- Inference panels: `{paths.panels_dir}`",
        f"- Lanes: `{paths.lanes_dir}`",
        f"- Objects: `{paths.objects_dir}`",
        f"- Lane attributes: `{paths.attr_path}`",
        f"- Source image search dirs: `{', '.join(str(path) for path in paths.image_dirs)}`",
        f"- Instance YAML: `{resolve_instances_yaml(args, paths)}`",
        "",
        "## Outputs",
        "",
        "- `center_line_2d_overlay/`: center-line image points from `center_line_2d/output`.",
        "- `lanes_overlay/`: lane segmentation polylines from `lanes/*_segments.json`.",
        "- `objects_overlay/`: object detector boxes from `objects/pred`.",
        "- `lane_attr_overlay/`: center lines colored by `output/output_lanes_attr.json` Attr.",
        "- `combined_overlay/`: lanes + objects + lane attributes on one source image.",
        "- `summary/`: six-panel per-frame overview.",
        "- `panels_contact_sheet/`: existing `inference/panels` PNGs grouped by frame.",
        "- `manifest.json`: machine-readable inventory.",
        "- `index.html`: quick browser index.",
        "- `<base_dir>/visualize_instances.yaml`: optional frame list. If the file exists, only listed frame stems are visualized; if it is missing, all discovered frames are visualized.",
        "",
        "## Run",
        "",
        "```bash",
        f"python3 visualize_lane_modules.py --base-dir {paths.base_dir} --output-dir {paths.out_dir} --attr-path {paths.attr_path} --overwrite --workers {args.workers}",
        "```",
        "",
        "## Counts",
        "",
        f"- Frames requested: {len(frames)}",
        f"- Frames visualized: {ok_count}",
        f"- Frames with panel contact sheets: {panel_count}",
    ]
    (paths.out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def rel_link(paths: Paths, output_path: str | None) -> str:
    if not output_path:
        return ""
    try:
        return html.escape(str(Path(output_path).relative_to(paths.out_dir)))
    except Exception:
        return html.escape(output_path)


def write_index(paths: Paths, results: list[dict[str, Any]]) -> None:
    rows = []
    for r in sorted(results, key=lambda x: x.get("frame", "")):
        outs = r.get("outputs", {})
        frame = html.escape(str(r.get("frame", "")))
        status = "ok" if r.get("ok") else html.escape(str(r.get("error", "failed")))
        cells = [f"<td>{frame}</td>", f"<td>{status}</td>"]
        for key in [
            "summary",
            "combined_overlay",
            "center_line_2d_overlay",
            "lanes_overlay",
            "objects_overlay",
            "lane_attr_overlay",
            "panels_contact_sheet",
        ]:
            link = rel_link(paths, outs.get(key))
            cells.append(f'<td><a href="{link}">{key}</a></td>' if link else "<td></td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    page = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>LaneCenterLine Visual Debug</title>
  <style>
    body {{ font-family: sans-serif; margin: 24px; }}
    table {{ border-collapse: collapse; font-size: 13px; }}
    th, td {{ border: 1px solid #ccc; padding: 4px 6px; white-space: nowrap; }}
    th {{ background: #f2f2f2; position: sticky; top: 0; }}
  </style>
</head>
<body>
  <h1>LaneCenterLine Visual Debug</h1>
  <p>Frames: {len(results)}</p>
  <table>
    <thead>
      <tr>
        <th>frame</th><th>status</th><th>summary</th><th>combined</th>
        <th>center</th><th>lanes</th><th>objects</th><th>attr</th><th>panels</th>
      </tr>
    </thead>
    <tbody>
      {''.join(rows)}
    </tbody>
  </table>
</body>
</html>
"""
    (paths.out_dir / "index.html").write_text(page, encoding="utf-8")


def write_manifest(
    paths: Paths,
    args: argparse.Namespace,
    frames: list[str],
    results: list[dict[str, Any]],
    requested_instances: list[str] | None = None,
    missing_instances: list[str] | None = None,
) -> None:
    payload = {
        "base_dir": str(paths.base_dir),
        "output_dir": str(paths.out_dir),
        "image_dirs": [str(path) for path in paths.image_dirs],
        "instances_yaml": str(resolve_instances_yaml(args, paths)),
        "requested_instances": requested_instances,
        "missing_instances": missing_instances or [],
        "frame_count": len(frames),
        "ok_count": sum(1 for r in results if r.get("ok")),
        "failed_count": sum(1 for r in results if not r.get("ok")),
        "args": {
            "las_dir": str(args.las_dir) if args.las_dir else None,
            "base_dir": str(args.base_dir) if args.base_dir else None,
            "attr_path": str(args.attr_path) if args.attr_path else None,
            "limit": args.limit,
            "workers": args.workers,
            "overwrite": args.overwrite,
            "frames": args.frames,
            "no_panels": args.no_panels,
            "panel_thumb_width": args.panel_thumb_width,
            "panel_max": args.panel_max,
            "panel_page_size": args.panel_page_size,
            "summary_width": args.summary_width,
        },
        "results": results,
    }
    (paths.out_dir / "manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    base_dir = resolve_base_dir(args)
    paths = build_paths(base_dir, args.output_dir, args.attr_path)
    ensure_dirs(paths)
    frames = discover_frames(paths, args.limit)
    instances_yaml = resolve_instances_yaml(args, paths)
    frames, requested_instances, missing_instances = apply_instance_selection(frames, instances_yaml, args.frames)
    print(f"[info] base_dir={paths.base_dir}")
    print(f"[info] output_dir={paths.out_dir}")
    print(f"[info] instances_yaml={instances_yaml} exists={instances_yaml.exists()}")
    if requested_instances is not None:
        print(f"[info] requested_instances={len(requested_instances)} selected={len(frames)} missing={len(missing_instances)}")
        if missing_instances:
            print(f"[warn] missing requested instances: {', '.join(missing_instances)}", file=sys.stderr)
        removed = prune_unselected_outputs(paths, frames)
        if removed:
            print(f"[info] pruned_unselected_visual_outputs={removed}")
    print(f"[info] frames={len(frames)} workers={args.workers}")
    attrs = load_attrs(paths)
    results: list[dict[str, Any]] = []

    worker_count = max(1, args.workers)
    if worker_count == 1:
        for idx, frame in enumerate(frames, 1):
            results.append(process_frame(paths, frame, attrs, args))
            if idx % 25 == 0 or idx == len(frames):
                print(f"[info] processed {idx}/{len(frames)}")
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {executor.submit(process_frame, paths, frame, attrs, args): frame for frame in frames}
            for idx, future in enumerate(as_completed(futures), 1):
                frame = futures[future]
                try:
                    results.append(future.result())
                except Exception as exc:
                    print(f"[error] frame {frame}: {exc}", file=sys.stderr)
                    results.append({"frame": frame, "ok": False, "error": str(exc), "outputs": {}})
                if idx % 25 == 0 or idx == len(frames):
                    print(f"[info] processed {idx}/{len(frames)}")

    results.sort(key=lambda x: x.get("frame", ""))
    if args.frames:
        manifest_path = paths.out_dir / "manifest.json"
        old_manifest = load_json(manifest_path, default={})
        old_results = old_manifest.get("results", []) if isinstance(old_manifest, dict) else []
        merged: dict[str, dict[str, Any]] = {}
        for item in old_results:
            if isinstance(item, dict) and item.get("frame"):
                merged[str(item["frame"])] = item
        for item in results:
            if isinstance(item, dict) and item.get("frame"):
                merged[str(item["frame"])] = item
        if merged:
            results = [merged[key] for key in sorted(merged)]
            frames = sorted(merged)
            print(f"[info] merged results into existing manifest; total_frames={len(frames)}")
    write_manifest(paths, args, frames, results, requested_instances, missing_instances)
    write_index(paths, results)
    write_readme(paths, args, frames, results)
    print(f"[done] ok={sum(1 for r in results if r.get('ok'))} failed={sum(1 for r in results if not r.get('ok'))}")
    print(f"[done] manifest={paths.out_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
