"""Packet-based Qwen-VL pipeline for special-lane classification."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from special_lane_common.data import append_jsonl, dump_json, iter_dataset_samples, read_json_or_jsonl
from special_lane_common.geometry import (
    bbox_from_points,
    clamp,
    match_left_right_boundaries,
    norm1000_bbox,
    norm1000_point,
    norm1000_polyline,
    object_lane_geometry,
)
from special_lane_common.inference_config import resolve_runtime
from special_lane_common.output import make_result_record
from special_lane_common.prompts import (
    SYSTEM_PROMPT,
    alpha_compact_prompt,
    alpha_line_prompt,
    alpha_line_recall_prompt,
    alpha_prompt,
)
from special_lane_common.qwen_client import QwenOpenAIClient, build_openai_messages, extract_json_object


FONT = ImageFont.load_default()


def _draw_polyline(draw: ImageDraw.ImageDraw, points: list[list[float]], color: str, width: int) -> None:
    if len(points) >= 2:
        draw.line([tuple(point) for point in points], fill=color, width=width)
    elif len(points) == 1:
        x, y = points[0]
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)


def _draw_text(draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str, color: str) -> None:
    x, y = xy
    draw.text((x + 1, y + 1), text, fill="white", font=FONT)
    draw.text((x, y), text, fill=color, font=FONT)


def _expand_bbox(bbox: list[float], image_size: tuple[int, int], margin: float = 0.25) -> list[int]:
    width, height = image_size
    x1, y1, x2, y2 = bbox
    bw = max(x2 - x1, 8)
    bh = max(y2 - y1, 8)
    return [
        int(clamp(x1 - bw * margin, 0, width - 2)),
        int(clamp(y1 - bh * margin, 0, height - 2)),
        int(clamp(x2 + bw * margin, 1, width - 1)),
        int(clamp(y2 + bh * margin, 1, height - 1)),
    ]


def _union_bbox(boxes: list[list[float]], image_size: tuple[int, int]) -> list[int]:
    boxes = [box for box in boxes if box]
    if not boxes:
        width, height = image_size
        return [0, 0, width - 1, height - 1]
    return [
        int(min(box[0] for box in boxes)),
        int(min(box[1] for box in boxes)),
        int(max(box[2] for box in boxes)),
        int(max(box[3] for box in boxes)),
    ]


def _shift_polyline(points: list[list[float]], offset_x: int, offset_y: int) -> list[list[float]]:
    return [[x - offset_x, y - offset_y] for x, y in points]


def _shift_bbox(bbox: list[float], offset_x: int, offset_y: int) -> list[float]:
    return [bbox[0] - offset_x, bbox[1] - offset_y, bbox[2] - offset_x, bbox[3] - offset_y]


OBJECT_SEMANTIC_PRIORITY = {
    0: 0,   # road text / road symbol
    9: 1,   # bicycle sign/icon
    10: 2,  # bus-related time restriction sign
    5: 3,   # bus-related sign/signal
    1: 4,   # traffic sign/signal
    2: 4,
    4: 4,
    6: 5,   # broad lane signal / road text bucket, often noisy
}


def _bbox_area(bbox: list[float] | None) -> float:
    if not bbox or len(bbox) < 4:
        return 0.0
    return max(0.0, float(bbox[2]) - float(bbox[0])) * max(0.0, float(bbox[3]) - float(bbox[1]))


def _object_sort_key(obj: dict, mode: str = "geometry") -> tuple:
    geometry = obj.get("target_lane_geometry") or {}
    inside_rank = 0 if geometry.get("inside_target_lane_band") else 1
    distance = geometry.get("distance_to_centerline")
    distance_value = float(distance) if distance is not None else 99999.0
    if mode == "semantic_area":
        label_rank = OBJECT_SEMANTIC_PRIORITY.get(obj.get("label_id"), 9)
        score = float(obj.get("score") or obj.get("det_score") or 0.0)
        return (
            inside_rank,
            label_rank,
            -_bbox_area(obj.get("bbox")),
            distance_value,
            -score,
            str(obj.get("object_id")),
        )
    return (inside_rank, distance_value, str(obj.get("object_id")))


def _save_object_contact_sheet(image: Image.Image, objects: list[dict], output_path: Path, max_objects: int, object_sort_mode: str) -> bool:
    selected = sorted(objects, key=lambda obj: _object_sort_key(obj, object_sort_mode))[:max_objects]
    if not selected:
        return False
    tile_w, tile_h = 320, 220
    caption_h = 34
    cols = 2
    rows = (len(selected) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * tile_w, rows * (tile_h + caption_h)), "#F8FAFC")
    draw = ImageDraw.Draw(canvas)
    width, height = image.size
    for idx, obj in enumerate(selected):
        row = idx // cols
        col = idx % cols
        x0 = col * tile_w
        y0 = row * (tile_h + caption_h)
        crop_box = _expand_bbox(obj["bbox"], (width, height), margin=0.45)
        crop = image.crop(tuple(crop_box))
        crop.thumbnail((tile_w - 18, tile_h - 18))
        canvas.paste(crop, (x0 + (tile_w - crop.width) // 2, y0 + (tile_h - crop.height) // 2))
        draw.rectangle((x0, y0, x0 + tile_w - 1, y0 + tile_h + caption_h - 1), outline="#CBD5E1", width=1)
        label = f"{obj['object_id']} {obj.get('label_name')} {obj.get('score'):.2f}" if obj.get("score") is not None else f"{obj['object_id']} {obj.get('label_name')}"
        _draw_text(draw, (x0 + 8, y0 + tile_h + 8), label[:48], "#111827")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return True


def _save_boundary_zoom(
    image: Image.Image,
    boundary: dict | None,
    target_lane: dict,
    output_path: Path,
    side: str,
) -> bool:
    if boundary is None:
        return False
    width, height = image.size
    boundary_box = bbox_from_points(boundary["polyline"])
    lane_box = bbox_from_points(target_lane["centerline"])
    x1 = min(boundary_box[0], lane_box[0])
    y1 = min(boundary_box[1], lane_box[1])
    x2 = max(boundary_box[2], lane_box[2])
    y2 = max(boundary_box[3], lane_box[3])
    crop_box = _expand_bbox([x1, y1, x2, y2], (width, height), margin=0.18)
    crop = image.crop(tuple(crop_box))

    # Keep the road marking itself unobscured.  Only the border and caption
    # identify which packet boundary this raw crop is meant to inspect.
    border = "#2563EB" if side == "left" else "#DC2626"
    draw = ImageDraw.Draw(crop)
    draw.rectangle((0, 0, crop.width - 1, crop.height - 1), outline=border, width=6)
    _draw_text(draw, (10, 10), f"{side}_boundary raw zoom: {boundary['boundary_id']}", border)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    crop.save(output_path)
    return True


def _save_boundary_pair_raw_context(
    image: Image.Image,
    lane: dict,
    boundary_match,
    output_path: Path,
) -> bool:
    left = boundary_match.left
    right = boundary_match.right
    if left is None and right is None:
        return False
    width, height = image.size
    boxes = [bbox_from_points(lane["centerline"])]
    if left:
        boxes.append(bbox_from_points(left["polyline"]))
    if right:
        boxes.append(bbox_from_points(right["polyline"]))
    crop_box = _expand_bbox(_union_bbox(boxes, (width, height)), (width, height), margin=0.35)
    ox, oy = crop_box[0], crop_box[1]
    raw = image.crop(tuple(crop_box))
    overlay = raw.copy()
    draw = ImageDraw.Draw(overlay)
    _draw_polyline(draw, _shift_polyline(lane["centerline"], ox, oy), "#16A34A", 4)
    if left:
        _draw_polyline(draw, _shift_polyline(left["polyline"], ox, oy), "#2563EB", 3)
    if right:
        _draw_polyline(draw, _shift_polyline(right["polyline"], ox, oy), "#DC2626", 3)
    _draw_text(draw, (8, 8), "overlay: green target | blue left | red right", "#111827")

    caption_h = 30
    canvas = Image.new("RGB", (raw.width * 2, raw.height + caption_h), "#F8FAFC")
    canvas.paste(raw, (0, caption_h))
    canvas.paste(overlay, (raw.width, caption_h))
    canvas_draw = ImageDraw.Draw(canvas)
    canvas_draw.rectangle((0, caption_h, raw.width - 1, raw.height + caption_h - 1), outline="#111827", width=2)
    canvas_draw.rectangle((raw.width, caption_h, raw.width * 2 - 1, raw.height + caption_h - 1), outline="#111827", width=2)
    _draw_text(canvas_draw, (8, 8), "LEFT: raw unobscured lane-band crop", "#111827")
    _draw_text(canvas_draw, (raw.width + 8, 8), "RIGHT: same crop with target-boundary overlay", "#111827")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return True


def _save_lane_band_raw_strip(
    image: Image.Image,
    lane: dict,
    boundary_match,
    output_path: Path,
) -> bool:
    left = boundary_match.left
    right = boundary_match.right
    if left is None and right is None:
        return False
    width, height = image.size
    boxes = [bbox_from_points(lane["centerline"])]
    if left:
        boxes.append(bbox_from_points(left["polyline"]))
    if right:
        boxes.append(bbox_from_points(right["polyline"]))
    crop_box = _expand_bbox(_union_bbox(boxes, (width, height)), (width, height), margin=0.55)
    raw = image.crop(tuple(crop_box))
    max_w, max_h = 1600, 1200
    scale = min(max_w / max(raw.width, 1), max_h / max(raw.height, 1))
    if scale > 1.0:
        resample = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.BICUBIC
        raw = raw.resize((int(raw.width * scale), int(raw.height * scale)), resample)

    caption_h = 28
    canvas = Image.new("RGB", (raw.width, raw.height + caption_h), "#F8FAFC")
    canvas.paste(raw, (0, caption_h))
    draw = ImageDraw.Draw(canvas)
    _draw_text(draw, (8, 8), "raw high-resolution target lane band: no overlay, inspect markings/text first", "#111827")
    draw.rectangle((0, caption_h, raw.width - 1, raw.height + caption_h - 1), outline="#111827", width=2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return True


def _save_signal_alignment_whiteboard(
    image_size: tuple[int, int],
    lane: dict,
    boundary_match,
    objects: list[dict],
    output_path: Path,
) -> bool:
    width, height = image_size
    canvas = Image.new("RGB", (width, height), "#F8FAFC")
    draw = ImageDraw.Draw(canvas)
    left = boundary_match.left
    right = boundary_match.right

    if left:
        _draw_polyline(draw, left["polyline"], "#2563EB", 6)
    if right:
        _draw_polyline(draw, right["polyline"], "#DC2626", 6)
    _draw_polyline(draw, lane["centerline"], "#16A34A", 8)

    # Extend the target-lane direction upward so overhead signs can be compared
    # against the target lane without relying on perspective intuition alone.
    if len(lane["centerline"]) >= 2:
        p0 = lane["centerline"][0]
        p1 = lane["centerline"][-1]
        top = min(lane["centerline"], key=lambda point: point[1])
        bottom = max(lane["centerline"], key=lambda point: point[1])
        dy = top[1] - bottom[1]
        dx = top[0] - bottom[0]
        if abs(dy) > 1e-6:
            x_at_top = top[0] + dx / dy * (0 - top[1])
        else:
            x_at_top = top[0]
        draw.line([(top[0], top[1]), (x_at_top, 0)], fill="#16A34A", width=4)
        _draw_text(draw, (min(max(x_at_top, 0), width - 80), 8), "target lane projection", "#16A34A")

    for obj in sorted(objects, key=lambda item: _object_sort_key(item, "geometry")):
        bbox = obj["bbox"]
        geometry = obj.get("target_lane_geometry") or {}
        inside = geometry.get("inside_target_lane_band")
        color = "#F59E0B" if inside else "#64748B"
        draw.rectangle(bbox, outline=color, width=4)
        anchor = geometry.get("anchor_point_px")
        if anchor:
            ax, ay = anchor
            draw.ellipse((ax - 6, ay - 6, ax + 6, ay + 6), fill=color)
            center_x = geometry.get("center_x_at_anchor_y")
            if center_x is not None:
                draw.line([(ax, ay), (center_x, ay)], fill=color, width=3)
                draw.ellipse((center_x - 5, ay - 5, center_x + 5, ay + 5), outline="#16A34A", width=3)
        hint = geometry.get("relation_hint", "missing")
        label = f"{obj['object_id']}:{obj.get('label_name')}:{hint}"
        _draw_text(draw, (bbox[0], max(0, bbox[1] - 14)), label[:64], color)

    _draw_text(
        draw,
        (12, height - 28),
        "green=target/projection | blue=left | red=right | orange=inside band | gray=nearby/outside",
        "#111827",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return True


def _save_signal_lane_ruler_whiteboard(objects: list[dict], output_path: Path) -> bool:
    if not objects:
        return False
    sorted_objects = sorted(objects, key=lambda item: _object_sort_key(item, "geometry"))[:24]
    row_h = 48
    header_h = 72
    width = 1180
    height = header_h + row_h * len(sorted_objects) + 18
    axis_x1 = 500
    axis_x2 = 1060

    canvas = Image.new("RGB", (width, height), "#F8FAFC")
    draw = ImageDraw.Draw(canvas)
    _draw_text(draw, (16, 14), "signal-lane alignment ruler", "#111827")
    _draw_text(draw, (16, 36), "target lane band is rel_x 0..1; center is 0.5; outside values are adjacent lanes", "#334155")
    draw.line((axis_x1, header_h - 16, axis_x2, header_h - 16), fill="#CBD5E1", width=3)
    for rel, label in ((0.0, "0 left"), (0.5, "0.5 center"), (1.0, "1 right")):
        x = axis_x1 + int(rel * (axis_x2 - axis_x1))
        color = "#16A34A" if rel == 0.5 else "#64748B"
        draw.line((x, header_h - 26, x, header_h - 6), fill=color, width=2)
        _draw_text(draw, (x - 34, header_h - 4), label, color)

    for idx, obj in enumerate(sorted_objects):
        y = header_h + idx * row_h
        geometry = obj.get("target_lane_geometry") or {}
        relation_hint = str(geometry.get("relation_hint") or "unknown")
        inside = geometry.get("inside_target_lane_band") is True
        color = "#F59E0B" if inside else "#64748B"
        if "left_adjacent" in relation_hint:
            color = "#2563EB"
        elif "right_adjacent" in relation_hint:
            color = "#DC2626"

        draw.rectangle((12, y + 5, width - 12, y + row_h - 5), outline="#E2E8F0", width=1)
        label = str(obj.get("label_name") or obj.get("label_id") or "object")
        text = f"{obj['object_id']} {label[:30]} hint={relation_hint[:28]}"
        distance = geometry.get("distance_to_centerline")
        if distance is not None:
            try:
                text += f" dist={float(distance):.1f}"
            except (TypeError, ValueError):
                pass
        _draw_text(draw, (22, y + 15), text[:78], color)

        draw.rectangle((axis_x1, y + 14, axis_x2, y + row_h - 14), outline="#CBD5E1", width=1)
        if inside:
            draw.rectangle((axis_x1, y + 14, axis_x2, y + row_h - 14), fill="#ECFDF5")
        center_x = axis_x1 + int(0.5 * (axis_x2 - axis_x1))
        draw.line((center_x, y + 10, center_x, y + row_h - 10), fill="#16A34A", width=2)
        lane_relative_x = geometry.get("lane_relative_x")
        if lane_relative_x is None:
            _draw_text(draw, (axis_x1 + 12, y + 14), "rel_x=unknown", "#64748B")
            continue
        try:
            rel = float(lane_relative_x)
            clamped = clamp(rel, -0.35, 1.35)
            marker_x = axis_x1 + int((clamped + 0.35) / 1.7 * (axis_x2 - axis_x1))
            draw.ellipse((marker_x - 8, y + 16, marker_x + 8, y + 32), fill=color)
            _draw_text(draw, (marker_x + 12, y + 14), f"rel_x={rel:.2f}", color)
        except (TypeError, ValueError):
            _draw_text(draw, (axis_x1 + 12, y + 14), "rel_x=bad", "#64748B")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return True


def render_alpha_views(sample, lane: dict, boundary_match, output_dir: Path, max_object_crops: int, object_sort_mode: str) -> list[dict]:
    output_dir = output_dir / sample.image_id / lane["lane_id"]
    output_dir.mkdir(parents=True, exist_ok=True)
    image = Image.open(sample.image_path).convert("RGB")
    width, height = image.size
    all_lanes = [item["centerline"] for item in sample.lanes]
    left = boundary_match.left
    right = boundary_match.right

    views: list[dict] = []

    global_img = image.copy()
    draw = ImageDraw.Draw(global_img)
    for item in sample.lanes:
        color = "#9CA3AF"
        line_width = 4
        if item["lane_id"] == lane["lane_id"]:
            color = "#16A34A"
            line_width = 8
        _draw_polyline(draw, item["centerline"], color, line_width)
        mid = item["centerline"][len(item["centerline"]) // 2]
        lane_label = item["lane_id"]
        if item.get("gt_type"):
            lane_label = f"{lane_label} gt={item['gt_type']}"
        _draw_text(draw, tuple(mid), lane_label, color)
    if left:
        _draw_polyline(draw, left["polyline"], "#2563EB", 6)
        mid = left["polyline"][len(left["polyline"]) // 2]
        _draw_text(draw, tuple(mid), f"left:{left['boundary_id']}", "#2563EB")
    if right:
        _draw_polyline(draw, right["polyline"], "#DC2626", 6)
        mid = right["polyline"][len(right["polyline"]) // 2]
        _draw_text(draw, tuple(mid), f"right:{right['boundary_id']}", "#DC2626")
    objects_with_geometry = enrich_objects_with_geometry(sample.objects, lane, boundary_match, width, height)

    for obj in objects_with_geometry:
        bbox = obj["bbox"]
        draw.rectangle(bbox, outline="#F59E0B", width=4)
        geometry = obj.get("target_lane_geometry") or {}
        hint = "in" if geometry.get("inside_target_lane_band") else geometry.get("relation_hint", "out")
        _draw_text(draw, (bbox[0], max(0, bbox[1] - 14)), f"{obj['object_id']}:{obj.get('label_name')}:{hint}", "#F59E0B")
    global_path = output_dir / "01_global_overlay.png"
    global_img.save(global_path)
    views.append({"id": "global_overlay", "path": str(global_path.resolve()), "desc": "full frame overlay with target lane, left/right boundaries, object ids"})

    boxes = [bbox_from_points(lane["centerline"])]
    if left:
        boxes.append(bbox_from_points(left["polyline"]))
    if right:
        boxes.append(bbox_from_points(right["polyline"]))
    boxes.extend([obj["bbox"] for obj in objects_with_geometry])
    focus_box = _expand_bbox(_union_bbox(boxes, (width, height)), (width, height), margin=0.25)
    ox, oy = focus_box[0], focus_box[1]

    focus = image.crop(tuple(focus_box))
    focus_draw = ImageDraw.Draw(focus)
    _draw_polyline(focus_draw, _shift_polyline(lane["centerline"], ox, oy), "#16A34A", 7)
    if left:
        _draw_polyline(focus_draw, _shift_polyline(left["polyline"], ox, oy), "#2563EB", 5)
    if right:
        _draw_polyline(focus_draw, _shift_polyline(right["polyline"], ox, oy), "#DC2626", 5)
    for obj in objects_with_geometry:
        focus_draw.rectangle(_shift_bbox(obj["bbox"], ox, oy), outline="#F59E0B", width=3)
    focus_path = output_dir / "02_lane_focus_crop.png"
    focus.save(focus_path)
    views.append({"id": "lane_focus_crop", "path": str(focus_path.resolve()), "desc": "crop around target lane, boundaries, and detected objects"})

    mask = Image.new("RGB", (focus.width, focus.height), "#F8FAFC")
    mask_draw = ImageDraw.Draw(mask)
    for other in all_lanes:
        _draw_polyline(mask_draw, _shift_polyline(other, ox, oy), "#CBD5E1", 3)
    _draw_polyline(mask_draw, _shift_polyline(lane["centerline"], ox, oy), "#16A34A", 7)
    if left:
        _draw_polyline(mask_draw, _shift_polyline(left["polyline"], ox, oy), "#2563EB", 5)
    if right:
        _draw_polyline(mask_draw, _shift_polyline(right["polyline"], ox, oy), "#DC2626", 5)
    for obj in objects_with_geometry:
        shifted = _shift_bbox(obj["bbox"], ox, oy)
        mask_draw.rectangle(shifted, outline="#F59E0B", width=3)
        geometry = obj.get("target_lane_geometry") or {}
        hint = "IN" if geometry.get("inside_target_lane_band") else "OUT"
        anchor = geometry.get("anchor_point_px")
        if anchor:
            ax, ay = anchor[0] - ox, anchor[1] - oy
            mask_draw.ellipse((ax - 5, ay - 5, ax + 5, ay + 5), fill="#F59E0B")
        _draw_text(mask_draw, (shifted[0], max(0, shifted[1] - 14)), f"{obj['object_id']}:{hint}", "#F59E0B")
    _draw_text(mask_draw, (12, 12), f"target={lane['lane_id']} green | left=blue | right=red | objects=orange", "#111827")
    mask_path = output_dir / "03_lane_mask_whiteboard.png"
    mask.save(mask_path)
    views.append({"id": "lane_mask_whiteboard", "path": str(mask_path.resolve()), "desc": "whiteboard thin-mask view for target lane, boundaries, and object boxes"})

    if getattr(render_alpha_views, "include_signal_alignment_view", False):
        alignment_path = output_dir / "04_signal_alignment_whiteboard.png"
        if _save_signal_alignment_whiteboard((width, height), lane, boundary_match, objects_with_geometry, alignment_path):
            views.append({"id": "signal_alignment_whiteboard", "path": str(alignment_path.resolve()), "desc": "whiteboard for overhead sign/object to target-lane alignment; green projection shows the target lane direction"})

    if getattr(render_alpha_views, "include_signal_ruler_view", False):
        ruler_path = output_dir / "04_signal_lane_ruler_whiteboard.png"
        if _save_signal_lane_ruler_whiteboard(objects_with_geometry, ruler_path):
            views.append({"id": "signal_lane_ruler_whiteboard", "path": str(ruler_path.resolve()), "desc": "structured ruler table: each object anchor is plotted by lane_relative_x against the target lane band; reject left/right adjacent signals"})

    if getattr(render_alpha_views, "include_boundary_pair_view", False):
        pair_path = output_dir / "04_boundary_pair_raw_context.png"
        if _save_boundary_pair_raw_context(image, lane, boundary_match, pair_path):
            views.append({"id": "boundary_pair_raw_context", "path": str(pair_path.resolve()), "desc": "side-by-side lane-band crop: raw unobscured view and same crop with target/left/right overlay for boundary ownership"})

    if getattr(render_alpha_views, "include_lane_band_strip_view", False):
        strip_path = output_dir / "04_lane_band_raw_strip.png"
        if _save_lane_band_raw_strip(image, lane, boundary_match, strip_path):
            views.append({"id": "lane_band_raw_strip", "path": str(strip_path.resolve()), "desc": "high-resolution raw crop of the target lane band and both boundaries; no overlay covers road markings or road text"})

    left_zoom_path = output_dir / "04_left_boundary_raw_zoom.png"
    if _save_boundary_zoom(image, left, lane, left_zoom_path, "left"):
        views.append({"id": "left_boundary_raw_zoom", "path": str(left_zoom_path.resolve()), "desc": "raw image crop for inspecting left_boundary color, dashed/solid, double line, and zigzag without overlay covering the marking"})

    right_zoom_path = output_dir / "05_right_boundary_raw_zoom.png"
    if _save_boundary_zoom(image, right, lane, right_zoom_path, "right"):
        views.append({"id": "right_boundary_raw_zoom", "path": str(right_zoom_path.resolve()), "desc": "raw image crop for inspecting right_boundary color, dashed/solid, double line, and zigzag without overlay covering the marking"})

    contact_path = output_dir / "06_object_contact_sheet.png"
    if _save_object_contact_sheet(image, objects_with_geometry, contact_path, max_objects=max(max_object_crops, 1), object_sort_mode=object_sort_mode):
        views.append({"id": "object_contact_sheet", "path": str(contact_path.resolve()), "desc": "local crops of detected signs/text/signals/road symbols"})

    for idx, obj in enumerate(sorted(objects_with_geometry, key=lambda item: _object_sort_key(item, object_sort_mode))[:max_object_crops], start=1):
        crop_box = _expand_bbox(obj["bbox"], (width, height), margin=0.55)
        crop = image.crop(tuple(crop_box))
        crop_draw = ImageDraw.Draw(crop)
        crop_draw.rectangle(_shift_bbox(obj["bbox"], crop_box[0], crop_box[1]), outline="#F59E0B", width=3)
        _draw_text(crop_draw, (8, 8), f"{obj['object_id']} {obj.get('label_name')}", "#F59E0B")
        crop_path = output_dir / f"{6 + idx:02d}_object_crop_{obj['object_id']}.png"
        crop.save(crop_path)
        views.append({"id": f"object_crop_{obj['object_id']}", "path": str(crop_path.resolve()), "desc": f"crop around {obj['object_id']} ({obj.get('label_name')})"})

    return views


def _boundary_packet(boundary: dict | None, width: int, height: int) -> dict | None:
    if boundary is None:
        return None
    polyline = boundary["polyline"]
    return {
        "boundary_id": boundary["boundary_id"],
        "category_id": boundary.get("category_id"),
        "category_name": boundary.get("category_name"),
        "det_score": boundary.get("score"),
        "polyline_norm1000": norm1000_polyline(polyline, width, height, n=10),
        "bbox_2d": norm1000_bbox(bbox_from_points(polyline), width, height),
    }


def enrich_objects_with_geometry(objects: list[dict], lane: dict, boundary_match, width: int, height: int) -> list[dict]:
    left_polyline = boundary_match.left["polyline"] if boundary_match.left else None
    right_polyline = boundary_match.right["polyline"] if boundary_match.right else None
    enriched = []
    for obj in objects:
        geometry = object_lane_geometry(
            obj["bbox"],
            lane["centerline"],
            left_polyline,
            right_polyline,
            label_id=obj.get("label_id"),
        )
        geometry_norm = {
            "anchor_point_2d": norm1000_point(geometry["anchor_point"], width, height),
            "center_x_at_anchor_y": (
                norm1000_point([geometry["center_x_at_anchor_y"], geometry["anchor_point"][1]], width, height)[0]
                if geometry.get("center_x_at_anchor_y") is not None
                else None
            ),
            "left_x_at_anchor_y": (
                norm1000_point([geometry["left_x_at_anchor_y"], geometry["anchor_point"][1]], width, height)[0]
                if geometry.get("left_x_at_anchor_y") is not None
                else None
            ),
            "right_x_at_anchor_y": (
                norm1000_point([geometry["right_x_at_anchor_y"], geometry["anchor_point"][1]], width, height)[0]
                if geometry.get("right_x_at_anchor_y") is not None
                else None
            ),
            "lane_relative_x": round(geometry["lane_relative_x"], 3) if geometry.get("lane_relative_x") is not None else None,
            "distance_to_centerline": (
                round(geometry["distance_to_centerline"] / max(width, 1) * 1000, 3)
                if geometry.get("distance_to_centerline") is not None
                else None
            ),
            "inside_target_lane_band": geometry["inside_target_lane_band"],
            "relation_hint": geometry["relation_hint"],
        }
        enriched_obj = dict(obj)
        enriched_obj["target_lane_geometry"] = geometry_norm
        enriched_obj["target_lane_geometry"]["anchor_point_px"] = geometry["anchor_point"]
        enriched.append(enriched_obj)
    return enriched


def build_alpha_packet(sample, lane: dict, boundary_match, object_sort_mode: str = "geometry") -> dict:
    width, height = sample.width, sample.height
    objects_with_geometry = enrich_objects_with_geometry(sample.objects, lane, boundary_match, width, height)
    return {
        "pipeline": "alpha_packet",
        "image_id": sample.image_id,
        "image_size": {"width": width, "height": height, "coordinate": "norm_1000"},
        "target_lane": {
            "lane_id": lane["lane_id"],
            "centerline_norm1000": norm1000_polyline(lane["centerline"], width, height, n=8),
            "bbox_2d": norm1000_bbox(bbox_from_points(lane["centerline"]), width, height),
        },
        "left_boundary": _boundary_packet(boundary_match.left, width, height),
        "right_boundary": _boundary_packet(boundary_match.right, width, height),
        "objects": [
            {
                "object_id": obj["object_id"],
                "label_id": obj.get("label_id"),
                "label_name": obj.get("label_name"),
                "det_score": obj.get("score"),
                "bbox_2d": norm1000_bbox(obj["bbox"], width, height),
                "anchor_point_2d": obj["target_lane_geometry"]["anchor_point_2d"],
                "target_lane_geometry": {
                    key: value
                    for key, value in obj["target_lane_geometry"].items()
                    if key != "anchor_point_px"
                },
            }
            for obj in sorted(objects_with_geometry, key=lambda item: _object_sort_key(item, object_sort_mode))
        ],
        "views": [],
        "closed_set_labels": ["bus", "tidal", "variable", "bicycle", "normal"],
    }


def run(args: argparse.Namespace) -> dict:
    category_ids = set(args.line_category_ids) if args.line_category_ids else None
    samples = iter_dataset_samples(Path(args.data_dir), limit=args.limit_images, line_category_ids=category_ids)
    runtime = resolve_runtime(
        config_path=args.resource_config,
        profile=args.resource_profile,
        model=args.model,
        base_url=args.base_url,
        chat_url=args.chat_url,
        auth_scheme=args.auth_scheme,
        timeout=args.timeout,
        default_model="qwen3.5-vl",
        default_timeout=120,
    )
    client = None if args.dry_run else QwenOpenAIClient(
        api_key=runtime.api_key,
        base_url=runtime.base_url,
        chat_url=runtime.chat_url,
        auth_scheme=runtime.auth_scheme,
        timeout=runtime.timeout or 120,
        headers=runtime.headers,
        default_body=runtime.default_body,
        response_format=runtime.response_format,
        enable_thinking=runtime.enable_thinking,
    )

    jobs = [(sample, lane) for sample in samples for lane in sample.lanes]
    existing_records = []
    done_keys = set()
    incremental_path = Path(args.incremental_jsonl) if args.incremental_jsonl else None
    if incremental_path is not None:
        if args.resume_from_incremental and incremental_path.exists():
            existing_records = read_json_or_jsonl(incremental_path)
            done_keys = {(record["image_id"], record["lane_id"]) for record in existing_records}
            jobs = [(sample, lane) for sample, lane in jobs if (sample.image_id, lane["lane_id"]) not in done_keys]
            print(f"[alpha] resumed {len(existing_records)} records from {incremental_path}", file=sys.stderr, flush=True)
        elif incremental_path.exists():
            incremental_path.unlink()
    render_alpha_views.include_signal_alignment_view = args.include_signal_alignment_view
    render_alpha_views.include_signal_ruler_view = args.include_signal_ruler_view
    render_alpha_views.include_boundary_pair_view = args.include_boundary_pair_view
    render_alpha_views.include_lane_band_strip_view = args.include_lane_band_strip_view

    def process(job: tuple) -> dict:
        sample, lane = job
        boundary_match = match_left_right_boundaries(
            lane["centerline"],
            sample.boundaries,
            mode=args.boundary_match_mode,
        )
        packet = build_alpha_packet(sample, lane, boundary_match, args.object_sort_mode)
        views = render_alpha_views(sample, lane, boundary_match, Path(args.panel_dir), args.max_object_crops, args.object_sort_mode)
        if runtime.max_images_per_request and runtime.max_images_per_request > 0:
            views = views[: runtime.max_images_per_request]
        packet["views"] = [
            {
                "id": view["id"],
                "path": view["path"],
                "desc": view["desc"],
                "upload_index": idx,
            }
            for idx, view in enumerate(views, start=1)
        ]
        if args.prompt_mode == "compact":
            user_prompt = alpha_compact_prompt(packet)
        elif args.prompt_mode == "line":
            user_prompt = alpha_line_prompt(packet)
        elif args.prompt_mode == "line_recall":
            user_prompt = alpha_line_recall_prompt(packet)
        else:
            user_prompt = alpha_prompt(packet)
        request_payload = {
            "system_prompt": SYSTEM_PROMPT,
            "user_prompt": user_prompt,
            "packet": packet,
            "images": packet["views"],
            "resource_profile": runtime.profile,
            "max_images_per_request": runtime.max_images_per_request,
            "messages_preview": {
                "image_paths": [view["path"] for view in packet["views"]],
                "content_order": ["images_in_view_order", "text"],
            },
        }

        raw_response = None
        parsed = None
        status = "pending" if args.dry_run else "ok"
        error = None
        if not args.dry_run:
            try:
                messages = build_openai_messages(
                    SYSTEM_PROMPT,
                    user_prompt,
                    [Path(view["path"]) for view in packet["views"]],
                    image_max_side=runtime.image_max_side,
                    image_jpeg_quality=runtime.image_jpeg_quality,
                )
                last_exc: Exception | None = None
                for attempt in range(args.request_retries + 1):
                    try:
                        raw_response = client.chat(
                            model=runtime.model or "qwen3.5-vl",
                            messages=messages,
                            temperature=args.temperature,
                            max_tokens=args.max_tokens,
                        )
                        last_exc = None
                        break
                    except Exception as exc:  # noqa: BLE001 - retry transient gateway/backend failures.
                        last_exc = exc
                        if attempt >= args.request_retries:
                            break
                        time.sleep(args.retry_sleep_seconds)
                if last_exc is not None:
                    raise last_exc
                parsed = extract_json_object(raw_response)
            except Exception as exc:  # noqa: BLE001 - keep per-lane failures in unified output.
                status = "error"
                error = str(exc)

        return make_result_record(
            pipeline="alpha",
            model=args.model,
            sample=sample,
            lane=lane,
            boundary_match=boundary_match,
            request_payload=request_payload,
            raw_response=raw_response,
            parsed_response=parsed,
            status=status,
            error=error,
        )

    records = list(existing_records)
    if args.workers <= 1:
        for idx, job in enumerate(jobs, start=1):
            record = process(job)
            records.append(record)
            if incremental_path is not None:
                append_jsonl(incremental_path, [record])
            print(f"[alpha] {len(existing_records) + idx}/{len(existing_records) + len(jobs)}", file=sys.stderr, flush=True)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_map = {executor.submit(process, job): idx for idx, job in enumerate(jobs)}
            for done, future in enumerate(as_completed(future_map), start=1):
                record = future.result()
                records.append(record)
                if incremental_path is not None:
                    append_jsonl(incremental_path, [record])
                print(f"[alpha] {len(existing_records) + done}/{len(existing_records) + len(jobs)}", file=sys.stderr, flush=True)

    records.sort(
        key=lambda record: (
            record["image_id"],
            int(str(record["source_lane_id"])) if str(record["source_lane_id"]).isdigit() else str(record["source_lane_id"]),
        )
    )

    output = {
        "pipeline": "alpha",
        "model": runtime.model,
        "resource_profile": runtime.profile,
        "dry_run": args.dry_run,
        "num_images": len(samples),
        "num_records": len(records),
        "records": records,
    }
    dump_json(Path(args.output), output)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output", default="outputs/pipeline_alpha_results.json")
    parser.add_argument("--resource-config", default=None, help="JSON file containing selectable VLM resource profiles.")
    parser.add_argument("--resource-profile", default=None, help="Profile name from --resource-config, or VLM_RESOURCE_PROFILE.")
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--chat-url", default=None)
    parser.add_argument("--auth-scheme", default=None, choices=["auto", "raw", "bearer", "none", "no_auth"])
    parser.add_argument("--limit-images", type=int, default=None)
    parser.add_argument("--line-category-ids", type=int, nargs="*", default=None)
    parser.add_argument("--dry-run", action="store_true", help="Build packets/prompts without calling the VLM API.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--request-retries", type=int, default=0, help="Retry each VLM request after transient gateway/backend failures.")
    parser.add_argument("--retry-sleep-seconds", type=float, default=5.0)
    parser.add_argument("--panel-dir", default="outputs/pipeline_alpha_panels")
    parser.add_argument("--max-object-crops", type=int, default=4)
    parser.add_argument("--object-sort-mode", choices=("geometry", "semantic_area"), default="geometry")
    parser.add_argument("--include-signal-alignment-view", action="store_true")
    parser.add_argument("--include-signal-ruler-view", action="store_true")
    parser.add_argument("--include-boundary-pair-view", action="store_true")
    parser.add_argument("--include-lane-band-strip-view", action="store_true")
    parser.add_argument(
        "--boundary-match-mode",
        choices=("nearest", "strict_y_overlap", "overlap_rescue"),
        default="nearest",
        help="nearest keeps legacy matching; strict_y_overlap hard-filters low-overlap candidates; overlap_rescue only replaces zero-overlap legacy winners with close overlapping alternatives.",
    )
    parser.add_argument(
        "--prompt-mode",
        choices=("standard", "compact", "line", "line_recall"),
        default="standard",
    )
    parser.add_argument("--incremental-jsonl", default=None, help="Append each completed lane record to this JSONL file.")
    parser.add_argument("--resume-from-incremental", action="store_true", help="Skip records already present in --incremental-jsonl.")
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps({k: result[k] for k in ("pipeline", "model", "resource_profile", "dry_run", "num_images", "num_records")}, ensure_ascii=False))
