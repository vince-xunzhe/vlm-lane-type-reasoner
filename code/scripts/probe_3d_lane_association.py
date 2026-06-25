#!/usr/bin/env python3
"""Probe camera-space object-to-lane soft association.

This diagnostic script combines DA3 depth, SAM road-surface masks, 2D lane
centerlines, and object boxes.  It does not change VLM inference outputs; it
only writes per-frame association scores and visual overlays for inspection.
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
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


SIGN_SIGNAL_LABEL_IDS = {2, 6, 9}
ROAD_MARKING_LABEL_IDS = {0, 1, 4, 5, 10}
ALWAYS_FILTER_LABEL_IDS = {6}

LABEL_NAMES = {
    0: "bus_text_gong",
    1: "bus_text_jiao",
    2: "bus_related_time_restriction_sign",
    4: "variable_text_ke",
    5: "variable_text_bian",
    6: "mixed_lane_signal_candidate",
    9: "bicycle_sign",
    10: "bicycle_icon",
}

PALETTE_RGB = [
    (255, 210, 0),
    (50, 210, 80),
    (40, 140, 255),
    (220, 70, 255),
    (255, 120, 60),
    (40, 230, 220),
    (180, 220, 40),
    (255, 80, 150),
]


@dataclass(frozen=True)
class Paths:
    base_dir: Path
    out_dir: Path
    artifact_dir: Path
    center_dir: Path
    objects_dir: Path
    depth_dir: Path
    sam3_dir: Path
    image_dirs: tuple[Path, ...]


@dataclass
class DepthScene:
    depth: np.ndarray
    conf: np.ndarray
    intrinsics: np.ndarray
    road_mask: np.ndarray
    image_width: int
    image_height: int
    min_conf: float

    @property
    def depth_shape(self) -> tuple[int, int]:
        return int(self.depth.shape[0]), int(self.depth.shape[1])

    def image_to_depth_xy(self, x: float, y: float) -> tuple[int, int]:
        dh, dw = self.depth_shape
        u = int(round(float(x) / max(self.image_width - 1, 1) * (dw - 1)))
        v = int(round(float(y) / max(self.image_height - 1, 1) * (dh - 1)))
        return max(0, min(dw - 1, u)), max(0, min(dh - 1, v))

    def depth_to_camera(self, u: float, v: float, z: float) -> list[float]:
        intr = self.intrinsics
        fx = float(intr[0, 0]) if intr.size else 1.0
        fy = float(intr[1, 1]) if intr.size else fx
        cx = float(intr[0, 2]) if intr.size else self.depth.shape[1] / 2
        cy = float(intr[1, 2]) if intr.size else self.depth.shape[0] / 2
        fx = fx if abs(fx) > 1e-6 else 1.0
        fy = fy if abs(fy) > 1e-6 else 1.0
        return [float((u - cx) / fx * z), float((v - cy) / fy * z), float(z)]

    def robust_depth_at_image_xy(self, x: float, y: float, radius: int = 3) -> dict[str, Any]:
        u, v = self.image_to_depth_xy(x, y)
        dh, dw = self.depth_shape
        x0, x1 = max(0, u - radius), min(dw, u + radius + 1)
        y0, y1 = max(0, v - radius), min(dh, v + radius + 1)
        values = self.depth[y0:y1, x0:x1].reshape(-1)
        conf = self.conf[y0:y1, x0:x1].reshape(-1)
        good = np.isfinite(values) & (values > 0) & np.isfinite(conf) & (conf >= self.min_conf)
        if not np.any(good):
            return {"ok": False, "u": u, "v": v, "count": 0}
        vals = values[good].astype(float)
        z = float(np.median(vals))
        return {
            "ok": True,
            "depth": z,
            "u": u,
            "v": v,
            "count": int(vals.size),
            "conf_median": float(np.median(conf[good])),
            "xyz": self.depth_to_camera(u, v, z),
        }

    def robust_depth_in_bbox(self, bbox: list[float], shrink_ratio: float = 0.12) -> dict[str, Any]:
        x1, y1, x2, y2 = [float(v) for v in bbox]
        if x2 <= x1 or y2 <= y1:
            return {"ok": False, "reason": "invalid_bbox", "count": 0, "valid_ratio": 0.0}
        dx = (x2 - x1) * shrink_ratio
        dy = (y2 - y1) * shrink_ratio
        x1, x2 = x1 + dx, x2 - dx
        y1, y2 = y1 + dy, y2 - dy
        u0, v0 = self.image_to_depth_xy(x1, y1)
        u1, v1 = self.image_to_depth_xy(x2, y2)
        x0, xh = sorted([u0, u1])
        y0, yh = sorted([v0, v1])
        xh = min(self.depth.shape[1] - 1, max(x0, xh))
        yh = min(self.depth.shape[0] - 1, max(y0, yh))
        values = self.depth[y0 : yh + 1, x0 : xh + 1].reshape(-1)
        conf = self.conf[y0 : yh + 1, x0 : xh + 1].reshape(-1)
        total = int(values.size)
        good = np.isfinite(values) & (values > 0) & np.isfinite(conf) & (conf >= self.min_conf)
        if not np.any(good):
            return {"ok": False, "reason": "no_valid_depth", "count": 0, "valid_ratio": 0.0, "total": total}
        vals = values[good].astype(float)
        return {
            "ok": True,
            "depth_median": float(np.median(vals)),
            "depth_p25": float(np.percentile(vals, 25)),
            "depth_p75": float(np.percentile(vals, 75)),
            "count": int(vals.size),
            "total": total,
            "valid_ratio": float(vals.size / max(total, 1)),
            "conf_min": float(np.min(conf[good])),
            "conf_p25": float(np.percentile(conf[good], 25)),
            "conf_median": float(np.median(conf[good])),
            "conf_p75": float(np.percentile(conf[good], 75)),
            "conf_p90": float(np.percentile(conf[good], 90)),
            "conf_max": float(np.max(conf[good])),
        }

    def road_mask_at_image_xy(self, x: float, y: float) -> bool:
        xi = int(round(x))
        yi = int(round(y))
        h, w = self.road_mask.shape
        if xi < 0 or yi < 0 or xi >= w or yi >= h:
            return False
        return bool(self.road_mask[yi, xi])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--las-dir", type=Path, default=None)
    parser.add_argument("--base-dir", type=Path, default=None)
    parser.add_argument("--round-name", default=os.environ.get("ROUND_NAME", ""))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--artifact-dir", type=Path, default=None)
    parser.add_argument("--instances-yaml", type=Path, default=None)
    parser.add_argument("--frames", default="", help="Comma-separated frame stems, or a text file with one stem per line.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--sign-max-depth-m", type=float, default=100.0)
    parser.add_argument("--enable-sign-depth-confidence-filter", action="store_true", help="Filter sign/signal objects by DA3 confidence thresholds. Disabled by default.")
    parser.add_argument("--sign-min-conf-median", type=float, default=1.01)
    parser.add_argument("--sign-min-conf-p75", type=float, default=1.01)
    parser.add_argument("--min-conf", type=float, default=1.0)
    parser.add_argument("--min-object-depth-valid-ratio", type=float, default=0.03)
    parser.add_argument("--lane-samples", type=int, default=96)
    parser.add_argument("--lane-min-depth-samples", type=int, default=4)
    parser.add_argument("--lane-depth-radius", type=int, default=3)
    parser.add_argument("--lane-extension-samples", type=int, default=96)
    parser.add_argument("--lane-extension-max-px", type=float, default=900.0)
    parser.add_argument("--lane-search-radii", default="", help="Deprecated; lateral road-mask search is disabled.")
    parser.add_argument("--lane-band-default-width-m", type=float, default=3.5)
    parser.add_argument("--lane-band-distance-scale-m", type=float, default=0.75)
    parser.add_argument("--lane-band-center-scale-m", type=float, default=2.0)
    parser.add_argument("--sign-lane-z-window-m", type=float, default=18.0, help="Deprecated; BEV scoring uses normalized XZ distance.")
    parser.add_argument("--sign-lane-nearest-k", type=int, default=16)
    parser.add_argument("--sign-lateral-scale-m", type=float, default=1.8)
    parser.add_argument("--sign-longitudinal-scale-m", type=float, default=18.0)
    parser.add_argument("--sign-ground-search-px", type=int, default=320, help="Deprecated; sign ground scatter search is disabled.")
    parser.add_argument("--max-ground-candidates", type=int, default=220)
    parser.add_argument("--overwrite", action="store_true")
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
    if (cwd / "center_line_2d").exists() and (cwd / "objects" / "pred").exists():
        return cwd
    raise SystemExit("Provide --base-dir or --las-dir.")


def build_paths(base_dir: Path, args: argparse.Namespace) -> Paths:
    round_name = str(args.round_name or "").strip()
    if args.output_dir is not None:
        out_dir = args.output_dir.expanduser().resolve()
    elif round_name:
        out_dir = base_dir / "vis_debug" / round_name / "association_3d"
    else:
        out_dir = base_dir / "vis_debug" / "association_3d"

    if args.artifact_dir is not None:
        artifact_dir = args.artifact_dir.expanduser().resolve()
    elif round_name:
        artifact_dir = base_dir / "inference" / round_name / "association_3d"
    else:
        artifact_dir = base_dir / "inference" / "association_3d"

    las_dir = base_dir.parent
    image_dirs = (
        las_dir / las_dir.name / "Data" / "Img" / "Camera0",
        base_dir / "inference" / "output" / "input" / "images",
        base_dir,
        las_dir,
    )
    return Paths(
        base_dir=base_dir,
        out_dir=out_dir,
        artifact_dir=artifact_dir,
        center_dir=base_dir / "center_line_2d" / "output",
        objects_dir=base_dir / "objects" / "pred",
        depth_dir=base_dir / "depth",
        sam3_dir=base_dir / "sam3",
        image_dirs=tuple(path for path in image_dirs if path.exists()),
    )


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)


def _clean_yaml_scalar(value: str) -> str:
    value = value.split("#", 1)[0].strip().rstrip(",")
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        value = value[1:-1]
    return value.strip()


def load_instances_yaml(path: Path) -> list[str] | None:
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
        if key.strip() not in {"instances", "frames", "frame_ids"}:
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


def discover_frames(paths: Paths, limit: int = 0) -> list[str]:
    stems = {path.stem for path in paths.center_dir.glob("*.json")}
    stems.update(path.stem for path in paths.objects_dir.glob("*.json"))
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
    requested_set = {Path(item).stem for item in requested}
    return [frame for frame in frames if frame in requested_set]


def apply_instance_selection(
    frames: list[str],
    instances_yaml: Path,
    frame_arg: str,
) -> tuple[list[str], list[str] | None, list[str]]:
    requested = load_instances_yaml(instances_yaml)
    if requested is None:
        selected = frames
        requested_instances = None
        missing: list[str] = []
    else:
        available = set(frames)
        selected = [item for item in requested if item in available]
        missing = [item for item in requested if item not in available]
        requested_instances = requested
    selected = filter_frames(selected, frame_arg)
    return selected, requested_instances, missing


def image_path_for_frame(paths: Paths, frame: str, object_data: dict[str, Any]) -> Path | None:
    img_path = object_data.get("img_path") if isinstance(object_data, dict) else None
    candidates: list[Path] = []
    for image_dir in paths.image_dirs:
        candidates.append(image_dir / f"{frame}.jpg")
        candidates.append(image_dir / f"{frame}.png")
    if img_path:
        candidates.append(Path(str(img_path)))
    for path in candidates:
        if path.exists():
            return path
    return None


def sam_mask_path(paths: Paths, frame: str) -> Path | None:
    json_path = paths.sam3_dir / "json" / f"{frame}.json"
    data = load_json(json_path, default={})
    selected = (((data or {}).get("result") or {}).get("selected") or []) if isinstance(data, dict) else []
    if selected:
        rel = selected[0].get("mask_path")
        if rel:
            path = paths.sam3_dir / str(rel)
            if path.exists():
                return path
    fallback = paths.sam3_dir / "masks" / f"{frame}_mask_000.png"
    return fallback if fallback.exists() else None


def depth_npz_path(paths: Paths, frame: str) -> Path:
    return paths.depth_dir / f"{frame}_jpg" / "exports" / "mini_npz" / "results.npz"


def load_depth_scene(paths: Paths, frame: str, image_size: tuple[int, int], min_conf: float) -> DepthScene:
    npz_path = depth_npz_path(paths, frame)
    if not npz_path.exists():
        raise FileNotFoundError(f"missing depth npz: {npz_path}")
    mask_path = sam_mask_path(paths, frame)
    if mask_path is None:
        raise FileNotFoundError(f"missing sam3 road mask for frame: {frame}")
    npz = np.load(npz_path, allow_pickle=True)
    depth = np.asarray(npz["depth"][0], dtype=np.float32)
    conf = np.asarray(npz["conf"][0], dtype=np.float32)
    intrinsics = np.asarray(npz["intrinsics"][0], dtype=np.float32)
    road_mask = np.asarray(Image.open(mask_path).convert("L")) > 0
    image_width, image_height = image_size
    if road_mask.shape != (image_height, image_width):
        road_mask = cv2.resize(
            road_mask.astype(np.uint8),
            (image_width, image_height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
    return DepthScene(depth=depth, conf=conf, intrinsics=intrinsics, road_mask=road_mask, image_width=image_width, image_height=image_height, min_conf=min_conf)


def parse_points(value: Any) -> list[list[float]]:
    points: list[list[float]] = []
    for item in value or []:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        try:
            points.append([float(item[0]), float(item[1])])
        except (TypeError, ValueError):
            continue
    return points


def load_lanes(center_path: Path) -> list[dict[str, Any]]:
    payload = load_json(center_path, default={})
    raw_lanes = payload.get("lane") or payload.get("lanes") or payload.get("lines") or []
    lanes: list[dict[str, Any]] = []
    for idx, lane in enumerate(raw_lanes):
        if not isinstance(lane, dict):
            continue
        points = parse_points(lane.get("points") or lane.get("points_2d") or lane.get("polyline"))
        if not points:
            continue
        lane_id = str(lane.get("id", idx))
        lanes.append({"lane_id": lane_id, "index": idx, "points": points, "attribute": lane.get("attribute")})
    return lanes


def load_objects(objects_path: Path) -> list[dict[str, Any]]:
    payload = load_json(objects_path, default={})
    labels = payload.get("labels_info", payload.get("labels", [])) if isinstance(payload, dict) else []
    scores = payload.get("scores_info", payload.get("scores", [])) if isinstance(payload, dict) else []
    boxes = payload.get("bboxes_info", payload.get("bboxes", [])) if isinstance(payload, dict) else []
    objects = []
    for idx, bbox in enumerate(boxes):
        if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            continue
        label_id = labels[idx] if idx < len(labels) else None
        try:
            label_int = int(label_id) if label_id is not None else None
        except (TypeError, ValueError):
            label_int = None
        objects.append(
            {
                "object_id": f"O{idx}",
                "index": idx,
                "label_id": label_int,
                "label_name": LABEL_NAMES.get(label_int, f"det_label_{label_id}"),
                "score": float(scores[idx]) if idx < len(scores) else None,
                "bbox": [float(v) for v in bbox[:4]],
            }
        )
    return objects


def sample_polyline(points: list[list[float]], n: int) -> list[list[float]]:
    if not points:
        return []
    if len(points) == 1:
        return [points[0] for _ in range(max(1, n))]
    segs = []
    total = 0.0
    cumulative = [0.0]
    for a, b in zip(points[:-1], points[1:]):
        length = math.dist(a, b)
        segs.append((a, b, length))
        total += length
        cumulative.append(total)
    if total <= 1e-6:
        return [points[0] for _ in range(max(1, n))]
    out: list[list[float]] = []
    seg_idx = 0
    for idx in range(max(1, n)):
        target = total * idx / max(n - 1, 1)
        while seg_idx < len(segs) - 1 and cumulative[seg_idx + 1] < target:
            seg_idx += 1
        a, b, length = segs[seg_idx]
        if length <= 1e-6:
            out.append([float(a[0]), float(a[1])])
            continue
        t = max(0.0, min(1.0, (target - cumulative[seg_idx]) / length))
        out.append([float(a[0] + t * (b[0] - a[0])), float(a[1] + t * (b[1] - a[1]))])
    return out


def ego_extension_line(
    points: list[list[float]],
    n: int,
    image_width: int,
    image_height: int,
    max_px: float,
) -> dict[str, Any]:
    if len(points) < 2:
        return {"points": [], "source": "ego_extension", "reason": "not_enough_points"}
    endpoints = [
        (np.asarray(points[0], dtype=float), np.asarray(points[1], dtype=float)),
        (np.asarray(points[-1], dtype=float), np.asarray(points[-2], dtype=float)),
    ]
    anchor, neighbor = max(endpoints, key=lambda item: float(item[0][1]))
    direction = anchor - neighbor
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-6:
        return {"points": [], "source": "ego_extension", "reason": "degenerate_ego_direction"}
    direction = direction / norm
    out: list[list[float]] = []
    for step in np.linspace(4.0, max(4.0, max_px), max(1, n)):
        p = anchor + direction * float(step)
        if 0 <= p[0] < image_width and 0 <= p[1] < image_height:
            out.append([float(p[0]), float(p[1])])
    return {
        "points": out,
        "anchor": [float(anchor[0]), float(anchor[1])],
        "direction": [float(direction[0]), float(direction[1])],
        "source": "ego_extension",
        "reason": None,
    }


def forward_extension_line(
    points: list[list[float]],
    n: int,
    image_width: int,
    image_height: int,
    max_px: float,
) -> dict[str, Any]:
    if len(points) < 2:
        return {"points": [], "source": "forward_extension", "reason": "not_enough_points"}
    first = np.asarray(points[0], dtype=float)
    last = np.asarray(points[-1], dtype=float)
    near, far = (first, last) if float(first[1]) >= float(last[1]) else (last, first)
    direction = far - near
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-6:
        return {"points": [], "source": "forward_extension", "reason": "degenerate_forward_direction"}
    direction = direction / norm
    out: list[list[float]] = []
    for step in np.linspace(4.0, max(4.0, max_px), max(1, n)):
        p = near + direction * float(step)
        if 0 <= p[0] < image_width and 0 <= p[1] < image_height:
            out.append([float(p[0]), float(p[1])])
    return {
        "points": out,
        "anchor": [float(near[0]), float(near[1])],
        "direction": [float(direction[0]), float(direction[1])],
        "source": "forward_extension",
        "reason": None,
    }


def fit_lane_bev_profile(samples: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [sample for sample in samples if sample.get("ok") and sample.get("xyz")]
    if len(valid) < 2:
        return {"ok": False, "reason": "not_enough_samples", "sample_count": len(valid)}
    xyz = np.asarray([sample["xyz"] for sample in valid], dtype=float)
    x = xyz[:, 0]
    z = xyz[:, 2]
    good = np.isfinite(x) & np.isfinite(z) & (z > 1e-6)
    if int(good.sum()) < 2:
        return {"ok": False, "reason": "invalid_bev_samples", "sample_count": int(good.sum())}
    x = x[good]
    z = z[good]
    if float(np.ptp(z)) <= 1e-6:
        return {
            "ok": True,
            "model": "constant_x",
            "x_const": float(np.median(x)),
            "z_min": float(np.min(z)),
            "z_max": float(np.max(z)),
            "sample_count": int(x.size),
        }
    slope, intercept = np.polyfit(z, x, 1)
    residual = x - (slope * z + intercept)
    return {
        "ok": True,
        "model": "linear_x_of_z",
        "slope_x_per_z": float(slope),
        "intercept_x": float(intercept),
        "z_min": float(np.min(z)),
        "z_max": float(np.max(z)),
        "sample_count": int(x.size),
        "residual_median_abs": float(np.median(np.abs(residual))),
    }


def lane_center_x_at_z(lane: dict[str, Any], z: float) -> float | None:
    fit = lane.get("bev_fit") or {}
    if not fit.get("ok"):
        return None
    if fit.get("model") == "constant_x":
        return float(fit["x_const"])
    if fit.get("model") == "linear_x_of_z":
        return float(fit["slope_x_per_z"]) * float(z) + float(fit["intercept_x"])
    return None


def lane_bands_at_z(lanes: list[dict[str, Any]], z: float, default_width_m: float) -> list[dict[str, Any]]:
    centers = []
    for lane in lanes:
        x = lane_center_x_at_z(lane, z)
        if x is None or not math.isfinite(x):
            continue
        centers.append({"lane": lane, "lane_id": lane["lane_id"], "center_x": float(x)})
    centers.sort(key=lambda item: item["center_x"])
    if not centers:
        return []
    if len(centers) == 1:
        half = float(default_width_m) * 0.5
        centers[0]["left_x"] = centers[0]["center_x"] - half
        centers[0]["right_x"] = centers[0]["center_x"] + half
        return centers

    for idx, item in enumerate(centers):
        center = item["center_x"]
        if idx == 0:
            right = 0.5 * (center + centers[idx + 1]["center_x"])
            half = max(float(default_width_m) * 0.5, abs(right - center))
            left = center - half
        elif idx == len(centers) - 1:
            left = 0.5 * (centers[idx - 1]["center_x"] + center)
            half = max(float(default_width_m) * 0.5, abs(center - left))
            right = center + half
        else:
            left = 0.5 * (centers[idx - 1]["center_x"] + center)
            right = 0.5 * (center + centers[idx + 1]["center_x"])
        if left > right:
            left, right = right, left
        item["left_x"] = float(left)
        item["right_x"] = float(right)
    return centers


def lane_depth_profile(scene: DepthScene, lane: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    base_samples = sample_polyline(lane["points"], args.lane_samples)
    used_extension = False
    extension_info: dict[str, Any] | None = None
    extension_attempts: list[dict[str, Any]] = []

    def collect(samples: list[list[float]], source: str) -> list[dict[str, Any]]:
        collected = []
        for x, y in samples:
            item: dict[str, Any] = {
                "x": x,
                "y": y,
                "road_x": x,
                "road_y": y,
                "road_radius": 0,
                "road_found": False,
                "sample_source": source,
            }
            if not scene.road_mask_at_image_xy(x, y):
                item["reason"] = "not_on_road_mask"
                collected.append(item)
                continue
            depth_info = scene.robust_depth_at_image_xy(x, y, radius=args.lane_depth_radius)
            item["road_found"] = True
            item.update(depth_info)
            collected.append(item)
        return collected

    samples = collect(base_samples, "centerline_exact")
    depth_count = sum(1 for sample in samples if sample.get("ok"))
    if depth_count < args.lane_min_depth_samples:
        used_extension = True
        for source, builder in (
            ("forward_extension", forward_extension_line),
            ("ego_extension", ego_extension_line),
        ):
            info = builder(
                lane["points"],
                args.lane_extension_samples,
                scene.image_width,
                scene.image_height,
                args.lane_extension_max_px,
            )
            extension_samples = info.get("points") or []
            collected = collect(extension_samples, source)
            attempt = {key: value for key, value in info.items() if key != "points"}
            attempt["sample_count"] = len(extension_samples)
            attempt["road_hit_count"] = sum(1 for sample in collected if sample.get("road_found"))
            attempt["depth_count"] = sum(1 for sample in collected if sample.get("ok"))
            extension_attempts.append(attempt)
            samples.extend(collected)
            depth_count = sum(1 for sample in samples if sample.get("ok"))
            if depth_count >= args.lane_min_depth_samples:
                break
        extension_info = {
            "attempts": extension_attempts,
            "selected_source": next((item.get("source") for item in extension_attempts if item.get("depth_count", 0) > 0), None),
        }

    depths = [float(sample["depth"]) for sample in samples if sample.get("ok")]
    xyz = [sample["xyz"] for sample in samples if sample.get("ok") and sample.get("xyz")]
    road_hits = sum(1 for sample in samples if sample.get("road_found"))
    extension_road_hits = sum(1 for sample in samples if sample.get("sample_source") != "centerline_exact" and sample.get("road_found"))
    forward_extension_road_hits = sum(1 for sample in samples if sample.get("sample_source") == "forward_extension" and sample.get("road_found"))
    ego_extension_road_hits = sum(1 for sample in samples if sample.get("sample_source") == "ego_extension" and sample.get("road_found"))
    valid_for_association = len(depths) >= args.lane_min_depth_samples
    invalid_reason = None if valid_for_association else "insufficient_strict_road_depth_samples"
    bev_fit = fit_lane_bev_profile(samples) if valid_for_association else {"ok": False, "reason": invalid_reason, "sample_count": len(depths)}
    return {
        "lane_id": lane["lane_id"],
        "index": lane["index"],
        "attribute": lane.get("attribute"),
        "points": lane["points"],
        "sampling_policy": "centerline_exact_then_forward_extension_then_ego_extension",
        "sample_count": len(samples),
        "road_hit_count": road_hits,
        "road_hit_ratio": road_hits / max(len(samples), 1),
        "extension_road_hit_count": extension_road_hits,
        "forward_extension_road_hit_count": forward_extension_road_hits,
        "ego_extension_road_hit_count": ego_extension_road_hits,
        "depth_count": len(depths),
        "depth_valid_ratio": len(depths) / max(len(samples), 1),
        "depth_median": float(np.median(depths)) if depths else None,
        "depth_p25": float(np.percentile(depths, 25)) if depths else None,
        "depth_p75": float(np.percentile(depths, 75)) if depths else None,
        "used_extension": used_extension,
        "extension_info": extension_info,
        "valid_for_association": valid_for_association,
        "invalid_reason": invalid_reason,
        "bev_fit": bev_fit,
        "samples": samples,
        "xyz": xyz if valid_for_association else [],
    }


def object_kind(label_id: int | None) -> str:
    if label_id in SIGN_SIGNAL_LABEL_IDS:
        return "sign_signal"
    if label_id in ROAD_MARKING_LABEL_IDS:
        return "road_marking"
    return "unknown"


def sample_road_points_in_box(
    scene: DepthScene,
    box: tuple[float, float, float, float],
    max_points: int,
) -> list[dict[str, Any]]:
    x1, y1, x2, y2 = box
    h, w = scene.road_mask.shape
    x0, xh = int(max(0, math.floor(x1))), int(min(w - 1, math.ceil(x2)))
    y0, yh = int(max(0, math.floor(y1))), int(min(h - 1, math.ceil(y2)))
    if xh <= x0 or yh <= y0:
        return []
    ys, xs = np.where(scene.road_mask[y0 : yh + 1, x0 : xh + 1])
    if xs.size == 0:
        return []
    xs = xs + x0
    ys = ys + y0
    if xs.size > max_points:
        order = np.linspace(0, xs.size - 1, max_points).astype(int)
        xs = xs[order]
        ys = ys[order]
    points = []
    for x, y in zip(xs, ys):
        info = scene.robust_depth_at_image_xy(float(x), float(y))
        if not info.get("ok"):
            continue
        points.append({"x": float(x), "y": float(y), "depth": info["depth"], "xyz": info["xyz"]})
    return points


def object_ground_candidates(scene: DepthScene, obj: dict[str, Any], kind: str, args: argparse.Namespace) -> list[dict[str, Any]]:
    x1, y1, x2, y2 = obj["bbox"]
    if kind != "road_marking":
        return []
    box = (x1, y1, x2, y2)
    return sample_road_points_in_box(scene, box, args.max_ground_candidates)


def object_depth_anchor(scene: DepthScene, obj: dict[str, Any], body_depth: dict[str, Any]) -> dict[str, Any]:
    if not body_depth.get("ok"):
        return {"ok": False, "reason": body_depth.get("reason", "invalid_body_depth")}
    x1, y1, x2, y2 = [float(v) for v in obj["bbox"]]
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    z = float(body_depth["depth_median"])
    u, v = scene.image_to_depth_xy(cx, cy)
    xyz = scene.depth_to_camera(u, v, z)
    return {
        "ok": True,
        "image_xy": [float(cx), float(cy)],
        "depth_xy": [int(u), int(v)],
        "depth": z,
        "xyz": xyz,
        "bev_xz": [float(xyz[0]), float(xyz[2])],
    }


def score_object_to_lanes(candidates: list[dict[str, Any]], lanes: list[dict[str, Any]]) -> dict[str, Any]:
    if not candidates:
        return {"scores": [], "reason": "no_ground_candidates"}
    cand_xyz = np.asarray([item["xyz"] for item in candidates if item.get("xyz")], dtype=float)
    if cand_xyz.size == 0:
        return {"scores": [], "reason": "no_candidate_xyz"}
    raw_scores = []
    details = []
    for lane in lanes:
        lane_xyz = np.asarray(lane.get("xyz") or [], dtype=float)
        if lane_xyz.size == 0:
            raw_scores.append(0.0)
            details.append({"lane_id": lane["lane_id"], "raw_score": 0.0, "reason": "no_lane_xyz"})
            continue
        dx = cand_xyz[:, None, 0] - lane_xyz[None, :, 0]
        dz = cand_xyz[:, None, 2] - lane_xyz[None, :, 2]
        dist = np.sqrt(dx * dx + dz * dz)
        min_dist = np.min(dist, axis=1)
        robust_dist = float(np.percentile(min_dist, 20))
        lane_quality = min(1.0, float(lane.get("depth_valid_ratio") or 0.0) * 2.0)
        raw = lane_quality / max(robust_dist, 1e-3)
        raw_scores.append(raw)
        details.append(
            {
                "lane_id": lane["lane_id"],
                "raw_score": raw,
                "robust_xz_distance": robust_dist,
                "lane_quality": lane_quality,
                "lane_depth_median": lane.get("depth_median"),
            }
        )
    total = float(sum(raw_scores))
    if total <= 0:
        return {"scores": details, "reason": "zero_total_score"}
    for item, raw in zip(details, raw_scores):
        item["score"] = float(raw / total)
    details.sort(key=lambda item: item.get("score", 0.0), reverse=True)
    probs = np.asarray([item.get("score", 0.0) for item in details], dtype=float)
    entropy = float(-(probs * np.log(np.clip(probs, 1e-9, 1.0))).sum() / max(math.log(max(len(probs), 2)), 1e-9))
    margin = float(probs[0] - probs[1]) if len(probs) > 1 else float(probs[0])
    return {
        "scores": details,
        "top_lane_id": details[0]["lane_id"] if details else None,
        "top_score": float(probs[0]) if probs.size else 0.0,
        "top2_margin": margin,
        "entropy": entropy,
        "assignment_mode": "single_lane_default",
    }


def score_road_marking_to_lane_bands(candidates: list[dict[str, Any]], lanes: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    if not candidates:
        return {"scores": [], "reason": "no_ground_candidates", "method": "road_marking_lane_band_bev"}
    candidate_items = [item for item in candidates if item.get("xyz")]
    if not candidate_items:
        return {"scores": [], "reason": "no_candidate_xyz", "method": "road_marking_lane_band_bev"}

    raw_scores = []
    details = []
    distance_scale = max(1e-3, float(args.lane_band_distance_scale_m))
    center_scale = max(1e-3, float(args.lane_band_center_scale_m))
    default_width = max(1e-3, float(args.lane_band_default_width_m))

    for lane in lanes:
        if not (lane.get("bev_fit") or {}).get("ok"):
            raw_scores.append(0.0)
            details.append({"lane_id": lane["lane_id"], "raw_score": 0.0, "reason": "no_lane_bev_fit"})
            continue

        band_distances: list[float] = []
        center_offsets: list[float] = []
        inside_count = 0
        preview = []
        for cand in candidate_items:
            xyz = np.asarray(cand["xyz"], dtype=float)
            cx, cz = float(xyz[0]), float(xyz[2])
            bands = lane_bands_at_z(lanes, cz, default_width)
            band = next((item for item in bands if item["lane_id"] == lane["lane_id"]), None)
            if band is None:
                continue
            left = float(band["left_x"])
            right = float(band["right_x"])
            center_x = float(band["center_x"])
            if left <= cx <= right:
                dist = 0.0
                inside = True
            else:
                dist = min(abs(cx - left), abs(cx - right))
                inside = False
            offset = abs(cx - center_x)
            band_distances.append(float(dist))
            center_offsets.append(float(offset))
            inside_count += int(inside)
            if len(preview) < 24:
                preview.append(
                    {
                        "lane_id": lane["lane_id"],
                        "x": float(cand["x"]),
                        "y": float(cand["y"]),
                        "depth": float(cand["depth"]),
                        "bev_xz": [cx, cz],
                        "band_left_x": left,
                        "band_right_x": right,
                        "band_center_x": center_x,
                        "inside_band": inside,
                        "band_distance": float(dist),
                        "center_offset": float(offset),
                    }
                )

        if not band_distances:
            raw_scores.append(0.0)
            details.append({"lane_id": lane["lane_id"], "raw_score": 0.0, "reason": "no_candidate_band_eval"})
            continue

        inside_fraction = inside_count / max(len(band_distances), 1)
        robust_band_distance = float(np.percentile(band_distances, 50))
        robust_center_offset = float(np.percentile(center_offsets, 50))
        band_score = math.exp(-robust_band_distance / distance_scale)
        center_score = math.exp(-robust_center_offset / center_scale)
        raw = (0.05 + inside_fraction) * band_score * (0.35 + 0.65 * center_score)
        raw_scores.append(raw)
        details.append(
            {
                "lane_id": lane["lane_id"],
                "raw_score": raw,
                "method": "road_marking_lane_band_bev",
                "inside_fraction": float(inside_fraction),
                "robust_band_distance": robust_band_distance,
                "robust_center_offset": robust_center_offset,
                "candidate_count": len(band_distances),
                "lane_bev_fit": lane.get("bev_fit"),
                "band_samples_preview": preview,
            }
        )

    total = float(sum(raw_scores))
    if total <= 0:
        return {"scores": details, "reason": "zero_total_score", "method": "road_marking_lane_band_bev"}
    for item, raw in zip(details, raw_scores):
        item["score"] = float(raw / total)
    details.sort(key=lambda item: item.get("score", 0.0), reverse=True)
    probs = np.asarray([item.get("score", 0.0) for item in details], dtype=float)
    entropy = float(-(probs * np.log(np.clip(probs, 1e-9, 1.0))).sum() / max(math.log(max(len(probs), 2)), 1e-9))
    margin = float(probs[0] - probs[1]) if len(probs) > 1 else float(probs[0])
    return {
        "scores": details,
        "top_lane_id": details[0]["lane_id"] if details else None,
        "top_score": float(probs[0]) if probs.size else 0.0,
        "top2_margin": margin,
        "entropy": entropy,
        "assignment_mode": "single_lane_default",
        "method": "road_marking_lane_band_bev",
    }


def score_sign_bev_anchor_to_lanes(anchor: dict[str, Any], lanes: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    if not anchor.get("ok") or not anchor.get("xyz"):
        return {"scores": [], "reason": anchor.get("reason", "invalid_sign_anchor")}
    anchor_xyz = np.asarray(anchor["xyz"], dtype=float)
    anchor_bev = np.asarray([anchor_xyz[0], anchor_xyz[2]], dtype=float)
    details = []
    raw_scores = []
    lateral_scale = max(1e-3, float(args.sign_lateral_scale_m))
    default_width = max(1e-3, float(args.lane_band_default_width_m))
    bands = lane_bands_at_z(lanes, float(anchor_bev[1]), default_width)
    band_by_lane = {str(item["lane_id"]): item for item in bands}

    for lane in lanes:
        band = band_by_lane.get(str(lane["lane_id"]))
        if band is None:
            raw_scores.append(0.0)
            details.append({"lane_id": lane["lane_id"], "raw_score": 0.0, "reason": "no_lane_bev_fit"})
            continue

        left = float(band["left_x"])
        right = float(band["right_x"])
        center_x = float(band["center_x"])
        anchor_x = float(anchor_bev[0])
        inside = left <= anchor_x <= right
        band_distance = 0.0 if inside else min(abs(anchor_x - left), abs(anchor_x - right))
        center_offset = abs(anchor_x - center_x)
        center_score = math.exp(-center_offset / lateral_scale)
        band_score = 1.0 if inside else math.exp(-band_distance / lateral_scale)
        lane_quality = min(1.0, float(lane.get("depth_valid_ratio") or 0.0) * 2.0)
        quality_factor = 0.75 + 0.25 * lane_quality
        raw = quality_factor * band_score * center_score
        raw_scores.append(raw)

        fit = lane.get("bev_fit") or {}
        z_min = fit.get("z_min")
        z_max = fit.get("z_max")
        z_extrapolation = 0.0
        if z_min is not None and float(anchor_bev[1]) < float(z_min):
            z_extrapolation = float(z_min) - float(anchor_bev[1])
        elif z_max is not None and float(anchor_bev[1]) > float(z_max):
            z_extrapolation = float(anchor_bev[1]) - float(z_max)
        details.append(
            {
                "lane_id": lane["lane_id"],
                "raw_score": raw,
                "method": "sign_bev_lateral_lane_band",
                "lane_quality": lane_quality,
                "quality_factor": quality_factor,
                "lane_depth_median": lane.get("depth_median"),
                "anchor_bev_xz": [float(anchor_bev[0]), float(anchor_bev[1])],
                "lane_center_x_at_anchor_z": center_x,
                "band_left_x": left,
                "band_right_x": right,
                "inside_band": bool(inside),
                "band_distance": float(band_distance),
                "center_offset": float(center_offset),
                "center_score": float(center_score),
                "band_score": float(band_score),
                "z_extrapolation_m": float(z_extrapolation),
                "lane_bev_fit": fit,
            }
        )

    total = float(sum(raw_scores))
    if total <= 0:
        return {"scores": details, "reason": "zero_total_score", "method": "sign_bev_lateral_lane_band"}
    for item, raw in zip(details, raw_scores):
        item["score"] = float(raw / total)
    details.sort(key=lambda item: item.get("score", 0.0), reverse=True)
    probs = np.asarray([item.get("score", 0.0) for item in details], dtype=float)
    entropy = float(-(probs * np.log(np.clip(probs, 1e-9, 1.0))).sum() / max(math.log(max(len(probs), 2)), 1e-9))
    margin = float(probs[0] - probs[1]) if len(probs) > 1 else float(probs[0])
    return {
        "scores": details,
        "top_lane_id": details[0]["lane_id"] if details else None,
        "top_score": float(probs[0]) if probs.size else 0.0,
        "top2_margin": margin,
        "entropy": entropy,
        "assignment_mode": "single_lane_default",
        "method": "sign_bev_lateral_lane_band",
    }


def sign_depth_confidence_ok(body_depth: dict[str, Any], args: argparse.Namespace) -> tuple[bool, str | None, dict[str, Any]]:
    median = float(body_depth.get("conf_median") or 0.0)
    p75 = float(body_depth.get("conf_p75") or 0.0)
    enabled = bool(getattr(args, "enable_sign_depth_confidence_filter", False))
    gate = {
        "enabled": enabled,
        "conf_median": median,
        "conf_p75": p75,
        "min_conf_median": float(args.sign_min_conf_median),
        "min_conf_p75": float(args.sign_min_conf_p75),
    }
    median_ok = median >= float(args.sign_min_conf_median)
    p75_ok = p75 >= float(args.sign_min_conf_p75)
    gate["median_ok"] = median_ok
    gate["p75_ok"] = p75_ok
    if not enabled:
        return True, None, gate
    if not median_ok or not p75_ok:
        return False, "sign_depth_confidence_low", gate
    return True, None, gate


def analyze_object(scene: DepthScene, obj: dict[str, Any], lane_profiles: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    kind = object_kind(obj.get("label_id"))
    body_depth = scene.robust_depth_in_bbox(obj["bbox"])
    anchor = object_depth_anchor(scene, obj, body_depth)
    filtered = False
    filter_reason = None
    if obj.get("label_id") in ALWAYS_FILTER_LABEL_IDS:
        filtered = True
        filter_reason = "mixed_lane_signal_candidate_filtered"
    elif kind == "sign_signal":
        if not body_depth.get("ok") or float(body_depth.get("valid_ratio") or 0.0) < args.min_object_depth_valid_ratio:
            filtered = True
            filter_reason = "sign_depth_unreliable"
        elif float(body_depth["depth_median"]) > args.sign_max_depth_m:
            filtered = True
            filter_reason = "sign_depth_gt_max"
        else:
            confidence_ok, confidence_reason, confidence_gate = sign_depth_confidence_ok(body_depth, args)
            if not confidence_ok:
                filtered = True
                filter_reason = confidence_reason
            obj["sign_confidence_gate"] = confidence_gate

    result = {
        **obj,
        "kind": kind,
        "body_depth": body_depth,
        "anchor": anchor,
        "filtered": filtered,
        "filter_reason": filter_reason,
        "sign_max_depth_m": args.sign_max_depth_m if kind == "sign_signal" else None,
    }
    if filtered:
        result["ground_candidate_count"] = 0
        result["assignment"] = {"scores": [], "reason": filter_reason}
        return result
    if kind == "sign_signal":
        result["ground_candidate_count"] = 0
        result["ground_depth_median"] = None
        result["association_input"] = "sign_bev_depth_anchor"
        result["assignment"] = score_sign_bev_anchor_to_lanes(anchor, lane_profiles, args)
        top_score = (result["assignment"].get("scores") or [{}])[0] if result["assignment"].get("scores") else {}
        result["sign_match_preview"] = top_score.get("match_samples_preview", [])
        return result
    candidates = object_ground_candidates(scene, obj, kind, args)
    result["ground_candidate_count"] = len(candidates)
    result["ground_depth_median"] = float(np.median([item["depth"] for item in candidates])) if candidates else None
    result["association_input"] = "road_marking_ground_candidates"
    result["ground_candidates_preview"] = candidates[:30]
    result["assignment"] = score_road_marking_to_lane_bands(candidates, lane_profiles, args)
    top_score = (result["assignment"].get("scores") or [{}])[0] if result["assignment"].get("scores") else {}
    result["road_marking_band_preview"] = top_score.get("band_samples_preview", [])
    return result


def color_for_index(idx: int) -> tuple[int, int, int]:
    return PALETTE_RGB[idx % len(PALETTE_RGB)]


def draw_label(draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str, fill: tuple[int, int, int]) -> None:
    font = ImageFont.load_default()
    x, y = xy
    try:
        bbox = draw.multiline_textbbox((x, y), text, font=font, spacing=2)
    except AttributeError:
        bbox = draw.textbbox((x, y), text.replace("\n", " "), font=font)
    pad = 3
    draw.rectangle((bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad), fill=(0, 0, 0))
    draw.multiline_text((x, y), text, fill=fill, font=font, spacing=2)


def object_label_prefix(obj: dict[str, Any]) -> str:
    label_name = obj.get("label_name") or f"label_{obj.get('label_id')}"
    return f"{obj['object_id']} {label_name}"


def draw_overlay(image: Image.Image, scene: DepthScene, lane_profiles: list[dict[str, Any]], objects: list[dict[str, Any]]) -> Image.Image:
    out = image.convert("RGBA")
    road = Image.new("RGBA", out.size, (0, 0, 0, 0))
    road_arr = np.zeros((scene.image_height, scene.image_width, 4), dtype=np.uint8)
    road_arr[scene.road_mask] = [40, 220, 80, 45]
    road = Image.fromarray(road_arr)
    out = Image.alpha_composite(out, road)
    draw = ImageDraw.Draw(out)
    for lane in lane_profiles:
        color = color_for_index(int(lane.get("index", 0)))
        pts = [tuple(point) for point in lane.get("points", [])]
        if len(pts) >= 2:
            draw.line(pts, fill=color + (255,), width=4)
        for sample in lane.get("samples", []):
            if sample.get("ok"):
                x = sample.get("road_x", sample.get("x"))
                y = sample.get("road_y", sample.get("y"))
                sample_fill = (255, 170, 0, 220) if sample.get("sample_source") != "centerline_exact" else (0, 255, 255, 210)
                draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=sample_fill)
        if pts:
            if not lane.get("valid_for_association"):
                label = f"lane={lane['lane_id']} invalid"
            elif lane.get("depth_median"):
                label = f"lane={lane['lane_id']} d={lane.get('depth_median'):.1f}"
            else:
                label = f"lane={lane['lane_id']}"
            draw_label(
                draw,
                (pts[0][0] + 4, pts[0][1] - 14),
                label,
                color,
            )
    for obj in objects:
        x1, y1, x2, y2 = obj["bbox"]
        label_prefix = object_label_prefix(obj)
        if obj.get("filtered"):
            color = (255, 70, 70)
            label = f"{label_prefix}\nfiltered {obj.get('filter_reason')}"
        else:
            color = (255, 220, 40)
            assignment = obj.get("assignment") or {}
            label = f"{label_prefix}\ntop={assignment.get('top_lane_id')} p={assignment.get('top_score', 0):.2f}"
        draw.rectangle((x1, y1, x2, y2), outline=color + (255,), width=4)
        depth = (obj.get("body_depth") or {}).get("depth_median")
        if depth is not None:
            label += f" z={depth:.1f}m"
        conf_median = (obj.get("body_depth") or {}).get("conf_median")
        conf_p75 = (obj.get("body_depth") or {}).get("conf_p75")
        if conf_median is not None and conf_p75 is not None:
            label += f"\nconf={float(conf_median):.2f}/{float(conf_p75):.2f}"
        draw_label(draw, (x1, max(0, y1 - 44)), label, color)
        anchor = obj.get("anchor") or {}
        if obj.get("kind") == "sign_signal" and anchor.get("ok") and anchor.get("image_xy"):
            ax, ay = anchor["image_xy"]
            draw.ellipse((ax - 4, ay - 4, ax + 4, ay + 4), outline=(255, 255, 255, 255), width=2)
            draw.line((ax - 6, ay, ax + 6, ay), fill=(255, 255, 255, 255), width=2)
            draw.line((ax, ay - 6, ax, ay + 6), fill=(255, 255, 255, 255), width=2)
            for sample in obj.get("sign_match_preview", [])[:12]:
                sx, sy = sample["x"], sample["y"]
                draw.ellipse((sx - 4, sy - 4, sx + 4, sy + 4), outline=(255, 60, 255, 230), width=2)
        else:
            for cand in obj.get("ground_candidates_preview", [])[:40]:
                x, y = cand["x"], cand["y"]
                draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(255, 150, 0, 180))
            for cand in obj.get("road_marking_band_preview", [])[:20]:
                x, y = cand["x"], cand["y"]
                draw.ellipse((x - 4, y - 4, x + 4, y + 4), outline=(255, 255, 255, 230), width=2)
    return out.convert("RGB")


def draw_depth_overlay(image: Image.Image, scene: DepthScene, lane_profiles: list[dict[str, Any]], objects: list[dict[str, Any]]) -> Image.Image:
    depth = scene.depth.astype(float)
    valid = np.isfinite(depth) & (depth > 0)
    if np.any(valid):
        lo, hi = np.percentile(depth[valid], [2, 98])
    else:
        lo, hi = 0.0, 1.0
    norm = np.clip((depth - lo) / max(hi - lo, 1e-6), 0, 1)
    depth_u8 = (255 - norm * 255).astype(np.uint8)
    colored = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    colored = cv2.resize(colored, image.size, interpolation=cv2.INTER_CUBIC)
    blend = (np.asarray(image.convert("RGB")).astype(float) * 0.45 + colored.astype(float) * 0.55).astype(np.uint8)
    return draw_overlay(Image.fromarray(blend), scene, lane_profiles, objects)


def confidence_stats(conf: np.ndarray) -> dict[str, float | int | None]:
    values = conf.astype(float).reshape(-1)
    valid = values[np.isfinite(values)]
    if valid.size == 0:
        return {"count": 0}
    return {
        "count": int(valid.size),
        "min": float(np.min(valid)),
        "p01": float(np.percentile(valid, 1)),
        "p10": float(np.percentile(valid, 10)),
        "median": float(np.median(valid)),
        "p90": float(np.percentile(valid, 90)),
        "p99": float(np.percentile(valid, 99)),
        "max": float(np.max(valid)),
    }


def draw_confidence_overlay(
    image: Image.Image,
    scene: DepthScene,
    lane_profiles: list[dict[str, Any]],
    objects: list[dict[str, Any]],
    args: argparse.Namespace,
) -> Image.Image:
    conf = scene.conf.astype(float)
    valid = np.isfinite(conf)
    if np.any(valid):
        lo, hi = np.percentile(conf[valid], [1, 99])
        if hi - lo <= 1e-6:
            lo, hi = float(np.min(conf[valid])), float(np.max(conf[valid]))
        if hi - lo <= 1e-6:
            hi = lo + 1.0
    else:
        lo, hi = 0.0, 1.0
    norm = np.clip((conf - lo) / max(hi - lo, 1e-6), 0, 1)
    conf_u8 = (norm * 255).astype(np.uint8)
    colormap = getattr(cv2, "COLORMAP_VIRIDIS", cv2.COLORMAP_TURBO)
    colored = cv2.applyColorMap(conf_u8, colormap)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    colored = cv2.resize(colored, image.size, interpolation=cv2.INTER_CUBIC)
    blend = (np.asarray(image.convert("RGB")).astype(float) * 0.40 + colored.astype(float) * 0.60).astype(np.uint8)
    out = draw_overlay(Image.fromarray(blend), scene, lane_profiles, objects).convert("RGB")
    stats = confidence_stats(scene.conf)
    legend = (
        "DA3 confidence heatmap\n"
        f"p01={stats.get('p01', 0):.2f} median={stats.get('median', 0):.2f} p99={stats.get('p99', 0):.2f}\n"
        f"pixel min_conf={scene.min_conf:.2f}; "
        + (
            f"sign conf filter on: median>={args.sign_min_conf_median:.2f}, p75>={args.sign_min_conf_p75:.2f}"
            if getattr(args, "enable_sign_depth_confidence_filter", False)
            else "sign conf filter off"
        )
    )
    draw_label(ImageDraw.Draw(out), (12, 12), legend, (255, 255, 255))
    return out


def draw_bev_debug(lane_profiles: list[dict[str, Any]], objects: list[dict[str, Any]], args: argparse.Namespace) -> Image.Image:
    width, height = 1100, 760
    plot_w, plot_h = 720, 660
    left, top = 55, 45
    right_panel_x = left + plot_w + 30
    canvas = Image.new("RGB", (width, height), (248, 248, 248))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    points: list[tuple[float, float]] = []
    for lane in lane_profiles:
        for sample in lane.get("samples", []):
            if sample.get("ok") and sample.get("xyz"):
                xyz = sample["xyz"]
                points.append((float(xyz[0]), float(xyz[2])))
    for obj in objects:
        anchor = obj.get("anchor") or {}
        if anchor.get("ok") and anchor.get("bev_xz"):
            points.append((float(anchor["bev_xz"][0]), float(anchor["bev_xz"][1])))
        for cand in obj.get("ground_candidates_preview", []):
            if cand.get("xyz"):
                xyz = cand["xyz"]
                points.append((float(xyz[0]), float(xyz[2])))
        for cand in obj.get("road_marking_band_preview", []):
            if cand.get("bev_xz"):
                points.append((float(cand["bev_xz"][0]), float(cand["bev_xz"][1])))
    if not points:
        points = [(-5.0, 0.0), (5.0, 40.0)]
    xs = [p[0] for p in points]
    zs = [p[1] for p in points]
    xmin, xmax = min(xs), max(xs)
    zmin, zmax = min(zs), max(zs)
    xpad = max(2.0, (xmax - xmin) * 0.18)
    zpad = max(3.0, (zmax - zmin) * 0.18)
    xmin -= xpad
    xmax += xpad
    zmin = max(0.0, zmin - zpad)
    zmax += zpad
    if xmax <= xmin:
        xmax = xmin + 1.0
    if zmax <= zmin:
        zmax = zmin + 1.0

    def xy(x: float, z: float) -> tuple[float, float]:
        px = left + (float(x) - xmin) / (xmax - xmin) * plot_w
        py = top + (zmax - float(z)) / (zmax - zmin) * plot_h
        return px, py

    draw.rectangle((left, top, left + plot_w, top + plot_h), outline=(180, 180, 180), width=1)
    draw.text((left, 18), "BEV debug: X lateral, Z forward", fill=(0, 0, 0), font=font)
    draw.text((left + 5, top + plot_h + 8), f"X {xmin:.1f}..{xmax:.1f}m", fill=(80, 80, 80), font=font)
    draw.text((left + plot_w - 100, top + 5), f"Z {zmax:.1f}m", fill=(80, 80, 80), font=font)
    draw.text((left + plot_w - 100, top + plot_h - 18), f"Z {zmin:.1f}m", fill=(80, 80, 80), font=font)

    z_grid = np.linspace(zmin, zmax, 48)
    lane_band_lines: dict[str, dict[str, list[tuple[float, float]]]] = {}
    for z in z_grid:
        bands = lane_bands_at_z(lane_profiles, float(z), args.lane_band_default_width_m)
        for band in bands:
            lid = str(band["lane_id"])
            lane_band_lines.setdefault(lid, {"left": [], "center": [], "right": []})
            lane_band_lines[lid]["left"].append(xy(band["left_x"], z))
            lane_band_lines[lid]["center"].append(xy(band["center_x"], z))
            lane_band_lines[lid]["right"].append(xy(band["right_x"], z))
    for lane in lane_profiles:
        lid = str(lane["lane_id"])
        color = color_for_index(int(lane.get("index", 0)))
        rgb = color
        lines = lane_band_lines.get(lid)
        if lines:
            if len(lines["center"]) >= 2:
                draw.line(lines["center"], fill=rgb, width=3)
            if len(lines["left"]) >= 2:
                draw.line(lines["left"], fill=(160, 160, 160), width=1)
            if len(lines["right"]) >= 2:
                draw.line(lines["right"], fill=(160, 160, 160), width=1)
            label_pt = lines["center"][min(len(lines["center"]) - 1, max(0, len(lines["center"]) // 3))]
            draw.text((label_pt[0] + 4, label_pt[1]), f"lane {lid}", fill=rgb, font=font)
        for sample in lane.get("samples", []):
            if sample.get("ok") and sample.get("xyz"):
                px, py = xy(float(sample["xyz"][0]), float(sample["xyz"][2]))
                draw.ellipse((px - 1.5, py - 1.5, px + 1.5, py + 1.5), fill=rgb)

    table_y = 45
    draw.text((right_panel_x, 18), "Object confidence / BEV association", fill=(0, 0, 0), font=font)
    for obj in objects:
        body = obj.get("body_depth") or {}
        assignment = obj.get("assignment") or {}
        prefix = object_label_prefix(obj)
        reason = obj.get("filter_reason") or assignment.get("method") or assignment.get("reason") or ""
        top_lane = assignment.get("top_lane_id")
        top_score = assignment.get("top_score")
        conf_med = body.get("conf_median")
        conf_p75 = body.get("conf_p75")
        depth = body.get("depth_median")
        line1 = f"{prefix}"
        line2 = f"z={depth:.1f} conf={conf_med:.2f}/{conf_p75:.2f}" if depth is not None and conf_med is not None and conf_p75 is not None else "z/conf=n/a"
        line3 = f"top={top_lane} p={float(top_score):.2f}" if top_lane is not None else f"filtered={obj.get('filter_reason')}"
        draw.text((right_panel_x, table_y), line1[:48], fill=(0, 0, 0), font=font)
        draw.text((right_panel_x, table_y + 14), line2[:48], fill=(60, 60, 60), font=font)
        draw.text((right_panel_x, table_y + 28), f"{line3} {reason}"[:52], fill=(120, 0, 0) if obj.get("filtered") else (0, 80, 0), font=font)

        for cand in obj.get("ground_candidates_preview", [])[:80]:
            if cand.get("xyz"):
                px, py = xy(float(cand["xyz"][0]), float(cand["xyz"][2]))
                draw.ellipse((px - 2, py - 2, px + 2, py + 2), fill=(255, 150, 0))
        for cand in obj.get("road_marking_band_preview", [])[:24]:
            if cand.get("bev_xz"):
                bx, bz = float(cand["bev_xz"][0]), float(cand["bev_xz"][1])
                px, py = xy(bx, bz)
                draw.ellipse((px - 4, py - 4, px + 4, py + 4), outline=(0, 0, 0), width=1)
                center_x = cand.get("band_center_x")
                left_x = cand.get("band_left_x")
                right_x = cand.get("band_right_x")
                if center_x is not None:
                    cx_px, cy_px = xy(float(center_x), bz)
                    draw.line((px, py, cx_px, cy_px), fill=(45, 105, 220), width=1)
                    draw.ellipse((cx_px - 2, cy_px - 2, cx_px + 2, cy_px + 2), fill=(45, 105, 220))
                if left_x is not None and right_x is not None and not cand.get("inside_band"):
                    nearest_x = float(left_x) if abs(bx - float(left_x)) < abs(bx - float(right_x)) else float(right_x)
                    nx_px, ny_px = xy(nearest_x, bz)
                    draw.line((px, py, nx_px, ny_px), fill=(220, 60, 60), width=2)
        anchor = obj.get("anchor") or {}
        if anchor.get("ok") and anchor.get("bev_xz"):
            px, py = xy(float(anchor["bev_xz"][0]), float(anchor["bev_xz"][1]))
            obj_color = (255, 60, 60) if obj.get("filtered") else (255, 220, 40)
            draw.ellipse((px - 5, py - 5, px + 5, py + 5), outline=obj_color, width=2)
            draw.line((px - 7, py, px + 7, py), fill=obj_color, width=2)
            draw.line((px, py - 7, px, py + 7), fill=obj_color, width=2)
        score_lines = assignment.get("scores") or []
        for idx, score in enumerate(score_lines[:3]):
            if score.get("method") == "road_marking_lane_band_bev":
                msg = f"  lane {score.get('lane_id')}: p={float(score.get('score') or 0):.2f} in={float(score.get('inside_fraction') or 0):.2f} d={float(score.get('robust_band_distance') or 0):.2f} c={float(score.get('robust_center_offset') or 0):.2f}"
            elif score.get("method") == "sign_bev_lateral_lane_band":
                msg = f"  lane {score.get('lane_id')}: p={float(score.get('score') or 0):.2f} in={int(bool(score.get('inside_band')))} d={float(score.get('band_distance') or 0):.2f} c={float(score.get('center_offset') or 0):.2f} ex={float(score.get('z_extrapolation_m') or 0):.1f}"
            elif score.get("method") == "sign_bev_anchor_lane_profile":
                msg = f"  lane {score.get('lane_id')}: p={float(score.get('score') or 0):.2f} xz=({float(score.get('robust_bev_lateral') or 0):.1f},{float(score.get('robust_bev_longitudinal') or 0):.1f})"
            else:
                msg = f"  lane {score.get('lane_id')}: p={float(score.get('score') or 0):.2f}"
            draw.text((right_panel_x, table_y + 44 + idx * 13), msg[:52], fill=(50, 50, 50), font=font)
        table_y += 92
    return canvas


def process_frame(paths: Paths, frame: str, args: argparse.Namespace) -> dict[str, Any]:
    center_path = paths.center_dir / f"{frame}.json"
    object_path = paths.objects_dir / f"{frame}.json"
    object_data = load_json(object_path, default={})
    image_path = image_path_for_frame(paths, frame, object_data if isinstance(object_data, dict) else {})
    if image_path is None:
        raise FileNotFoundError(f"missing source image for frame {frame}")
    image = Image.open(image_path).convert("RGB")
    lanes = load_lanes(center_path)
    objects = load_objects(object_path)
    scene = load_depth_scene(paths, frame, image.size, args.min_conf)
    lane_profiles = [lane_depth_profile(scene, lane, args) for lane in lanes]
    object_results = [analyze_object(scene, obj, lane_profiles, args) for obj in objects]

    assoc_dir = paths.out_dir / "association_overlay"
    depth_dir = paths.out_dir / "depth_overlay"
    confidence_dir = paths.out_dir / "confidence_overlay"
    bev_dir = paths.out_dir / "bev_debug"
    assoc_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)
    confidence_dir.mkdir(parents=True, exist_ok=True)
    bev_dir.mkdir(parents=True, exist_ok=True)
    assoc_path = assoc_dir / f"{frame}.jpg"
    depth_path = depth_dir / f"{frame}.jpg"
    confidence_path = confidence_dir / f"{frame}.jpg"
    bev_path = bev_dir / f"{frame}.jpg"
    if args.overwrite or not assoc_path.exists():
        draw_overlay(image, scene, lane_profiles, object_results).save(assoc_path, quality=92)
    if args.overwrite or not depth_path.exists():
        draw_depth_overlay(image, scene, lane_profiles, object_results).save(depth_path, quality=92)
    if args.overwrite or not confidence_path.exists():
        draw_confidence_overlay(image, scene, lane_profiles, object_results, args).save(confidence_path, quality=92)
    if args.overwrite or not bev_path.exists():
        draw_bev_debug(lane_profiles, object_results, args).save(bev_path, quality=92)

    result = {
        "frame": frame,
        "ok": True,
        "image_path": str(image_path),
        "depth_npz": str(depth_npz_path(paths, frame)),
        "sam_mask": str(sam_mask_path(paths, frame)),
        "confidence_stats": confidence_stats(scene.conf),
        "lanes": [
            {key: value for key, value in lane.items() if key not in {"samples", "xyz"}}
            for lane in lane_profiles
        ],
        "objects": [
            {key: value for key, value in obj.items() if key not in {"ground_candidates_preview"}}
            for obj in object_results
        ],
        "outputs": {
            "association_overlay": str(assoc_path),
            "depth_overlay": str(depth_path),
            "confidence_overlay": str(confidence_path),
            "bev_debug": str(bev_path),
        },
        "filtered_object_count": sum(1 for obj in object_results if obj.get("filtered")),
        "active_object_count": sum(1 for obj in object_results if not obj.get("filtered")),
    }
    frame_json = paths.artifact_dir / "frames" / f"{frame}.json"
    dump_json(frame_json, result)
    result["artifact_json"] = str(frame_json)
    return result


def rel_link(root: Path, path: str | None) -> str:
    if not path:
        return ""
    try:
        return html.escape(str(Path(path).relative_to(root)))
    except Exception:
        return html.escape(str(path))


def write_index(paths: Paths, results: list[dict[str, Any]]) -> None:
    rows = []
    for item in sorted(results, key=lambda x: x.get("frame", "")):
        outputs = item.get("outputs", {})
        frame = html.escape(str(item.get("frame", "")))
        status = "ok" if item.get("ok") else html.escape(str(item.get("error", "failed")))
        object_bits = []
        for obj in item.get("objects", []):
            label_prefix = object_label_prefix(obj)
            if obj.get("filtered"):
                object_bits.append(f"{label_prefix}:filtered:{obj.get('filter_reason')}")
            else:
                assignment = obj.get("assignment") or {}
                object_bits.append(f"{label_prefix}:top={assignment.get('top_lane_id')} p={assignment.get('top_score', 0):.2f}")
        rows.append(
            "<tr>"
            f"<td>{frame}</td><td>{status}</td>"
            f"<td>{item.get('active_object_count', 0)}</td><td>{item.get('filtered_object_count', 0)}</td>"
            f"<td>{html.escape('; '.join(object_bits))}</td>"
            f"<td><a href=\"{rel_link(paths.out_dir, outputs.get('association_overlay'))}\">association</a></td>"
            f"<td><a href=\"{rel_link(paths.out_dir, outputs.get('depth_overlay'))}\">depth</a></td>"
            f"<td><a href=\"{rel_link(paths.out_dir, outputs.get('confidence_overlay'))}\">confidence</a></td>"
            f"<td><a href=\"{rel_link(paths.out_dir, outputs.get('bev_debug'))}\">bev</a></td>"
            f"<td><a href=\"{html.escape(str(Path(item.get('artifact_json', '')).relative_to(paths.out_dir) if item.get('artifact_json') and Path(item['artifact_json']).is_relative_to(paths.out_dir) else item.get('artifact_json', '')))}\">json</a></td>"
            "</tr>"
        )
    page = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>3D Lane Association Probe</title>
  <style>
    body {{ font-family: sans-serif; margin: 24px; }}
    table {{ border-collapse: collapse; font-size: 13px; }}
    th, td {{ border: 1px solid #ccc; padding: 4px 6px; vertical-align: top; }}
    th {{ background: #f2f2f2; position: sticky; top: 0; }}
  </style>
</head>
<body>
  <h1>3D Lane Association Probe</h1>
  <p>Frames: {len(results)}</p>
  <table>
    <thead>
      <tr><th>frame</th><th>status</th><th>active objects</th><th>filtered objects</th><th>object summary</th><th>association</th><th>depth</th><th>confidence</th><th>bev</th><th>json</th></tr>
    </thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
</body>
</html>
"""
    (paths.out_dir / "index.html").write_text(page, encoding="utf-8")


def write_readme(paths: Paths, args: argparse.Namespace, frames: list[str], results: list[dict[str, Any]]) -> None:
    lines = [
        "# 3D Lane Association Probe",
        "",
        "Diagnostic output from `probe_3d_lane_association.py`.",
        "",
        "This does not alter VLM or decision-head outputs.",
        "",
        "## Inputs",
        "",
        f"- Base directory: `{paths.base_dir}`",
        f"- Center line 2D: `{paths.center_dir}`",
        f"- Objects: `{paths.objects_dir}`",
        f"- Depth: `{paths.depth_dir}`",
        f"- SAM3 road masks: `{paths.sam3_dir}`",
        f"- Instance YAML: `{resolve_instances_yaml(args, paths)}`",
        "",
        "## Policy",
        "",
        f"- Sign/signal labels: `{sorted(SIGN_SIGNAL_LABEL_IDS)}`",
        f"- Always-filter labels: `{sorted(ALWAYS_FILTER_LABEL_IDS)}`",
        f"- Sign/signal object body depth filter: `depth_median <= {args.sign_max_depth_m}m`",
        (
            f"- Sign/signal depth confidence filter: enabled; gate is `conf_median >= {args.sign_min_conf_median}` and `conf_p75 >= {args.sign_min_conf_p75}`."
            if getattr(args, "enable_sign_depth_confidence_filter", False)
            else f"- Sign/signal depth confidence filter: disabled; confidence stats are still recorded and visualized. Stored thresholds for reference are `conf_median >= {args.sign_min_conf_median}` and `conf_p75 >= {args.sign_min_conf_p75}`."
        ),
        "- Lane depth samples are taken only on the exact 2D centerline if that pixel is inside the SAM3 road mask.",
        "- If strict centerline samples are insufficient, the lane is sampled along its forward extension first, then along the ego-side extension; lateral nearest-road rescue is disabled.",
        "- Sign/signal association uses lateral BEV lane-band assignment at the object's DA3 anchor depth; lane profile extrapolation is allowed and sign ground scatter search is disabled.",
        f"- Road-surface markings use BEV lane-band assignment. Lane bands are inferred from adjacent lane centers at each depth slice; default outer width is `{args.lane_band_default_width_m}m`.",
        "- Filtered objects are recorded for debugging but are not assigned to lanes.",
        "- Remaining objects receive frame-level soft scores across all candidate lane bands.",
        "- The default assignment mode is single-lane top-1 while preserving the full soft distribution.",
        "",
        "## Outputs",
        "",
        f"- Machine-readable summary: `{paths.artifact_dir / 'association_results.json'}`",
        "- `association_overlay/`: RGB + road mask + lane depth samples + object assignment.",
        "- `depth_overlay/`: DA3 depth visualization with the same overlays.",
        "- `confidence_overlay/`: DA3 confidence heatmap with the same overlays and sign gate thresholds.",
        "- `bev_debug/`: BEV lane bands, object confidence, and association-distance debug visualization.",
        "- `frames/*.json`: per-frame details.",
        "",
        "## Counts",
        "",
        f"- Frames requested: {len(frames)}",
        f"- Frames ok: {sum(1 for item in results if item.get('ok'))}",
        f"- Frames failed: {sum(1 for item in results if not item.get('ok'))}",
        f"- Active objects: {sum(item.get('active_object_count', 0) for item in results)}",
        f"- Filtered objects: {sum(item.get('filtered_object_count', 0) for item in results)}",
    ]
    (paths.out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    base_dir = resolve_base_dir(args)
    paths = build_paths(base_dir, args)
    paths.out_dir.mkdir(parents=True, exist_ok=True)
    paths.artifact_dir.mkdir(parents=True, exist_ok=True)
    frames = discover_frames(paths, args.limit)
    instances_yaml = resolve_instances_yaml(args, paths)
    frames, requested_instances, missing_instances = apply_instance_selection(frames, instances_yaml, args.frames)
    print(f"[info] base_dir={paths.base_dir}")
    print(f"[info] output_dir={paths.out_dir}")
    print(f"[info] artifact_dir={paths.artifact_dir}")
    print(f"[info] instances_yaml={instances_yaml} exists={instances_yaml.exists()}")
    if requested_instances is not None:
        print(f"[info] requested_instances={len(requested_instances)} selected={len(frames)} missing={len(missing_instances)}")
    print(f"[info] frames={len(frames)} workers={args.workers} sign_max_depth_m={args.sign_max_depth_m}")

    results: list[dict[str, Any]] = []
    worker_count = max(1, int(args.workers))
    if worker_count == 1:
        for idx, frame in enumerate(frames, 1):
            try:
                results.append(process_frame(paths, frame, args))
            except Exception as exc:
                print(f"[error] frame {frame}: {exc}", file=sys.stderr)
                results.append({"frame": frame, "ok": False, "error": str(exc), "outputs": {}})
            if idx % 10 == 0 or idx == len(frames):
                print(f"[info] processed {idx}/{len(frames)}")
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {executor.submit(process_frame, paths, frame, args): frame for frame in frames}
            for idx, future in enumerate(as_completed(futures), 1):
                frame = futures[future]
                try:
                    results.append(future.result())
                except Exception as exc:
                    print(f"[error] frame {frame}: {exc}", file=sys.stderr)
                    results.append({"frame": frame, "ok": False, "error": str(exc), "outputs": {}})
                if idx % 10 == 0 or idx == len(frames):
                    print(f"[info] processed {idx}/{len(frames)}")
    results.sort(key=lambda item: item.get("frame", ""))
    payload = {
        "schema_version": "camera_3d_lane_association_probe/v1",
        "base_dir": str(paths.base_dir),
        "output_dir": str(paths.out_dir),
        "artifact_dir": str(paths.artifact_dir),
        "round_name": args.round_name,
        "config": {
            "sign_max_depth_m": args.sign_max_depth_m,
            "sign_depth_confidence_filter_enabled": bool(args.enable_sign_depth_confidence_filter),
            "sign_min_conf_median": args.sign_min_conf_median,
            "sign_min_conf_p75": args.sign_min_conf_p75,
            "always_filter_label_ids": sorted(ALWAYS_FILTER_LABEL_IDS),
            "min_conf": args.min_conf,
            "min_object_depth_valid_ratio": args.min_object_depth_valid_ratio,
            "lane_samples": args.lane_samples,
            "lane_sampling_policy": "centerline_exact_then_forward_extension_then_ego_extension",
            "lane_min_depth_samples": args.lane_min_depth_samples,
            "lane_depth_radius": args.lane_depth_radius,
            "lane_extension_samples": args.lane_extension_samples,
            "lane_extension_max_px": args.lane_extension_max_px,
            "lane_search_radii": args.lane_search_radii,
            "sign_association_method": "sign_bev_lateral_lane_band",
            "road_marking_association_method": "road_marking_lane_band_bev",
            "lane_band_default_width_m": args.lane_band_default_width_m,
            "lane_band_distance_scale_m": args.lane_band_distance_scale_m,
            "lane_band_center_scale_m": args.lane_band_center_scale_m,
            "sign_lane_z_window_m": args.sign_lane_z_window_m,
            "sign_lane_nearest_k": args.sign_lane_nearest_k,
            "sign_lateral_scale_m": args.sign_lateral_scale_m,
            "sign_longitudinal_scale_m": args.sign_longitudinal_scale_m,
            "sign_ground_search_px": args.sign_ground_search_px,
            "max_ground_candidates": args.max_ground_candidates,
        },
        "requested_instances": requested_instances,
        "missing_instances": missing_instances,
        "frame_count": len(frames),
        "ok_count": sum(1 for item in results if item.get("ok")),
        "failed_count": sum(1 for item in results if not item.get("ok")),
        "active_object_count": sum(item.get("active_object_count", 0) for item in results),
        "filtered_object_count": sum(item.get("filtered_object_count", 0) for item in results),
        "results": results,
    }
    dump_json(paths.artifact_dir / "association_results.json", payload)
    dump_json(paths.out_dir / "manifest.json", payload)
    write_index(paths, results)
    write_readme(paths, args, frames, results)
    print(f"[done] ok={payload['ok_count']} failed={payload['failed_count']} active_objects={payload['active_object_count']} filtered_objects={payload['filtered_object_count']}")
    print(f"[done] artifact={paths.artifact_dir / 'association_results.json'}")
    print(f"[done] index={paths.out_dir / 'index.html'}")
    return 0 if payload["failed_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
