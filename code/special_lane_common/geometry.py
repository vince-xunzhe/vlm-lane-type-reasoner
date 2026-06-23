"""Geometry helpers for lane centerlines, detected boundaries, and boxes."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


Point = list[float]


@dataclass(frozen=True)
class BoundaryMatch:
    left: dict | None
    right: dict | None
    left_score: float | None
    right_score: float | None


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def norm1000_point(point: list[float] | tuple[float, float], width: int, height: int) -> list[int]:
    return [
        int(round(clamp(float(point[0]) / max(width, 1), 0.0, 1.0) * 1000)),
        int(round(clamp(float(point[1]) / max(height, 1), 0.0, 1.0) * 1000)),
    ]


def norm1000_bbox(bbox: list[float], width: int, height: int) -> list[int]:
    return [
        int(round(clamp(float(bbox[0]) / max(width, 1), 0.0, 1.0) * 1000)),
        int(round(clamp(float(bbox[1]) / max(height, 1), 0.0, 1.0) * 1000)),
        int(round(clamp(float(bbox[2]) / max(width, 1), 0.0, 1.0) * 1000)),
        int(round(clamp(float(bbox[3]) / max(height, 1), 0.0, 1.0) * 1000)),
    ]


def bbox_from_points(points: list[list[float]]) -> list[float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return [min(xs), min(ys), max(xs), max(ys)]


def polyline_length(polyline: list[list[float]]) -> float:
    return sum(math.dist(a, b) for a, b in zip(polyline[:-1], polyline[1:]))


def sample_polyline(polyline: list[list[float]], n: int) -> list[list[float]]:
    if not polyline:
        return []
    if len(polyline) == 1 or polyline_length(polyline) == 0:
        return [[float(polyline[0][0]), float(polyline[0][1])] for _ in range(max(n, 1))]

    segments = []
    cumulative = [0.0]
    total = 0.0
    for start, end in zip(polyline[:-1], polyline[1:]):
        seg_len = math.dist(start, end)
        segments.append((start, end, seg_len))
        total += seg_len
        cumulative.append(total)

    out = []
    seg_idx = 0
    for idx in range(max(n, 1)):
        target = total * idx / max(n - 1, 1)
        while seg_idx < len(segments) - 1 and cumulative[seg_idx + 1] < target:
            seg_idx += 1
        start, end, seg_len = segments[seg_idx]
        if seg_len == 0:
            out.append([float(start[0]), float(start[1])])
            continue
        t = clamp((target - cumulative[seg_idx]) / seg_len, 0.0, 1.0)
        out.append([
            round(float(start[0] + t * (end[0] - start[0])), 3),
            round(float(start[1] + t * (end[1] - start[1])), 3),
        ])
    return out


def norm1000_polyline(polyline: list[list[float]], width: int, height: int, n: int = 8) -> list[list[int]]:
    return [norm1000_point(point, width, height) for point in sample_polyline(polyline, n)]


def x_at_y(polyline: list[list[float]], y: float, *, fallback_to_nearest: bool = True) -> float | None:
    candidates = []
    for start, end in zip(polyline[:-1], polyline[1:]):
        x1, y1 = start
        x2, y2 = end
        if min(y1, y2) <= y <= max(y1, y2) and y1 != y2:
            t = (y - y1) / (y2 - y1)
            candidates.append(x1 + t * (x2 - x1))
    if candidates:
        return sum(candidates) / len(candidates)
    if fallback_to_nearest and polyline:
        nearest = min(polyline, key=lambda p: abs(p[1] - y))
        return float(nearest[0])
    return None


def object_anchor_point(bbox: list[float], label_id: int | None = None) -> list[float]:
    """Return the association anchor for an object bbox.

    Road text/symbols are associated by their bbox center.  Upright signs and
    signal heads are associated by the bottom-center because the bottom edge is
    closer to the lane they govern in image space.
    """
    x1, y1, x2, y2 = [float(value) for value in bbox]
    x = (x1 + x2) / 2
    y = (y1 + y2) / 2
    if label_id in {2, 6, 9}:
        y = y2
    return [x, y]


def object_lane_geometry(
    bbox: list[float],
    centerline: list[list[float]],
    left_boundary: list[list[float]] | None,
    right_boundary: list[list[float]] | None,
    *,
    label_id: int | None = None,
    margin_ratio: float = 0.16,
) -> dict:
    """Describe a bbox anchor relative to the target lane band.

    The output is intentionally simple so it can be embedded in VLM prompts as
    a geometric hint rather than as a hard label.
    """
    anchor = object_anchor_point(bbox, label_id=label_id)
    x, y = anchor
    center_x = x_at_y(centerline, y)
    left_x = x_at_y(left_boundary or [], y)
    right_x = x_at_y(right_boundary or [], y)
    distance_to_center = abs(x - center_x) if center_x is not None else None

    relation_hint = "unknown"
    lane_relative_x = None
    lane_width = None
    inside = False
    if left_x is not None and right_x is not None:
        low, high = sorted([left_x, right_x])
        lane_width = max(high - low, 1.0)
        margin = max(lane_width * margin_ratio, 18.0)
        lane_relative_x = (x - low) / lane_width
        if low - margin <= x <= high + margin:
            inside = True
            relation_hint = "inside_target_lane_band"
        elif x < low:
            relation_hint = "left_adjacent_or_outside"
        else:
            relation_hint = "right_adjacent_or_outside"
    elif center_x is not None:
        if distance_to_center is not None and distance_to_center <= 80:
            inside = True
            relation_hint = "near_target_centerline"
        else:
            relation_hint = "outside_target_centerline"

    return {
        "anchor_point": anchor,
        "center_x_at_anchor_y": center_x,
        "left_x_at_anchor_y": left_x,
        "right_x_at_anchor_y": right_x,
        "lane_width_at_anchor_y": lane_width,
        "lane_relative_x": lane_relative_x,
        "distance_to_centerline": distance_to_center,
        "inside_target_lane_band": inside,
        "relation_hint": relation_hint,
    }


def y_samples_for_centerline(centerline: list[list[float]], n: int = 5) -> list[float]:
    if not centerline:
        return []
    ys = [p[1] for p in sample_polyline(centerline, n)]
    return sorted(set(round(y, 3) for y in ys))


def _side_gap(
    centerline: list[list[float]],
    boundary: list[list[float]],
    ys: list[float],
    *,
    strict_y_overlap: bool = False,
    min_overlap_samples: int = 2,
) -> tuple[str, float, int] | None:
    gaps = []
    overlap_samples = 0
    for y in ys:
        cx = x_at_y(centerline, y)
        bx_strict = x_at_y(boundary, y, fallback_to_nearest=False)
        bx = bx_strict if bx_strict is not None else x_at_y(boundary, y, fallback_to_nearest=not strict_y_overlap)
        if cx is None or bx is None:
            continue
        if bx_strict is not None:
            overlap_samples += 1
        gaps.append(bx - cx)
    if strict_y_overlap and overlap_samples < min_overlap_samples:
        return None
    if not gaps:
        return None
    mean_gap = sum(gaps) / len(gaps)
    side = "right" if mean_gap > 0 else "left"
    return side, abs(mean_gap), overlap_samples


def match_left_right_boundaries(
    centerline: list[list[float]],
    boundaries: list[dict],
    *,
    mode: str = "nearest",
) -> BoundaryMatch:
    if mode not in {"nearest", "strict_y_overlap", "overlap_rescue"}:
        raise ValueError(f"Unsupported boundary match mode: {mode}")
    ys = y_samples_for_centerline(centerline, n=7)
    left_candidates = []
    right_candidates = []
    for boundary in boundaries:
        result = _side_gap(
            centerline,
            boundary["polyline"],
            ys,
            strict_y_overlap=mode == "strict_y_overlap",
        )
        if result is None:
            continue
        side, gap, overlap_samples = result
        item = (gap, boundary, overlap_samples)
        if side == "left":
            left_candidates.append(item)
        else:
            right_candidates.append(item)

    left_candidates.sort(key=lambda item: item[0])
    right_candidates.sort(key=lambda item: item[0])

    def choose(candidates: list[tuple[float, dict, int]]) -> tuple[float | None, dict | None]:
        if not candidates:
            return None, None
        best = candidates[0]
        if mode == "overlap_rescue" and best[2] == 0:
            for candidate in candidates[1:]:
                gap, _, overlap_samples = candidate
                if overlap_samples >= 2 and gap <= best[0] * 1.25 + 30.0:
                    best = candidate
                    break
        return best[0], best[1]

    left = choose(left_candidates)
    right = choose(right_candidates)
    return BoundaryMatch(
        left=left[1],
        right=right[1],
        left_score=round(left[0], 3) if left[0] is not None else None,
        right_score=round(right[0], 3) if right[0] is not None else None,
    )


def random_polyline_point(polyline: list[list[float]], seed_text: str) -> list[float] | None:
    sampled = sample_polyline(polyline, n=16)
    if not sampled:
        return None
    rng = random.Random(seed_text)
    return rng.choice(sampled)
