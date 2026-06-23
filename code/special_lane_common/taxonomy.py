"""Canonical labels and mappings for the special-lane task."""

from __future__ import annotations

SPECIAL_LANE_CLASSES = ("bus", "tidal", "variable", "bicycle", "normal")

CN_TO_CANONICAL = {
    "公交": "bus",
    "潮汐": "tidal",
    "可变": "variable",
    "自行车": "bicycle",
    "自行": "bicycle",
    "普通": "normal",
    "正常": "normal",
    "normal": "normal",
    "bus": "bus",
    "tidal": "tidal",
    "variable": "variable",
    "bicycle": "bicycle",
}

CANONICAL_TO_CN = {
    "bus": "公交",
    "tidal": "潮汐",
    "variable": "可变",
    "bicycle": "自行车",
    "normal": "普通",
}

LINE_COLORS = ("white", "yellow", "unknown")
LINE_PATTERNS = ("solid", "dashed", "unknown")
LINE_MULTIPLICITIES = ("single", "double", "unknown")
LINE_SHAPES = ("straight_or_curve", "zigzag", "others")


def canonical_lane_type(value: str | None) -> str:
    if value is None:
        return "normal"
    return CN_TO_CANONICAL.get(str(value).strip(), "normal")


def one_hot(label: str) -> dict[str, int]:
    canonical = canonical_lane_type(label)
    return {item: int(item == canonical) for item in SPECIAL_LANE_CLASSES}

