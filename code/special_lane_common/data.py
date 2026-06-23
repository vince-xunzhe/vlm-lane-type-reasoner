"""Dataset readers for raw annotations, lane detections, and object detections."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .image_utils import image_size
from .taxonomy import canonical_lane_type, one_hot


DEFAULT_OBJECT_LABELS = {
    0: "bus_text_gong",
    1: "bus_text_jiao",
    2: "bus_related_time_restriction_sign",
    4: "variable_text_ke",
    5: "variable_text_bian",
    6: "mixed_lane_signal_candidate",
    9: "bicycle_sign",
    10: "bicycle_icon",
}


EMPTY_ONE_HOT = {
    "bus": 0,
    "tidal": 0,
    "variable": 0,
    "bicycle": 0,
    "normal": 0,
}


@dataclass
class DatasetSample:
    image_id: str
    image_path: Path
    annotation_path: Path
    lane_detection_path: Path | None
    object_detection_path: Path | None
    width: int
    height: int
    lanes: list[dict]
    boundaries: list[dict]
    objects: list[dict]


def load_json(path: Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as fp:
        return json.load(fp)


def dump_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)


def append_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fp:
        for record in records:
            fp.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_json_or_jsonl(path: Path) -> list[dict]:
    path = Path(path)
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    payload = load_json(path)
    if isinstance(payload, list):
        return payload
    if "records" in payload:
        return payload["records"]
    return [payload]


def _find_file(directory: Path, image_id: str) -> Path | None:
    for name in (f"{image_id}.json", f"{image_id}_segments.json"):
        path = directory / name
        if path.exists():
            return path
    return None


def _is_boundary_detection_payload(payload: dict) -> bool:
    return "polylines_by_id" in payload or "segments_info" in payload


def _is_object_detection_payload(payload: dict) -> bool:
    return (
        "bboxes_info" in payload
        or "labels_info" in payload
        or "bboxes" in payload
        or "labels" in payload
    )


def _resolve_detection_paths(data_dir: Path, image_id: str) -> tuple[Path | None, Path | None]:
    candidates = []
    for folder in ("lanes", "objects"):
        path = _find_file(data_dir / folder, image_id)
        if path is not None:
            candidates.append(path)

    boundary_path = None
    object_path = None
    for path in candidates:
        payload = load_json(path)
        if _is_boundary_detection_payload(payload):
            boundary_path = path
        if _is_object_detection_payload(payload):
            object_path = path
    return boundary_path, object_path


def load_lanes_from_annotation(annotation_path: Path) -> list[dict]:
    raw = load_json(annotation_path)
    lanes = []
    for lane in raw.get("lanes", []):
        has_label = "lane_type" in lane and lane.get("lane_type") is not None
        label = canonical_lane_type(lane.get("lane_type")) if has_label else None
        lanes.append(
            {
                "lane_id": f"L{lane['id']}",
                "source_id": lane["id"],
                "centerline": lane.get("grounding", []),
                "grounding_type": lane.get("grounding_type", "line"),
                "gt_type": label,
                "gt_type_cn": lane.get("lane_type") if has_label else None,
                "gt_one_hot": one_hot(label) if label else dict(EMPTY_ONE_HOT),
            }
        )
    return lanes


def load_boundary_detections(path: Path | None, category_ids: set[int] | None = None) -> list[dict]:
    if path is None:
        return []
    payload = load_json(path)
    if not _is_boundary_detection_payload(payload):
        return []

    segments = {str(item.get("id")): item for item in payload.get("segments_info", [])}
    out = []
    for seg_id, polylines in payload.get("polylines_by_id", {}).items():
        meta = segments.get(str(seg_id), {})
        category_id = meta.get("category_id")
        category_name = meta.get("category_name") or meta.get("label_name") or meta.get("name")
        if category_ids is not None and category_id not in category_ids:
            continue
        if category_ids is None and category_name and category_name != "single_line":
            continue
        if not polylines:
            continue
        polyline = max(polylines, key=lambda p: len(p))
        out.append(
            {
                "boundary_id": f"B{seg_id}",
                "source_id": seg_id,
                "category_id": category_id,
                "category_name": category_name,
                "score": meta.get("score"),
                "polyline": polyline,
                "all_polylines": polylines,
            }
        )
    return out


def load_object_detections(path: Path | None, label_map: dict[int, str] | None = None) -> list[dict]:
    if path is None:
        return []
    payload = load_json(path)
    if not _is_object_detection_payload(payload):
        return []
    label_map = label_map or DEFAULT_OBJECT_LABELS
    labels = payload.get("labels_info", payload.get("labels", []))
    scores = payload.get("scores_info", payload.get("scores", []))
    bboxes = payload.get("bboxes_info", payload.get("bboxes", []))
    out = []
    for idx, bbox in enumerate(bboxes):
        label_id = labels[idx] if idx < len(labels) else None
        out.append(
            {
                "object_id": f"O{idx}",
                "label_id": label_id,
                "label_name": label_map.get(label_id, f"det_label_{label_id}"),
                "score": scores[idx] if idx < len(scores) else None,
                "bbox": [float(v) for v in bbox],
            }
        )
    return out


def iter_dataset_samples(
    data_dir: Path,
    *,
    limit: int | None = None,
    line_category_ids: set[int] | None = None,
    object_label_map: dict[int, str] | None = None,
) -> list[DatasetSample]:
    data_dir = Path(data_dir)
    samples = []
    for annotation_path in sorted((data_dir / "jsons").glob("*.json")):
        if annotation_path.name.startswith("._"):
            continue
        image_id = annotation_path.stem
        image_path = data_dir / "images" / f"{image_id}.jpg"
        if not image_path.exists():
            continue
        boundary_path, object_path = _resolve_detection_paths(data_dir, image_id)
        width, height = image_size(image_path)
        samples.append(
            DatasetSample(
                image_id=image_id,
                image_path=image_path.resolve(),
                annotation_path=annotation_path.resolve(),
                lane_detection_path=boundary_path.resolve() if boundary_path else None,
                object_detection_path=object_path.resolve() if object_path else None,
                width=width,
                height=height,
                lanes=load_lanes_from_annotation(annotation_path),
                boundaries=load_boundary_detections(boundary_path, category_ids=line_category_ids),
                objects=load_object_detections(object_path, label_map=object_label_map),
            )
        )
        if limit is not None and len(samples) >= limit:
            break
    return samples
