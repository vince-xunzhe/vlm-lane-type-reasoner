"""Write Alpha Clean85 predictions back to LAS centerline lane attributes."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


CN_LABELS = {
    "bus": "公交",
    "tidal": "潮汐",
    "variable": "可变",
    "bicycle": "自行车",
    "normal": "普通",
}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)


def resolve_default_output_dir(manifest: dict[str, Any]) -> Path:
    las_dir = Path(manifest["las_dir"])
    las_name = manifest["las_name"]
    return las_dir / f"{las_name}_r_LaneCenterLine" / "inference" / "output" / "center_line_2d"


def resolve_default_summary_path(manifest: dict[str, Any]) -> Path:
    las_dir = Path(manifest["las_dir"])
    las_name = manifest["las_name"]
    return las_dir / f"{las_name}_r_LaneCenterLine" / "inference" / "output" / "lane_attribute_summary.json"


def load_prediction_records(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    payload = load_json(path)
    if isinstance(payload, list):
        records = payload
    else:
        records = payload.get("records", [])
    return {
        (str(record.get("image_id")), str(record.get("lane_id"))): record
        for record in records
        if record.get("image_id") is not None and record.get("lane_id") is not None
    }


def format_attribute(label: str | None, mode: str) -> str:
    if not label:
        return "invalid_no_prediction"
    if mode == "cn":
        return CN_LABELS.get(label, label)
    return label


def write_attributes(args: argparse.Namespace) -> dict[str, Any]:
    manifest = load_json(Path(args.manifest))
    predictions = load_prediction_records(Path(args.predictions))
    output_dir = Path(args.output_dir) if args.output_dir else resolve_default_output_dir(manifest)
    summary_path = Path(args.summary) if args.summary else resolve_default_summary_path(manifest)

    attribute_counts: Counter[str] = Counter()
    missing_predictions = 0
    written_files = 0
    invalid_files = 0

    for item in manifest.get("records", []):
        centerline_path = Path(item["centerline_path"])
        payload = load_json(centerline_path)
        lanes = list(payload.get("lane") or [])
        for lane_info in item.get("lanes", []):
            lane_index = int(lane_info["lane_index"])
            key = (str(item["image_id"]), str(lane_info["internal_lane_id"]))
            record = predictions.get(key)
            pred_type = record.get("pred_type") if record else None
            if not pred_type:
                missing_predictions += 1
            attribute = format_attribute(pred_type, args.attribute_format)
            if 0 <= lane_index < len(lanes) and isinstance(lanes[lane_index], dict):
                lanes[lane_index]["attribute"] = attribute
                attribute_counts[attribute] += 1

        for invalid_lane in item.get("invalid_lanes", []):
            lane_index = int(invalid_lane.get("lane_index", -1))
            if 0 <= lane_index < len(lanes) and isinstance(lanes[lane_index], dict):
                lanes[lane_index]["attribute"] = invalid_lane.get("reason", "invalid_lane")
                attribute_counts[lanes[lane_index]["attribute"]] += 1

        dump_json(output_dir / centerline_path.name, {"lane": lanes})
        written_files += 1

    for item in manifest.get("invalid_files", []):
        centerline_path = Path(item["centerline_path"])
        payload = load_json(centerline_path)
        lanes = list(payload.get("lane") or [])
        reason = item.get("reason", "invalid_centerline")
        for lane in lanes:
            if isinstance(lane, dict):
                lane["attribute"] = reason
                attribute_counts[reason] += 1
        dump_json(output_dir / centerline_path.name, {"lane": lanes})
        written_files += 1
        invalid_files += 1

    summary = {
        "schema_version": "las_alpha_clean85_lane_attribute_summary/v1",
        "manifest": str(Path(args.manifest)),
        "predictions": str(Path(args.predictions)),
        "output_dir": str(output_dir),
        "attribute_format": args.attribute_format,
        "prediction_records": len(predictions),
        "written_files": written_files,
        "invalid_files": invalid_files,
        "missing_predictions": missing_predictions,
        "attribute_distribution": dict(attribute_counts),
    }
    dump_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--summary", default=None)
    parser.add_argument("--attribute-format", choices=("canonical", "cn"), default="canonical")
    return parser.parse_args()


if __name__ == "__main__":
    write_attributes(parse_args())
