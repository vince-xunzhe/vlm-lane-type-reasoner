#!/usr/bin/env python3
"""Convert chanxian LaneType VLM per-frame output to final overall lane attr JSON."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

try:
    from tqdm import tqdm
except ImportError:  # tqdm is optional; conversion should still work without it.
    tqdm = None


JsonDict = Dict[str, Any]
LaneDict = Dict[str, Any]


LANE_ATTRIBUTE_MAPPING: Dict[str, str] = {
    "0": "\u666e\u901a\u8f66\u9053",
    "1": "\u516c\u4ea4\u8f66\u9053",
    "2": "\u81ea\u884c\u8f66\u9053",
    "3": "\u53ef\u53d8\u8f66\u9053",
    "4": "\u6f6e\u6c50\u8f66\u9053",
}


VLM_ATTRIBUTE_TO_ATTR: Dict[str, int] = {
    "normal": 0,
    "bus": 1,
    "bicycle": 2,
    "variable": 3,
    "tidal": 4,
}


DEFAULT_FORWARD_SAMPLE_Y: List[float] = [
    8.0,
    13.333333333333332,
    18.666666666666664,
    24.0,
]
DEFAULT_VISIBLE_GEO_Y_RANGE: List[float] = [8.0, 24.0]
OUTPUT_JSON_NAME = "output_lanes_attr.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert per-frame chanxian LaneType VLM JSON outputs into "
            "overall_fake_lane_attr.json-compatible format."
        )
    )
    parser.add_argument(
        "--root-path",
        help=(
            "Root data path. The script derives bagId from its last path component "
            "and reads <root-path>/<bagId>_r_LaneCenterLine/inference/output."
        ),
    )
    parser.add_argument(
        "--vlm-output-dir",
        default=None,
        help="Directory containing per-frame VLM output JSON files.",
    )
    parser.add_argument(
        "--output-folder",
        default=None,
        help=(
            "Directory to save the converted overall lane attribute JSON. "
            "If omitted with --root-path, defaults to "
            "<root-path>/<bagId>_r_LaneCenterLine/output. "
            f"The output file name is always {OUTPUT_JSON_NAME}."
        ),
    )
    parser.add_argument(
        "--forward-sample-y",
        nargs="+",
        type=float,
        default=DEFAULT_FORWARD_SAMPLE_Y,
        help="forward_sample_y field written to non-empty frames.",
    )
    parser.add_argument(
        "--visible-geo-y-range",
        nargs=2,
        type=float,
        default=DEFAULT_VISIBLE_GEO_Y_RANGE,
        metavar=("START_Y", "END_Y"),
        help="visible_geo_y_range field written to non-empty frames.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indent for output. Use a negative value for compact JSON.",
    )
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="Allow overwriting output_lanes_attr.json if it already exists.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bar even if tqdm is installed.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level.",
    )
    return parser.parse_args()


def setup_logging(log_level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="[%(levelname)s] %(message)s",
    )


def iter_vlm_json_files(vlm_output_dir: Path) -> List[Path]:
    if not vlm_output_dir.exists():
        raise FileNotFoundError(f"VLM output directory does not exist: {vlm_output_dir}")
    if not vlm_output_dir.is_dir():
        raise NotADirectoryError(f"VLM output path is not a directory: {vlm_output_dir}")
    return sorted(p for p in vlm_output_dir.glob("*.json") if p.is_file())


def resolve_vlm_output_dir_from_root_path(root_path: Path) -> Path:
    if not root_path.exists():
        raise FileNotFoundError(f"root_path does not exist: {root_path}")
    if not root_path.is_dir():
        raise NotADirectoryError(f"root_path is not a directory: {root_path}")

    bag_id = root_path.name
    if not bag_id:
        raise ValueError(f"Cannot derive bagId from root_path: {root_path}")

    lane_centerline_dir = root_path / f"{bag_id}_r_LaneCenterLine"
    if not lane_centerline_dir.exists():
        raise FileNotFoundError(
            f"Target directory does not exist: {lane_centerline_dir}"
        )
    if not lane_centerline_dir.is_dir():
        raise NotADirectoryError(
            f"Target path exists but is not a directory: {lane_centerline_dir}"
        )

    inference_dir = lane_centerline_dir / "inference"
    if not inference_dir.exists():
        raise FileNotFoundError(f"inference subdirectory does not exist: {inference_dir}")
    if not inference_dir.is_dir():
        raise NotADirectoryError(
            f"inference path exists but is not a directory: {inference_dir}"
        )

    output_dir = inference_dir / "output"
    if not output_dir.exists():
        raise FileNotFoundError(f"output subdirectory does not exist: {output_dir}")
    if not output_dir.is_dir():
        raise NotADirectoryError(f"output path exists but is not a directory: {output_dir}")

    vlm_output_dir = output_dir / "center_line_2d"
    if not vlm_output_dir.exists():
        raise FileNotFoundError(f"vlm output subdirectory does not exist: {vlm_output_dir}")
    if not vlm_output_dir.is_dir():
        raise NotADirectoryError(f"vlm output path exists but is not a directory: {vlm_output_dir}")

    return vlm_output_dir


def resolve_vlm_output_dir(args: argparse.Namespace) -> Path:
    if args.root_path:
        if args.vlm_output_dir:
            logging.warning("--root-path is provided; --vlm-output-dir will be ignored.")
        return resolve_vlm_output_dir_from_root_path(Path(args.root_path))
    if args.vlm_output_dir:
        return Path(args.vlm_output_dir)
    raise ValueError("Either --root-path or --vlm-output-dir must be provided.")


def resolve_output_json_path(args: argparse.Namespace) -> Path:
    if args.output_folder:
        return Path(args.output_folder) / OUTPUT_JSON_NAME
    if not args.root_path:
        raise ValueError("--output-folder is required when --root-path is not provided.")

    root_path = Path(args.root_path)
    bag_id = root_path.name
    if not bag_id:
        raise ValueError(f"Cannot derive bagId from root_path: {root_path}")

    output_dir = root_path / f"{bag_id}_r_LaneCenterLine" / "output"
    return output_dir / OUTPUT_JSON_NAME


def load_json(json_path: Path) -> JsonDict:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {json_path}, got {type(data).__name__}")
    return data


def normalize_vlm_attribute(attribute: Any, json_path: Path, lane_id: str) -> int:
    attr_key = str(attribute).strip().lower()
    if attr_key not in VLM_ATTRIBUTE_TO_ATTR:
        raise ValueError(
            f"Unknown lane attribute {attribute!r} in {json_path}, lane id={lane_id}. "
            f"Expected one of: {sorted(VLM_ATTRIBUTE_TO_ATTR)}"
        )
    return VLM_ATTRIBUTE_TO_ATTR[attr_key]


def add_attr_to_points(points: Any, attr_value: int) -> List[JsonDict]:
    if points is None:
        return []
    if not isinstance(points, list):
        raise ValueError(f"Expected point list, got {type(points).__name__}")

    output: List[JsonDict] = []
    for point in points:
        if not isinstance(point, dict):
            raise ValueError(f"Expected point object, got {type(point).__name__}")
        point_with_attr = dict(point)
        point_with_attr["Attr"] = attr_value
        output.append(point_with_attr)
    return output


def convert_lane(lane: Any, json_path: Path) -> tuple[str, LaneDict]:
    if not isinstance(lane, dict):
        raise ValueError(f"Expected lane object in {json_path}, got {type(lane).__name__}")

    lane_id = str(lane.get("id", "")).strip()
    if not lane_id:
        raise ValueError(f"Missing lane id in {json_path}")

    attr_value = normalize_vlm_attribute(lane.get("attribute"), json_path, lane_id)
    converted_lane: LaneDict = {
        "id": lane_id,
        "points_utm": add_attr_to_points(lane.get("points_utm"), attr_value),
        "Geo_points_utm": add_attr_to_points(lane.get("Geo_points_utm"), attr_value),
    }
    return lane_id, converted_lane


def convert_frame(
    frame_data: JsonDict,
    json_path: Path,
    forward_sample_y: Sequence[float],
    visible_geo_y_range: Sequence[float],
) -> JsonDict:
    lanes = frame_data.get("lane", [])
    if lanes in (None, []):
        return {}
    if not isinstance(lanes, list):
        raise ValueError(f"Expected 'lane' to be a list in {json_path}")

    converted_lanes: Dict[str, LaneDict] = {}
    for lane in lanes:
        lane_id, converted_lane = convert_lane(lane, json_path)
        converted_lanes[lane_id] = converted_lane

    if not converted_lanes:
        return {}

    return {
        "forward_sample_y": list(forward_sample_y),
        "visible_geo_y_range": list(visible_geo_y_range),
        "lane_attribute_mapping": dict(LANE_ATTRIBUTE_MAPPING),
        "lane": converted_lanes,
    }


def convert_vlm_outputs(
    vlm_output_dir: Path,
    forward_sample_y: Sequence[float],
    visible_geo_y_range: Sequence[float],
    show_progress: bool = True,
) -> JsonDict:
    json_files = iter_vlm_json_files(vlm_output_dir)
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {vlm_output_dir}")

    iterator: Iterable[Path] = json_files
    if show_progress and tqdm is not None:
        iterator = tqdm(json_files, desc="Converting VLM outputs")

    output: JsonDict = {}
    non_empty_count = 0
    lane_count = 0
    for json_path in iterator:
        image_name = f"{json_path.stem}.jpg"
        frame_data = load_json(json_path)
        converted_frame = convert_frame(
            frame_data,
            json_path,
            forward_sample_y=forward_sample_y,
            visible_geo_y_range=visible_geo_y_range,
        )
        if converted_frame:
            non_empty_count += 1
            lane_count += len(converted_frame.get("lane", {}))
        output[image_name] = converted_frame

    logging.info(
        "Converted %d frames, %d non-empty frames, %d lanes.",
        len(json_files),
        non_empty_count,
        lane_count,
    )
    return output


def save_json(data: JsonDict, output_json: Path, indent: int, exist_ok: bool) -> None:
    if output_json.exists() and not exist_ok:
        raise FileExistsError(f"Output JSON already exists: {output_json}")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    json_indent: Optional[int] = indent if indent >= 0 else None
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=json_indent)
        if json_indent is None:
            f.write("\n")


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    vlm_output_dir = resolve_vlm_output_dir(args)
    output_json = resolve_output_json_path(args)
    logging.info("Using VLM output directory: %s", vlm_output_dir)
    logging.info("Using output JSON path: %s", output_json)

    converted = convert_vlm_outputs(
        vlm_output_dir=vlm_output_dir,
        forward_sample_y=args.forward_sample_y,
        visible_geo_y_range=args.visible_geo_y_range,
        show_progress=not args.no_progress,
    )
    save_json(
        converted,
        output_json=output_json,
        indent=args.indent,
        exist_ok=args.exist_ok,
    )
    logging.info("Saved converted JSON to %s", output_json)


if __name__ == "__main__":
    main()
