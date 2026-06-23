"""Prepare LAS lane-type inputs for the Alpha Clean85 release pipeline.

The production LAS layout is converted into the small four-folder layout used
by the alpha inference code:

  images/<image_id>.jpg
  jsons/<image_id>.json
  lanes/<image_id>_segments.json
  objects/<image_id>.json
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)


def resolve_las_paths(las_dir: Path) -> dict[str, Any]:
    las_dir = las_dir.resolve()
    las_name = las_dir.name
    lane_type_dir = las_dir / f"{las_name}_r_LaneCenterLine"
    return {
        "las_dir": las_dir,
        "las_name": las_name,
        "image_dir": las_dir / las_name / "Data" / "Img" / "Camera0",
        "centerline_dir": lane_type_dir / "center_line_2d" / "output",
        "lane_line_dir": lane_type_dir / "lanes",
        "object_dir": lane_type_dir / "objects" / "pred",
        "inference_dir": lane_type_dir / "inference",
    }


def points_from_lane(lane: dict[str, Any]) -> list[list[float]]:
    points = lane.get("points") or []
    parsed = []
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        try:
            parsed.append([float(point[0]), float(point[1])])
        except (TypeError, ValueError):
            continue
    return parsed


def clean_generated_output_dir(output_dir: Path) -> None:
    if not output_dir.exists():
        return
    parts = set(output_dir.resolve().parts)
    if "inference" not in parts:
        raise ValueError(f"Refuse to clean non-inference output directory: {output_dir}")
    shutil.rmtree(output_dir)


def materialize_file(src: Path, dst: Path, *, link_mode: str) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if link_mode == "copy":
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src.resolve())
    return True


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    paths = resolve_las_paths(Path(args.las_dir))
    las_name = str(paths["las_name"])
    inference_dir = Path(args.inference_dir) if args.inference_dir else paths["inference_dir"]
    output_dir = Path(args.output_data_dir) if args.output_data_dir else inference_dir / "output" / "input"
    manifest_path = Path(args.manifest) if args.manifest else inference_dir / "output" / "input_manifest.json"

    if args.overwrite:
        clean_generated_output_dir(output_dir)

    for key in ("image_dir", "centerline_dir", "lane_line_dir", "object_dir"):
        if not paths[key].exists():
            raise FileNotFoundError(f"Missing LAS input path {key}: {paths[key]}")

    for folder in ("images", "jsons", "lanes", "objects"):
        (output_dir / folder).mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    invalid_files: list[dict[str, Any]] = []
    stats = {
        "centerline_files": 0,
        "valid_images": 0,
        "valid_lane_records": 0,
        "invalid_centerline_files": 0,
        "invalid_lane_records": 0,
        "missing_images": 0,
        "missing_lane_line_files": 0,
        "missing_object_files": 0,
    }

    for centerline_path in sorted(paths["centerline_dir"].glob("*.json")):
        stats["centerline_files"] += 1
        payload = load_json(centerline_path)
        image_name = payload.get("image") or f"{centerline_path.stem}.jpg"
        image_id = Path(image_name).stem
        lanes = payload.get("lane")
        if not isinstance(lanes, list) or not lanes:
            invalid_files.append(
                {
                    "image_id": image_id,
                    "centerline_path": str(centerline_path),
                    "reason": "invalid_empty_lane",
                }
            )
            stats["invalid_centerline_files"] += 1
            continue

        image_path = paths["image_dir"] / image_name
        if not image_path.exists():
            invalid_files.append(
                {
                    "image_id": image_id,
                    "centerline_path": str(centerline_path),
                    "reason": "missing_image",
                    "image_path": str(image_path),
                    "lane_count": len(lanes),
                }
            )
            stats["invalid_centerline_files"] += 1
            stats["missing_images"] += 1
            continue

        converted_lanes = []
        lane_map = []
        invalid_lanes = []
        for lane_index, lane in enumerate(lanes):
            if not isinstance(lane, dict):
                invalid_lanes.append({"lane_index": lane_index, "reason": "lane_not_object"})
                stats["invalid_lane_records"] += 1
                continue
            points = points_from_lane(lane)
            if not points:
                invalid_lanes.append(
                    {
                        "lane_index": lane_index,
                        "original_lane_id": lane.get("id"),
                        "reason": "invalid_empty_points",
                    }
                )
                stats["invalid_lane_records"] += 1
                continue
            internal_id = len(converted_lanes)
            converted_lanes.append(
                {
                    "id": internal_id,
                    "grounding_type": "line",
                    "grounding": points,
                    "las_lane_id": lane.get("id"),
                    "las_lane_index": lane_index,
                }
            )
            lane_map.append(
                {
                    "internal_lane_id": f"L{internal_id}",
                    "internal_source_id": internal_id,
                    "lane_index": lane_index,
                    "original_lane_id": lane.get("id"),
                    "points_count": len(points),
                }
            )

        if not converted_lanes:
            invalid_files.append(
                {
                    "image_id": image_id,
                    "centerline_path": str(centerline_path),
                    "reason": "invalid_no_usable_lanes",
                    "invalid_lanes": invalid_lanes,
                }
            )
            stats["invalid_centerline_files"] += 1
            continue

        materialize_file(image_path, output_dir / "images" / f"{image_id}.jpg", link_mode=args.link_mode)

        lane_line_src = paths["lane_line_dir"] / f"{image_id}_segments.json"
        lane_line_dst = output_dir / "lanes" / f"{image_id}_segments.json"
        if not materialize_file(lane_line_src, lane_line_dst, link_mode=args.link_mode):
            stats["missing_lane_line_files"] += 1

        object_src = paths["object_dir"] / f"{image_id}.json"
        object_dst = output_dir / "objects" / f"{image_id}.json"
        if not materialize_file(object_src, object_dst, link_mode=args.link_mode):
            stats["missing_object_files"] += 1

        annotation_path = output_dir / "jsons" / f"{image_id}.json"
        dump_json(
            annotation_path,
            {
                "image": image_name,
                "source_centerline_path": str(centerline_path),
                "lanes": converted_lanes,
            },
        )
        stats["valid_images"] += 1
        stats["valid_lane_records"] += len(converted_lanes)
        records.append(
            {
                "image_id": image_id,
                "image_name": image_name,
                "centerline_path": str(centerline_path),
                "internal_annotation_path": str(annotation_path),
                "image_path": str(image_path),
                "lane_line_path": str(lane_line_src) if lane_line_src.exists() else None,
                "object_path": str(object_src) if object_src.exists() else None,
                "lanes": lane_map,
                "invalid_lanes": invalid_lanes,
            }
        )

    manifest = {
        "schema_version": "las_alpha_clean85_input_manifest/v1",
        "las_dir": str(paths["las_dir"]),
        "las_name": las_name,
        "input_paths": {
            "images": str(paths["image_dir"]),
            "centerline_output": str(paths["centerline_dir"]),
            "lane_lines": str(paths["lane_line_dir"]),
            "objects": str(paths["object_dir"]),
        },
        "generated_data_dir": str(output_dir),
        "stats": stats,
        "records": records,
        "invalid_files": invalid_files,
    }
    dump_json(manifest_path, manifest)
    print(json.dumps({"manifest": str(manifest_path), "generated_data_dir": str(output_dir), **stats}, ensure_ascii=False))
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--las-dir", required=True, help="Absolute LAS project directory, e.g. /.../3702-1-00L025-260315")
    parser.add_argument("--inference-dir", default=None)
    parser.add_argument("--output-data-dir", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--link-mode", choices=("symlink", "copy"), default="symlink")
    parser.add_argument("--overwrite", action="store_true", help="Clean and rebuild the generated inference input directory.")
    return parser.parse_args()


if __name__ == "__main__":
    prepare(parse_args())
