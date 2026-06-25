#!/usr/bin/env python3
"""Write temporary LaneType output from depth-aware object-lane association.

The production Clean85 path still uses Qwen evidence and the trained decision
head.  This temporary path is intentionally rule-based: every lane defaults to
``normal`` and only strong depth-aware object associations override the lane
attribute.  The output interface stays identical to the normal LaneType output.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any


CODE_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = CODE_DIR.parent

LABEL_TO_ATTR = {
    0: "bus",
    1: "bus",
    2: "bus",
    4: "variable",
    5: "variable",
    9: "bicycle",
    10: "bicycle",
}
SIGN_SIGNAL_LABEL_IDS = {2, 9}
ROAD_MARKING_LABEL_IDS = {0, 1, 4, 5, 10}
ATTR_PRIORITY = {
    "bus": 50,
    "bicycle": 40,
    "variable": 30,
    "tidal": 20,
    "normal": 0,
}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def dump_json(path: Path, payload: Any, *, indent: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=indent)
        fp.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--las-dir", type=Path, default=None, help="LAS root directory, e.g. /.../3702-1-00L025-260316")
    parser.add_argument("--base-dir", type=Path, default=None, help="LaneCenterLine directory. Overrides --las-dir.")
    parser.add_argument("--round-name", default=os.environ.get("ROUND_NAME", "depth-aware-temp-decision"))
    parser.add_argument("--association-results", type=Path, default=None, help="Existing association_results.json to consume.")
    parser.add_argument("--run-association", choices=("auto", "always", "never"), default="auto")
    parser.add_argument("--association-all-frames", action="store_true", help="Ignore visualize_instances.yaml when running association.")
    parser.add_argument("--instances-yaml", type=Path, default=None, help="Instance YAML for association/visualization selection.")
    parser.add_argument("--frames", default="", help="Comma-separated frame stems, forwarded to association/visualization.")
    parser.add_argument("--limit", type=int, default=0, help="Frame limit forwarded to association/visualization.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--min-top-score", type=float, default=0.35)
    parser.add_argument("--min-top-margin", type=float, default=0.0)
    parser.add_argument("--sign-weight", type=float, default=1.20)
    parser.add_argument("--road-marking-weight", type=float, default=1.00)
    parser.add_argument("--det-score-weight", type=float, default=0.25, help="How much detector confidence modulates evidence strength.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Per-frame center_line_2d output directory.")
    parser.add_argument("--final-output-dir", type=Path, default=None, help="Directory where output_lanes_attr.json is written.")
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--no-visualization", action="store_true")
    parser.add_argument("--visualization-output-dir", type=Path, default=None)
    return parser.parse_args()


def lane_centerline_dir_from_las_dir(las_dir: Path) -> Path:
    las_dir = las_dir.expanduser().resolve()
    return las_dir / f"{las_dir.name}_r_LaneCenterLine"


def resolve_base_dir(args: argparse.Namespace) -> Path:
    if args.base_dir is not None:
        return args.base_dir.expanduser().resolve()
    if args.las_dir is not None:
        return lane_centerline_dir_from_las_dir(args.las_dir)
    raise SystemExit("Provide --base-dir or --las-dir.")


def resolve_las_dir(base_dir: Path, args: argparse.Namespace) -> Path | None:
    if args.las_dir is not None:
        return args.las_dir.expanduser().resolve()
    suffix = "_r_LaneCenterLine"
    if base_dir.name.endswith(suffix):
        candidate = base_dir.parent
        if candidate.exists():
            return candidate
    return None


def default_paths(base_dir: Path, round_name: str) -> dict[str, Path]:
    if round_name:
        return {
            "association": base_dir / "inference" / round_name / "association_3d" / "association_results.json",
            "temp_root": base_dir / "inference" / round_name / "depth_aware_temp_decision",
            "output_dir": base_dir / "inference" / round_name / "depth_aware_temp_decision" / "center_line_2d",
            "final_output_dir": base_dir / "output" / round_name,
            "visualization_output_dir": base_dir / "vis_debug" / round_name / "depth_aware_temp_decision",
        }
    return {
        "association": base_dir / "inference" / "depth_aware_temp_decision" / "association_3d" / "association_results.json",
        "temp_root": base_dir / "inference" / "depth_aware_temp_decision",
        "output_dir": base_dir / "inference" / "depth_aware_temp_decision" / "center_line_2d",
        "final_output_dir": base_dir / "output" / "depth_aware_temp_decision",
        "visualization_output_dir": base_dir / "vis_debug" / "depth_aware_temp_decision",
    }


def clean_dir(path: Path) -> None:
    if not path.exists():
        return
    parts = set(path.resolve().parts)
    if "inference" not in parts and "vis_debug" not in parts and "output" not in parts:
        raise ValueError(f"Refuse to clean unexpected directory: {path}")
    for child in path.iterdir():
        if child.is_dir():
            clean_dir(child)
            child.rmdir()
        else:
            child.unlink()


def run_command(cmd: list[str]) -> None:
    print("[cmd]", " ".join(str(item) for item in cmd), flush=True)
    subprocess.run([str(item) for item in cmd], check=True)


def run_association_probe(base_dir: Path, args: argparse.Namespace, association_path: Path) -> None:
    probe = CODE_DIR / "scripts" / "probe_3d_lane_association.py"
    cmd: list[str] = [
        sys.executable,
        str(probe),
        "--base-dir",
        str(base_dir),
        "--round-name",
        str(args.round_name or ""),
        "--workers",
        str(args.workers),
    ]
    if args.overwrite:
        cmd.append("--overwrite")
    if args.limit > 0:
        cmd.extend(["--limit", str(args.limit)])
    if args.frames:
        cmd.extend(["--frames", args.frames])
    if args.association_all_frames:
        cmd.extend(["--instances-yaml", str(association_path.parent / "__all_frames_no_yaml__.yaml")])
    elif args.instances_yaml is not None:
        cmd.extend(["--instances-yaml", str(args.instances_yaml.expanduser().resolve())])
    run_command(cmd)


def ensure_association(base_dir: Path, args: argparse.Namespace, association_path: Path) -> None:
    should_run = args.run_association == "always" or (args.run_association == "auto" and not association_path.exists())
    if should_run:
        run_association_probe(base_dir, args, association_path)
    if not association_path.exists():
        raise FileNotFoundError(f"Association result not found: {association_path}")


def normalize_label_id(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def normalize_score(value: Any, default: float = 0.0) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(score):
        return default
    return score


def evidence_weight(label_id: int | None, args: argparse.Namespace) -> float:
    if label_id in SIGN_SIGNAL_LABEL_IDS:
        return float(args.sign_weight)
    if label_id in ROAD_MARKING_LABEL_IDS:
        return float(args.road_marking_weight)
    return 1.0


def detector_factor(det_score: Any, det_score_weight: float) -> float:
    score = normalize_score(det_score, default=1.0)
    score = min(1.0, max(0.0, score))
    weight = min(1.0, max(0.0, float(det_score_weight)))
    return (1.0 - weight) + weight * score


def collect_frame_evidence(item: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for obj in item.get("objects") or []:
        if obj.get("filtered"):
            continue
        label_id = normalize_label_id(obj.get("label_id"))
        attr = LABEL_TO_ATTR.get(label_id)
        if not attr:
            continue
        assignment = obj.get("assignment") or {}
        lane_id = assignment.get("top_lane_id")
        if lane_id is None:
            continue
        top_score = normalize_score(assignment.get("top_score"), default=0.0)
        top_margin = normalize_score(assignment.get("top2_margin"), default=top_score)
        if top_score < float(args.min_top_score):
            continue
        if top_margin < float(args.min_top_margin):
            continue
        strength = top_score * evidence_weight(label_id, args) * detector_factor(obj.get("score"), args.det_score_weight)
        evidence.append(
            {
                "frame": item.get("frame"),
                "object_id": obj.get("object_id"),
                "label_id": label_id,
                "label_name": obj.get("label_name"),
                "lane_id": str(lane_id),
                "attribute": attr,
                "top_score": top_score,
                "top2_margin": top_margin,
                "strength": float(strength),
                "method": assignment.get("method"),
            }
        )
    return evidence


def choose_lane_attribute(evidence_by_attr: dict[str, list[dict[str, Any]]]) -> tuple[str, float, list[dict[str, Any]]]:
    best_attr = "normal"
    best_score = 0.0
    best_evidence: list[dict[str, Any]] = []
    for attr, items in evidence_by_attr.items():
        score = sum(float(item.get("strength") or 0.0) for item in items)
        if (
            score > best_score + 1e-9
            or (
                abs(score - best_score) <= 1e-9
                and ATTR_PRIORITY.get(attr, 0) > ATTR_PRIORITY.get(best_attr, 0)
            )
        ):
            best_attr = attr
            best_score = score
            best_evidence = list(items)
    return best_attr, best_score, best_evidence


def build_evidence_index(association_payload: dict[str, Any], args: argparse.Namespace) -> dict[str, dict[str, dict[str, list[dict[str, Any]]]]]:
    by_frame: dict[str, dict[str, dict[str, list[dict[str, Any]]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for item in association_payload.get("results") or []:
        frame = item.get("frame")
        if not frame or not item.get("ok", False):
            continue
        for evidence in collect_frame_evidence(item, args):
            by_frame[str(frame)][evidence["lane_id"]][evidence["attribute"]].append(evidence)
    return by_frame


def load_centerline_files(base_dir: Path) -> list[Path]:
    center_dir = base_dir / "center_line_2d" / "output"
    if not center_dir.exists():
        raise FileNotFoundError(f"Missing center_line_2d output directory: {center_dir}")
    files = sorted(center_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No centerline JSON files found in {center_dir}")
    return files


def write_temp_centerline_outputs(
    base_dir: Path,
    output_dir: Path,
    association_payload: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    evidence_by_frame = build_evidence_index(association_payload, args)
    if args.overwrite:
        clean_dir(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    attr_counter: Counter[str] = Counter()
    evidence_counter: Counter[str] = Counter()
    frame_summaries: list[dict[str, Any]] = []
    written_files = 0
    lane_count = 0

    for center_path in load_centerline_files(base_dir):
        frame = center_path.stem
        payload = load_json(center_path)
        output_payload = deepcopy(payload)
        lanes = output_payload.get("lane") or []
        if not isinstance(lanes, list):
            lanes = []
            output_payload["lane"] = lanes

        frame_evidence = evidence_by_frame.get(frame, {})
        frame_summary = {
            "frame": frame,
            "lanes": [],
            "evidence_count": sum(len(items) for by_attr in frame_evidence.values() for items in by_attr.values()),
        }

        for lane_idx, lane in enumerate(lanes):
            if not isinstance(lane, dict):
                continue
            lane_id = str(lane.get("id", lane_idx))
            attr, score, evidence = choose_lane_attribute(frame_evidence.get(lane_id, {}))
            lane["attribute"] = attr
            attr_counter[attr] += 1
            lane_count += 1
            if evidence:
                evidence_counter[attr] += len(evidence)
                frame_summary["lanes"].append(
                    {
                        "lane_id": lane_id,
                        "attribute": attr,
                        "score": float(score),
                        "evidence": evidence,
                    }
                )

        dump_json(output_dir / center_path.name, {"lane": lanes})
        written_files += 1
        if frame_summary["lanes"]:
            frame_summaries.append(frame_summary)

    return {
        "schema_version": "depth_aware_temp_decision_summary/v1",
        "association_results": str(association_payload.get("output_dir") or association_payload.get("artifact_dir") or ""),
        "output_dir": str(output_dir),
        "written_files": written_files,
        "lane_count": lane_count,
        "attribute_distribution": dict(attr_counter),
        "evidence_distribution": dict(evidence_counter),
        "frames_with_special_evidence": len(frame_summaries),
        "frame_summaries": frame_summaries,
        "decision_policy": {
            "default_attribute": "normal",
            "label_to_attr": {str(k): v for k, v in LABEL_TO_ATTR.items()},
            "min_top_score": float(args.min_top_score),
            "min_top_margin": float(args.min_top_margin),
            "sign_weight": float(args.sign_weight),
            "road_marking_weight": float(args.road_marking_weight),
            "det_score_weight": float(args.det_score_weight),
        },
    }


def convert_to_final_output(output_dir: Path, final_output_dir: Path) -> Path:
    converter = CODE_DIR / "scripts" / "output_to_final_LaneType_output.py"
    run_command(
        [
            sys.executable,
            str(converter),
            "--vlm-output-dir",
            str(output_dir),
            "--output-folder",
            str(final_output_dir),
            "--exist-ok",
            "--no-progress",
        ]
    )
    final_path = final_output_dir / "output_lanes_attr.json"
    if not final_path.exists():
        raise FileNotFoundError(f"Final LaneType output was not created: {final_path}")
    return final_path


def run_visualization(base_dir: Path, las_dir: Path | None, final_output: Path, visualization_output_dir: Path, args: argparse.Namespace) -> None:
    visualizer = CODE_DIR / "scripts" / "visualize_lane_modules.py"
    cmd: list[str] = [
        sys.executable,
        str(visualizer),
        "--output-dir",
        str(visualization_output_dir),
        "--attr-path",
        str(final_output),
        "--workers",
        str(args.workers),
        "--summary-width",
        "640",
        "--overwrite",
    ]
    if las_dir is not None:
        cmd.extend(["--las-dir", str(las_dir)])
    else:
        cmd.extend(["--base-dir", str(base_dir)])
    if args.instances_yaml is not None:
        cmd.extend(["--instances-yaml", str(args.instances_yaml.expanduser().resolve())])
    if args.frames:
        cmd.extend(["--frames", args.frames])
    if args.limit > 0:
        cmd.extend(["--limit", str(args.limit)])
    run_command(cmd)


def main() -> None:
    args = parse_args()
    base_dir = resolve_base_dir(args)
    las_dir = resolve_las_dir(base_dir, args)
    paths = default_paths(base_dir, str(args.round_name or ""))

    association_path = args.association_results.expanduser().resolve() if args.association_results else paths["association"]
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else paths["output_dir"]
    final_output_dir = args.final_output_dir.expanduser().resolve() if args.final_output_dir else paths["final_output_dir"]
    summary_path = args.summary.expanduser().resolve() if args.summary else paths["temp_root"] / "summary.json"
    visualization_output_dir = args.visualization_output_dir.expanduser().resolve() if args.visualization_output_dir else paths["visualization_output_dir"]

    ensure_association(base_dir, args, association_path)
    association_payload = load_json(association_path)
    summary = write_temp_centerline_outputs(base_dir, output_dir, association_payload, args)
    final_output = convert_to_final_output(output_dir, final_output_dir)

    summary.update(
        {
            "base_dir": str(base_dir),
            "las_dir": str(las_dir) if las_dir is not None else None,
            "round_name": str(args.round_name or ""),
            "association_results": str(association_path),
            "final_output": str(final_output),
            "visualization_output_dir": str(visualization_output_dir) if not args.no_visualization else None,
        }
    )
    dump_json(summary_path, summary)

    if not args.no_visualization:
        run_visualization(base_dir, las_dir, final_output, visualization_output_dir, args)

    print(
        json.dumps(
            {
                "summary": str(summary_path),
                "per_frame_output_dir": str(output_dir),
                "final_output": str(final_output),
                "attribute_distribution": summary.get("attribute_distribution", {}),
                "frames_with_special_evidence": summary.get("frames_with_special_evidence"),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
