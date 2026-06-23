"""Train an internal decision head for one special-lane pipeline.

Inputs must all belong to the same pipeline family, for example several alpha
passes or several beta passes.  The script never joins alpha and beta unless the
caller explicitly passes both, which should not be done for independent
pipeline evaluation.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupKFold, LeaveOneOut
from sklearn.pipeline import Pipeline

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from special_lane_common.data import dump_json, read_json_or_jsonl
from special_lane_common.taxonomy import SPECIAL_LANE_CLASSES, one_hot


CLASSES = list(SPECIAL_LANE_CLASSES)


def parse_named_input(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--input items must be name=path")
    name, path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("input name cannot be empty")
    return name, Path(path)


def load_records(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    return {(record["image_id"], record["lane_id"]): record for record in read_json_or_jsonl(path)}


def add_num(feature: dict[str, Any], key: str, value: Any, default: float = 0.0) -> None:
    try:
        feature[key] = float(value if value is not None else default)
    except (TypeError, ValueError):
        feature[key] = default


def visible_text_categories(value: Any) -> set[str]:
    text = str(value or "").strip().lower()
    if not text or text in {"none", "null", "无", "unknown", "missing"}:
        return set()
    categories: set[str] = set()
    if "公" in text or "交" in text or "公交" in text or "bus" in text:
        categories.add("bus")
    if "可" in text or "变" in text or "ke" in text or "bian" in text or "variable" in text:
        categories.add("variable")
    if "潮" in text or "汐" in text or "tidal" in text:
        categories.add("tidal")
    if "自行" in text or "bicycle" in text or "bike" in text:
        categories.add("bicycle")
    if text in {"x", "red_x", "red x"} or "红叉" in text:
        categories.add("red_x")
    return categories


def add_record_features(feature: dict[str, Any], prefix: str, record: dict[str, Any]) -> None:
    prediction = record.get("prediction") or {}
    pred_type = record.get("pred_type") or prediction.get("special_lane_type") or "missing"
    feature[f"{prefix}.pred={pred_type}"] = 1
    add_num(feature, f"{prefix}.confidence", prediction.get("confidence"))

    for key, value in (record.get("fusion_scores") or prediction.get("fusion_scores") or {}).items():
        add_num(feature, f"{prefix}.score.{key}", value)

    attrs = prediction.get("line_attributes") or {}
    zigzag_confidences = []
    tidal_confidences = []
    side_zigzags: dict[str, int] = {}
    side_tidal_lines: dict[str, int] = {}
    for side in ("left", "right"):
        item = attrs.get(side) or {}
        text = json.dumps(item, ensure_ascii=False).lower()
        for key in ("color", "pattern", "multiplicity", "shape", "label"):
            feature[f"{prefix}.{side}.{key}={str(item.get(key) or 'missing').lower()}"] = 1
        add_num(feature, f"{prefix}.{side}.line_confidence", item.get("confidence"))
        is_zigzag = int(
            "zigzag" in text or item.get("is_variable_zigzag_boundary") is True
        )
        is_tidal = int(
            item.get("is_tidal_double_yellow_dashed_boundary") is True
            or "double_yellow_dash" in text
            or ("double" in text and "yellow" in text and "dash" in text)
        )
        feature[f"{prefix}.{side}.zigzag"] = is_zigzag
        feature[f"{prefix}.{side}.tidal_line"] = is_tidal
        side_zigzags[side] = is_zigzag
        side_tidal_lines[side] = is_tidal
        try:
            confidence = float(item.get("confidence"))
        except (TypeError, ValueError):
            confidence = 0.0
        if is_zigzag:
            zigzag_confidences.append(confidence)
        if is_tidal:
            tidal_confidences.append(confidence)
    zigzag_count = sum(side_zigzags.values())
    tidal_count = sum(side_tidal_lines.values())
    feature[f"{prefix}.zigzag.side_count"] = zigzag_count
    feature[f"{prefix}.zigzag.both_sides"] = int(zigzag_count == 2)
    feature[f"{prefix}.zigzag.one_side"] = int(zigzag_count == 1)
    feature[f"{prefix}.zigzag.no_sides"] = int(zigzag_count == 0)
    feature[f"{prefix}.tidal_line.side_count"] = tidal_count
    feature[f"{prefix}.tidal_line.both_sides"] = int(tidal_count == 2)
    feature[f"{prefix}.tidal_line.one_side"] = int(tidal_count == 1)
    feature[f"{prefix}.tidal_line.no_sides"] = int(tidal_count == 0)
    feature[f"{prefix}.pred_variable_with_both_zigzag"] = int(pred_type == "variable" and zigzag_count == 2)
    feature[f"{prefix}.pred_variable_with_one_zigzag"] = int(pred_type == "variable" and zigzag_count == 1)
    feature[f"{prefix}.pred_variable_without_zigzag"] = int(pred_type == "variable" and zigzag_count == 0)
    feature[f"{prefix}.pred_normal_without_zigzag"] = int(pred_type == "normal" and zigzag_count == 0)
    feature[f"{prefix}.pred_tidal_with_both_tidal_lines"] = int(pred_type == "tidal" and tidal_count == 2)
    feature[f"{prefix}.pred_tidal_without_tidal_lines"] = int(pred_type == "tidal" and tidal_count == 0)
    for label, values in (("zigzag", zigzag_confidences), ("tidal_line", tidal_confidences)):
        feature[f"{prefix}.{label}.support_count"] = len(values)
        if values:
            feature[f"{prefix}.{label}.confidence_max"] = max(values)
            feature[f"{prefix}.{label}.confidence_min"] = min(values)
            feature[f"{prefix}.{label}.confidence_mean"] = sum(values) / len(values)

    supporting_sides = [str(side).lower() for side in prediction.get("supporting_line_sides") or []]
    feature[f"{prefix}.supporting_line_sides.count"] = len(supporting_sides)
    for side in ("left", "right"):
        feature[f"{prefix}.supporting_line_sides.{side}"] = int(side in supporting_sides)
    if supporting_sides:
        feature[f"{prefix}.supporting_line_sides.combo={'+'.join(sorted(supporting_sides))}"] = 1

    relation_by_id = {}
    for relation in prediction.get("object_relations") or []:
        if relation.get("object_id") is not None:
            relation_by_id[str(relation.get("object_id"))] = str(relation.get("relation") or "missing").lower()
        relation_name = str(relation.get("relation") or "missing").lower()
        label_name = str(relation.get("label_name") or "missing").lower()
        evidence_type = str(relation.get("evidence_type") or "missing").lower()
        feature[f"{prefix}.relation.{relation_name}"] = feature.get(f"{prefix}.relation.{relation_name}", 0) + 1
        feature[f"{prefix}.relation.{relation_name}.{label_name}"] = (
            feature.get(f"{prefix}.relation.{relation_name}.{label_name}", 0) + 1
        )
        feature[f"{prefix}.evidence.{evidence_type}"] = feature.get(f"{prefix}.evidence.{evidence_type}", 0) + 1

    for semantic in prediction.get("object_semantics") or []:
        semantic_type = str(semantic.get("semantic_type") or "missing").lower()
        class_prior = str(semantic.get("target_class_prior") or "missing").lower()
        feature[f"{prefix}.semantic.{semantic_type}"] = feature.get(f"{prefix}.semantic.{semantic_type}", 0) + 1
        feature[f"{prefix}.prior.{class_prior}"] = feature.get(f"{prefix}.prior.{class_prior}", 0) + 1
        object_id = semantic.get("object_id")
        relation_name = relation_by_id.get(str(object_id), "missing") if object_id is not None else "missing"
        for category in visible_text_categories(semantic.get("visible_text")):
            feature[f"{prefix}.visible_text.{category}"] = feature.get(f"{prefix}.visible_text.{category}", 0) + 1
            feature[f"{prefix}.visible_text.{category}.relation.{relation_name}"] = (
                feature.get(f"{prefix}.visible_text.{category}.relation.{relation_name}", 0) + 1
            )

    packet = record.get("request", {}).get("packet", {})
    target_lane = packet.get("target_lane") or {}
    lane_id = str(record.get("lane_id") or target_lane.get("lane_id") or "missing")
    source_lane_id = str(record.get("source_lane_id") or "missing")
    feature[f"{prefix}.lane_id={lane_id}"] = 1
    feature[f"{prefix}.source_lane_id={source_lane_id}"] = 1
    target_points = (
        target_lane.get("centerline_norm1000")
        or target_lane.get("centerline_points_hint")
        or []
    )
    target_stats: dict[str, float] = {}
    if target_points:
        xs = [float(point[0]) for point in target_points]
        ys = [float(point[1]) for point in target_points]
        target_stats = {
            "x_min": min(xs),
            "x_max": max(xs),
            "x_mean": sum(xs) / len(xs),
            "y_min": min(ys),
            "y_max": max(ys),
            "y_range": max(ys) - min(ys),
        }
        feature[f"{prefix}.target.x_min"] = min(xs)
        feature[f"{prefix}.target.x_max"] = max(xs)
        feature[f"{prefix}.target.x_mean"] = sum(xs) / len(xs)
        feature[f"{prefix}.target.y_min"] = min(ys)
        feature[f"{prefix}.target.y_max"] = max(ys)
        feature[f"{prefix}.target.y_mean"] = sum(ys) / len(ys)
        feature[f"{prefix}.target.dx"] = xs[-1] - xs[0]
        feature[f"{prefix}.target.dy"] = ys[-1] - ys[0]

    side_stats: dict[str, dict[str, float]] = {}
    for side_key in ("left_boundary", "right_boundary", "left_line_points", "right_line_points"):
        side = packet.get(side_key) or {}
        feature[f"{prefix}.{side_key}.category_id={side.get('category_id')}"] = 1
        add_num(feature, f"{prefix}.{side_key}.det_score", side.get("det_score"))
        points = side.get("polyline_norm1000") or side.get("points_2d") or []
        if points:
            xs = [float(point[0]) for point in points]
            ys = [float(point[1]) for point in points]
            feature[f"{prefix}.{side_key}.x_min"] = min(xs)
            feature[f"{prefix}.{side_key}.x_max"] = max(xs)
            feature[f"{prefix}.{side_key}.x_mean"] = sum(xs) / len(xs)
            feature[f"{prefix}.{side_key}.y_min"] = min(ys)
            feature[f"{prefix}.{side_key}.y_max"] = max(ys)
            feature[f"{prefix}.{side_key}.y_mean"] = sum(ys) / len(ys)
            side_stats[side_key] = {
                "x_mean": sum(xs) / len(xs),
                "y_min": min(ys),
                "y_max": max(ys),
                "y_range": max(ys) - min(ys),
            }
            if target_stats:
                overlap = max(0.0, min(max(ys), target_stats["y_max"]) - max(min(ys), target_stats["y_min"]))
                feature[f"{prefix}.{side_key}.target_y_overlap_ratio"] = overlap / max(target_stats["y_range"], 1.0)
                feature[f"{prefix}.{side_key}.x_mean_minus_target"] = side_stats[side_key]["x_mean"] - target_stats["x_mean"]

    if target_stats and "left_boundary" in side_stats and "right_boundary" in side_stats:
        left_stats = side_stats["left_boundary"]
        right_stats = side_stats["right_boundary"]
        left_dx = left_stats["x_mean"] - target_stats["x_mean"]
        right_dx = right_stats["x_mean"] - target_stats["x_mean"]
        feature[f"{prefix}.boundary_match.left_right_order_ok"] = int(left_stats["x_mean"] < right_stats["x_mean"])
        feature[f"{prefix}.boundary_match.target_between_means"] = int(left_dx <= 0 <= right_dx)
        feature[f"{prefix}.boundary_match.min_y_overlap_ratio"] = min(
            feature.get(f"{prefix}.left_boundary.target_y_overlap_ratio", 0.0),
            feature.get(f"{prefix}.right_boundary.target_y_overlap_ratio", 0.0),
        )
        feature[f"{prefix}.boundary_match.mean_y_overlap_ratio"] = (
            feature.get(f"{prefix}.left_boundary.target_y_overlap_ratio", 0.0)
            + feature.get(f"{prefix}.right_boundary.target_y_overlap_ratio", 0.0)
        ) / 2.0
        feature[f"{prefix}.boundary_match.width_mean"] = abs(right_stats["x_mean"] - left_stats["x_mean"])

    left_points = (packet.get("left_line_points") or {}).get("points_2d") or (packet.get("left_boundary") or {}).get("polyline_norm1000") or []
    right_points = (packet.get("right_line_points") or {}).get("points_2d") or (packet.get("right_boundary") or {}).get("polyline_norm1000") or []
    if left_points and right_points:
        left_mean = sum(float(point[0]) for point in left_points) / len(left_points)
        right_mean = sum(float(point[0]) for point in right_points) / len(right_points)
        feature[f"{prefix}.lane_band.mean_width"] = abs(right_mean - left_mean)

    for obj in packet.get("objects") or []:
        label_id = obj.get("label_id")
        feature[f"{prefix}.object.label.{label_id}"] = feature.get(f"{prefix}.object.label.{label_id}", 0) + 1
        bbox = obj.get("bbox_2d") or []
        if len(bbox) >= 4:
            try:
                x1, y1, x2, y2 = [float(value) for value in bbox[:4]]
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
                feature[f"{prefix}.object.{label_id}.count"] = feature.get(f"{prefix}.object.{label_id}.count", 0) + 1
                feature[f"{prefix}.object.{label_id}.min_y"] = min(cy, feature.get(f"{prefix}.object.{label_id}.min_y", 9999.0))
                feature[f"{prefix}.object.{label_id}.max_area"] = max(area, feature.get(f"{prefix}.object.{label_id}.max_area", 0.0))
                if cy < 450:
                    feature[f"{prefix}.object.{label_id}.upper_count"] = feature.get(f"{prefix}.object.{label_id}.upper_count", 0) + 1
                if 250 <= cy <= 700:
                    feature[f"{prefix}.object.{label_id}.midfield_count"] = feature.get(f"{prefix}.object.{label_id}.midfield_count", 0) + 1
                add_num(feature, f"{prefix}.object.{label_id}.last_cx", cx)
            except (TypeError, ValueError):
                pass
        geometry = obj.get("target_lane_geometry") or {}
        hint = geometry.get("relation_hint")
        feature[f"{prefix}.object.hint.{label_id}.{hint}"] = (
            feature.get(f"{prefix}.object.hint.{label_id}.{hint}", 0) + 1
        )
        if geometry.get("inside_target_lane_band"):
            feature[f"{prefix}.object.inside.{label_id}"] = feature.get(
                f"{prefix}.object.inside.{label_id}", 0
            ) + 1
        if geometry.get("distance_to_centerline") is not None:
            key = f"{prefix}.object.min_distance.{label_id}"
            feature[key] = min(float(geometry["distance_to_centerline"]), feature.get(key, 999.0))
        lane_relative_x = geometry.get("lane_relative_x")
        if lane_relative_x is not None:
            try:
                rel = float(lane_relative_x)
                abs_center = abs(rel - 0.5)
                feature[f"{prefix}.object.{label_id}.min_lane_center_offset"] = min(
                    abs_center,
                    feature.get(f"{prefix}.object.{label_id}.min_lane_center_offset", 999.0),
                )
                if 0.0 <= rel <= 1.0:
                    feature[f"{prefix}.object.in_lane_x.{label_id}"] = feature.get(f"{prefix}.object.in_lane_x.{label_id}", 0) + 1
                if 0.25 <= rel <= 0.75:
                    feature[f"{prefix}.object.centered_in_lane_x.{label_id}"] = (
                        feature.get(f"{prefix}.object.centered_in_lane_x.{label_id}", 0) + 1
                    )
            except (TypeError, ValueError):
                pass


def build_features(records_by_name: dict[str, dict[tuple[str, str], dict[str, Any]]], key: tuple[str, str]) -> dict[str, Any]:
    feature: dict[str, Any] = {}
    preds = []
    boundary_ids_by_name: dict[str, tuple[str | None, str | None]] = {}
    line_signal_by_name: dict[str, dict[str, Any]] = {}
    line_votes: dict[str, list[int]] = {
        "left_zigzag": [],
        "right_zigzag": [],
        "left_tidal_line": [],
        "right_tidal_line": [],
    }
    object_relation_votes: dict[str, list[str]] = {}
    object_semantic_votes: dict[str, list[str]] = {}
    for name, records in records_by_name.items():
        record = records[key]
        prediction = record.get("prediction") or {}
        pred_type = record.get("pred_type") or (record.get("prediction") or {}).get("special_lane_type") or "missing"
        preds.append(pred_type)
        add_record_features(feature, name, record)
        packet = record.get("request", {}).get("packet", {})
        left_boundary = packet.get("left_boundary") or {}
        right_boundary = packet.get("right_boundary") or {}
        boundary_ids_by_name[name] = (
            left_boundary.get("boundary_id"),
            right_boundary.get("boundary_id"),
        )
        attrs = prediction.get("line_attributes") or {}
        for side in ("left", "right"):
            item = attrs.get(side) or {}
            text = json.dumps(item, ensure_ascii=False).lower()
            line_votes[f"{side}_zigzag"].append(int("zigzag" in text or item.get("is_variable_zigzag_boundary") is True))
            line_votes[f"{side}_tidal_line"].append(
                int(
                    item.get("is_tidal_double_yellow_dashed_boundary") is True
                    or "double_yellow_dash" in text
                    or ("double" in text and "yellow" in text and "dash" in text)
                )
            )
        line_signal_by_name[name] = {
            "pred_type": pred_type,
            "zigzag_count": int(line_votes["left_zigzag"][-1]) + int(line_votes["right_zigzag"][-1]),
            "tidal_count": int(line_votes["left_tidal_line"][-1]) + int(line_votes["right_tidal_line"][-1]),
        }
        for relation in prediction.get("object_relations") or []:
            object_id = relation.get("object_id")
            if object_id:
                object_relation_votes.setdefault(str(object_id), []).append(str(relation.get("relation") or "missing").lower())
        for semantic in prediction.get("object_semantics") or []:
            object_id = semantic.get("object_id")
            if object_id:
                object_semantic_votes.setdefault(str(object_id), []).append(str(semantic.get("semantic_type") or "missing").lower())
    for label in CLASSES:
        feature[f"agreement.{label}"] = sum(pred == label for pred in preds)
    feature["agreement.num_unique"] = len(set(preds))
    feature["agreement.max_count"] = max((preds.count(label) for label in set(preds)), default=0)
    feature["agreement.max_ratio"] = feature["agreement.max_count"] / max(len(preds), 1)
    feature["conflict.prediction"] = int(len(set(preds)) > 1)

    if boundary_ids_by_name:
        reference_name = next(iter(boundary_ids_by_name))
        reference_left, reference_right = boundary_ids_by_name[reference_name]
        for name, (left_id, right_id) in boundary_ids_by_name.items():
            if name == reference_name:
                continue
            left_changed = left_id != reference_left
            right_changed = right_id != reference_right
            feature[f"boundary_agreement.{name}.left_changed"] = int(left_changed)
            feature[f"boundary_agreement.{name}.right_changed"] = int(right_changed)
            feature[f"boundary_agreement.{name}.any_changed"] = int(left_changed or right_changed)
            feature[f"boundary_agreement.{name}.both_changed"] = int(left_changed and right_changed)
            signal = line_signal_by_name.get(name) or {}
            any_changed = left_changed or right_changed
            pred_type = signal.get("pred_type")
            zigzag_count = int(signal.get("zigzag_count") or 0)
            tidal_count = int(signal.get("tidal_count") or 0)
            feature[f"boundary_semantic_check.{name}.changed_pred_variable_both_zigzag"] = int(
                any_changed and pred_type == "variable" and zigzag_count == 2
            )
            feature[f"boundary_semantic_check.{name}.changed_pred_variable_one_zigzag"] = int(
                any_changed and pred_type == "variable" and zigzag_count == 1
            )
            feature[f"boundary_semantic_check.{name}.changed_pred_variable_no_zigzag"] = int(
                any_changed and pred_type == "variable" and zigzag_count == 0
            )
            feature[f"boundary_semantic_check.{name}.changed_pred_tidal_both_tidal"] = int(
                any_changed and pred_type == "tidal" and tidal_count == 2
            )
            feature[f"boundary_semantic_check.{name}.changed_pred_tidal_no_tidal"] = int(
                any_changed and pred_type == "tidal" and tidal_count == 0
            )

    for vote_name, votes in line_votes.items():
        if votes:
            yes = sum(votes)
            feature[f"consistency.{vote_name}.yes_count"] = yes
            feature[f"consistency.{vote_name}.yes_ratio"] = yes / len(votes)
            feature[f"consistency.{vote_name}.conflict"] = int(0 < yes < len(votes))
    if line_votes["left_zigzag"] and line_votes["right_zigzag"]:
        both_zigzag_votes = [int(left and right) for left, right in zip(line_votes["left_zigzag"], line_votes["right_zigzag"])]
        feature["consistency.both_zigzag.yes_count"] = sum(both_zigzag_votes)
        feature["consistency.both_zigzag.yes_ratio"] = sum(both_zigzag_votes) / len(both_zigzag_votes)
        feature["consistency.both_zigzag.conflict"] = int(0 < sum(both_zigzag_votes) < len(both_zigzag_votes))
    if line_votes["left_tidal_line"] and line_votes["right_tidal_line"]:
        both_tidal_votes = [int(left and right) for left, right in zip(line_votes["left_tidal_line"], line_votes["right_tidal_line"])]
        feature["consistency.both_tidal_line.yes_count"] = sum(both_tidal_votes)
        feature["consistency.both_tidal_line.yes_ratio"] = sum(both_tidal_votes) / len(both_tidal_votes)
        feature["consistency.both_tidal_line.conflict"] = int(0 < sum(both_tidal_votes) < len(both_tidal_votes))

    relation_conflicts = 0
    applies_votes = 0
    for object_id, votes in object_relation_votes.items():
        unique = set(votes)
        if len(unique) > 1:
            relation_conflicts += 1
        applies = sum(vote == "applies_to_target_lane" for vote in votes)
        applies_votes += applies
        feature[f"consistency.object.{object_id}.relation_unique"] = len(unique)
        feature[f"consistency.object.{object_id}.applies_count"] = applies
        feature[f"consistency.object.{object_id}.applies_ratio"] = applies / len(votes)
    feature["conflict.object_relation_count"] = relation_conflicts
    feature["consistency.object_relation_applies_votes"] = applies_votes

    semantic_conflicts = 0
    for object_id, votes in object_semantic_votes.items():
        unique = set(votes)
        if len(unique) > 1:
            semantic_conflicts += 1
        feature[f"consistency.object.{object_id}.semantic_unique"] = len(unique)
    feature["conflict.object_semantic_count"] = semantic_conflicts
    feature["conflict.total_count"] = (
        feature["conflict.prediction"]
        + relation_conflicts
        + semantic_conflicts
        + sum(feature.get(f"consistency.{name}.conflict", 0) for name in line_votes)
    )
    return feature


def make_model(args: argparse.Namespace, random_state: int) -> Pipeline:
    classifier = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        random_state=random_state,
        class_weight=args.class_weight if args.class_weight != "none" else None,
        max_features="sqrt",
        n_jobs=args.n_jobs,
    )
    return Pipeline([("vectorizer", DictVectorizer(sparse=True)), ("classifier", classifier)])


def predict_with_probabilities(model: Pipeline, features: list[dict[str, Any]]) -> tuple[list[str], list[dict[str, float]]]:
    preds = list(model.predict(features))
    raw = model.predict_proba(features)
    classes = list(model.named_steps["classifier"].classes_)
    probabilities = [
        {label: float(row[classes.index(label)]) if label in classes else 0.0 for label in CLASSES}
        for row in raw
    ]
    return preds, probabilities


def update_record(base: dict[str, Any], pred_type: str, probabilities: dict[str, float], pipeline_name: str) -> dict[str, Any]:
    updated = json.loads(json.dumps(base, ensure_ascii=False))
    updated["pipeline"] = pipeline_name
    updated["pre_internal_decision_pred_type"] = base.get("pred_type")
    updated["internal_decision_probabilities"] = {key: round(value, 6) for key, value in probabilities.items()}
    updated["pred_type"] = pred_type
    updated["pred_one_hot"] = one_hot(pred_type)
    updated["correct"] = pred_type == updated.get("gt_type")
    prediction = dict(updated.get("prediction") or {})
    prediction["special_lane_type"] = pred_type
    prediction["one_hot"] = one_hot(pred_type)
    prediction["internal_decision_probabilities"] = updated["internal_decision_probabilities"]
    prediction["pre_internal_decision_special_lane_type"] = base.get("pred_type")
    updated["prediction"] = prediction
    return updated


def run(args: argparse.Namespace) -> None:
    inputs = dict(args.input)
    if args.primary not in inputs:
        raise ValueError(f"--primary must be one of: {sorted(inputs)}")
    records_by_name = {name: load_records(path) for name, path in inputs.items()}
    keys = sorted(records_by_name[args.primary])
    for name, records in records_by_name.items():
        missing = set(keys) - set(records)
        if missing:
            raise ValueError(f"{name} is missing {len(missing)} records")

    features = [build_features(records_by_name, key) for key in keys]
    labels = np.array([records_by_name[args.primary][key]["gt_type"] for key in keys])
    groups = np.array([key[0] for key in keys])

    if args.mode == "oof":
        preds: list[str | None] = [None] * len(keys)
        probabilities: list[dict[str, float] | None] = [None] * len(keys)
        fold_metrics = []
        for fold_id, (train_idx, test_idx) in enumerate(GroupKFold(n_splits=args.folds).split(features, labels, groups), start=1):
            model = make_model(args, args.random_state + fold_id)
            model.fit([features[idx] for idx in train_idx], labels[train_idx])
            fold_preds, fold_probs = predict_with_probabilities(model, [features[idx] for idx in test_idx])
            for idx, pred, prob in zip(test_idx, fold_preds, fold_probs):
                preds[idx] = pred
                probabilities[idx] = prob
            fold_metrics.append(
                {
                    "fold": fold_id,
                    "records": len(test_idx),
                    "accuracy": round(float(accuracy_score(labels[test_idx], fold_preds)), 6),
                }
            )
        metadata = {"mode": "oof", "folds": args.folds, "fold_metrics": fold_metrics}
    elif args.mode == "loo":
        preds = [None] * len(keys)
        probabilities = [None] * len(keys)
        for split_id, (train_idx, test_idx) in enumerate(LeaveOneOut().split(features, labels), start=1):
            model = make_model(args, args.random_state + split_id)
            model.fit([features[idx] for idx in train_idx], labels[train_idx])
            fold_preds, fold_probs = predict_with_probabilities(model, [features[idx] for idx in test_idx])
            idx = int(test_idx[0])
            preds[idx] = fold_preds[0]
            probabilities[idx] = fold_probs[0]
        metadata = {"mode": "loo", "folds": len(keys), "train_records_per_split": len(keys) - 1}
    elif args.mode == "fit-all":
        model = make_model(args, args.random_state)
        model.fit(features, labels)
        preds, probabilities = predict_with_probabilities(model, features)
        if args.model_output:
            Path(args.model_output).parent.mkdir(parents=True, exist_ok=True)
            with Path(args.model_output).open("wb") as fp:
                pickle.dump(model, fp)
        metadata = {"mode": "fit-all", "model_output": args.model_output}
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")

    output_records = [
        update_record(records_by_name[args.primary][key], str(pred), dict(prob), args.pipeline_name)
        for key, pred, prob in zip(keys, preds, probabilities)
    ]
    payload = {
        "pipeline": args.pipeline_name,
        "mode": args.mode,
        "primary_input": args.primary,
        "input_files": {name: str(path) for name, path in inputs.items()},
        "calibration_metadata": metadata,
        "num_records": len(output_records),
        "pred_distribution": dict(Counter(record["pred_type"] for record in output_records)),
        "records": output_records,
    }
    dump_json(Path(args.output), payload)
    print(
        json.dumps(
            {
                "records": len(output_records),
                "correct": sum(record["correct"] for record in output_records),
                "accuracy": round(sum(record["correct"] for record in output_records) / len(output_records), 6),
                "pred_distribution": payload["pred_distribution"],
            },
            ensure_ascii=False,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=parse_named_input, required=True, help="name=path; repeatable")
    parser.add_argument("--primary", required=True, help="Input name used as base output record")
    parser.add_argument("--pipeline-name", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=("oof", "loo", "fit-all"), default="oof")
    parser.add_argument("--model-output", default=None)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=21)
    parser.add_argument("--n-estimators", type=int, default=400)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--min-samples-leaf", type=int, default=1)
    parser.add_argument("--class-weight", choices=("balanced", "balanced_subsample", "none"), default="balanced_subsample")
    parser.add_argument("--n-jobs", type=int, default=-1)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
