"""Autoresearch search loop for pipeline_alpha internal decision improvements.

This script runs cheap, no-VLM experiments over existing alpha result files.
It logs every tried configuration, tracks kept improvements, and writes a
progress plot similar to an autoresearch dashboard.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupKFold, LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.calibrate_pipeline_internal import build_features, load_records, parse_named_input  # noqa: E402
from special_lane_common.data import read_json_or_jsonl  # noqa: E402
from special_lane_common.taxonomy import SPECIAL_LANE_CLASSES  # noqa: E402


CLASSES = list(SPECIAL_LANE_CLASSES)
OBJECT_PRIORS = {
    0: "bus",
    1: "bus",
    2: "bus",
    4: "variable",
    5: "variable",
    9: "bicycle",
    10: "bicycle",
}


def clean_feature_dict(feature: dict[str, Any]) -> dict[str, Any]:
    """Remove feature families from discarded experiments.

    The historical current-best feature set came before visible_text keyword
    buckets, line-confidence/support aggregates, and scalar object-alignment
    bbox summaries were added. A `+clean` variant lets us compare against that
    cleaner feature surface without changing the lower-level packet parser.
    """

    cleaned = {}
    object_bbox_pattern = re.compile(
        r"\.object\.[^.]+\.(count|min_y|max_area|upper_count|midfield_count|last_cx|min_lane_center_offset)$"
    )
    for key, value in feature.items():
        if ".visible_text." in key:
            continue
        if ".supporting_line_sides" in key:
            continue
        if key.endswith(".line_confidence"):
            continue
        if ".zigzag.confidence_" in key or key.endswith(".zigzag.support_count"):
            continue
        if ".tidal_line.confidence_" in key or key.endswith(".tidal_line.support_count"):
            continue
        if ".object.in_lane_x." in key or ".object.centered_in_lane_x." in key:
            continue
        if object_bbox_pattern.search(key):
            continue
        cleaned[key] = value
    return cleaned


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as fp:
        return json.load(fp)


def normalize_label(value: Any) -> str:
    mapping = {
        "公交": "bus",
        "潮汐": "tidal",
        "可变": "variable",
        "自行": "bicycle",
        "自行车": "bicycle",
        "普通": "normal",
        "正常": "normal",
    }
    value = str(value)
    return mapping.get(value, value)


def lane_meta_features(data_dir: Path, key: tuple[str, str]) -> dict[str, Any]:
    image_id, lane_id = key
    path = data_dir / "jsons" / f"{image_id}.json"
    if not path.exists():
        return {}
    data = load_json(path)
    lanes = data.get("lanes") or []
    idx = int(str(lane_id)[1:]) if str(lane_id).startswith("L") and str(lane_id)[1:].isdigit() else None
    feature: dict[str, Any] = {"meta.lane_count": len(lanes)}
    centers = []
    for lane in lanes:
        points = lane.get("grounding") or []
        if points:
            centers.append(sum(point[0] for point in points) / len(points))
        else:
            centers.append(float("nan"))
    if idx is not None and idx < len(lanes) and not math.isnan(centers[idx]):
        rank = sorted(range(len(centers)), key=lambda item: centers[item] if not math.isnan(centers[item]) else 1e9).index(idx)
        feature["meta.target_lane_rank_from_left"] = rank
        feature["meta.target_lane_rank_norm"] = rank / max(len(lanes) - 1, 1)
        feature["meta.target_lane_x_mean_px"] = centers[idx]
        points = lanes[idx].get("grounding") or []
        if len(points) >= 2:
            dx = points[-1][0] - points[0][0]
            dy = points[-1][1] - points[0][1]
            feature["meta.target_lane_heading_rad"] = math.atan2(dy, dx)
            feature["meta.target_lane_abs_dx"] = abs(dx)
            feature["meta.target_lane_abs_dy"] = abs(dy)
    return feature


def _as_points(points: Any) -> list[tuple[float, float]]:
    parsed = []
    for point in points or []:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        try:
            parsed.append((float(point[0]), float(point[1])))
        except (TypeError, ValueError):
            continue
    return parsed


def _polyline_stats(prefix: str, points: list[tuple[float, float]]) -> dict[str, Any]:
    feature: dict[str, Any] = {f"{prefix}.point_count": len(points)}
    if not points:
        return feature
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    feature[f"{prefix}.x_range"] = max(xs) - min(xs)
    feature[f"{prefix}.y_range"] = max(ys) - min(ys)
    feature[f"{prefix}.x_std"] = float(np.std(xs)) if len(xs) > 1 else 0.0
    feature[f"{prefix}.y_std"] = float(np.std(ys)) if len(ys) > 1 else 0.0
    if len(points) < 2:
        return feature

    segments = []
    headings = []
    dxs = []
    dys = []
    for (x1, y1), (x2, y2) in zip(points, points[1:]):
        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy)
        if length <= 1e-6:
            continue
        segments.append(length)
        headings.append(math.atan2(dy, dx))
        dxs.append(dx)
        dys.append(dy)

    total_length = sum(segments)
    chord = math.hypot(points[-1][0] - points[0][0], points[-1][1] - points[0][1])
    feature[f"{prefix}.length"] = total_length
    feature[f"{prefix}.chord"] = chord
    feature[f"{prefix}.tortuosity"] = total_length / max(chord, 1.0)
    feature[f"{prefix}.endpoint_dx"] = points[-1][0] - points[0][0]
    feature[f"{prefix}.endpoint_dy"] = points[-1][1] - points[0][1]
    feature[f"{prefix}.abs_endpoint_dx"] = abs(points[-1][0] - points[0][0])
    feature[f"{prefix}.abs_endpoint_dy"] = abs(points[-1][1] - points[0][1])
    if headings:
        feature[f"{prefix}.heading_mean"] = float(np.mean(headings))
        feature[f"{prefix}.heading_std"] = float(np.std(headings)) if len(headings) > 1 else 0.0

    turns = []
    for prev, cur in zip(headings, headings[1:]):
        delta = (cur - prev + math.pi) % (2 * math.pi) - math.pi
        turns.append(abs(delta))
    if turns:
        feature[f"{prefix}.turn_abs_sum"] = sum(turns)
        feature[f"{prefix}.turn_abs_max"] = max(turns)
        feature[f"{prefix}.turn_abs_mean"] = sum(turns) / len(turns)
        feature[f"{prefix}.sharp_turn_count"] = sum(turn > 0.45 for turn in turns)

    def sign_changes(values: list[float]) -> int:
        signs = [1 if value > 1e-6 else -1 if value < -1e-6 else 0 for value in values]
        signs = [sign for sign in signs if sign]
        return sum(1 for left, right in zip(signs, signs[1:]) if left != right)

    feature[f"{prefix}.dx_sign_changes"] = sign_changes(dxs)
    feature[f"{prefix}.dy_sign_changes"] = sign_changes(dys)

    y_sorted = sorted(points, key=lambda item: item[1])
    y_dxs = [x2 - x1 for (x1, _), (x2, _) in zip(y_sorted, y_sorted[1:])]
    feature[f"{prefix}.x_sign_changes_by_y"] = sign_changes(y_dxs)
    feature[f"{prefix}.zigzag_geometry_score"] = (
        feature.get(f"{prefix}.x_sign_changes_by_y", 0)
        + 0.4 * feature.get(f"{prefix}.sharp_turn_count", 0)
        + 0.2 * min(feature.get(f"{prefix}.turn_abs_sum", 0.0), 10.0)
    )
    return feature


def _x_at_y(points: list[tuple[float, float]], y_value: float) -> float | None:
    xs = []
    for (x1, y1), (x2, y2) in zip(points, points[1:]):
        if abs(y2 - y1) < 1e-6:
            continue
        lo, hi = sorted((y1, y2))
        if lo <= y_value <= hi:
            ratio = (y_value - y1) / (y2 - y1)
            xs.append(x1 + ratio * (x2 - x1))
    if not xs:
        nearby = sorted(points, key=lambda point: abs(point[1] - y_value))[:2]
        if nearby:
            xs = [point[0] for point in nearby]
    if not xs:
        return None
    return sum(xs) / len(xs)


def boundary_geometry_features(records_by_name: dict[str, dict[tuple[str, str], dict[str, Any]]], key: tuple[str, str]) -> dict[str, Any]:
    """Production-legal shape features from the alpha packet geometry.

    These do not read labels or cue annotations. They summarize whether the two
    detected boundaries look straight, parallel, jagged, or width-stable.
    """

    feature: dict[str, Any] = {}
    for name, records in records_by_name.items():
        packet = records[key].get("request", {}).get("packet", {})
        target = packet.get("target_lane") or {}
        left = packet.get("left_boundary") or packet.get("left_line_points") or {}
        right = packet.get("right_boundary") or packet.get("right_line_points") or {}
        target_points = _as_points(target.get("centerline_norm1000") or target.get("centerline_points_hint"))
        left_points = _as_points(left.get("polyline_norm1000") or left.get("points_2d"))
        right_points = _as_points(right.get("polyline_norm1000") or right.get("points_2d"))

        prefix = f"geom.{name}"
        for side, points in (("target", target_points), ("left", left_points), ("right", right_points)):
            feature.update(_polyline_stats(f"{prefix}.{side}", points))

        if left_points and right_points:
            left_x_mean = sum(point[0] for point in left_points) / len(left_points)
            right_x_mean = sum(point[0] for point in right_points) / len(right_points)
            feature[f"{prefix}.band.mean_width"] = abs(right_x_mean - left_x_mean)
            feature[f"{prefix}.band.left_right_order_ok"] = int(left_x_mean < right_x_mean)
            left_heading = feature.get(f"{prefix}.left.heading_mean")
            right_heading = feature.get(f"{prefix}.right.heading_mean")
            if left_heading is not None and right_heading is not None:
                delta = (float(left_heading) - float(right_heading) + math.pi) % (2 * math.pi) - math.pi
                feature[f"{prefix}.band.heading_abs_delta"] = abs(delta)
            if target_points:
                target_x_mean = sum(point[0] for point in target_points) / len(target_points)
                center_x = (left_x_mean + right_x_mean) / 2
                width = abs(right_x_mean - left_x_mean)
                feature[f"{prefix}.band.target_center_offset_ratio"] = abs(target_x_mean - center_x) / max(width, 1.0)

            widths = []
            for y_value in (600.0, 700.0, 800.0, 900.0):
                left_x = _x_at_y(left_points, y_value)
                right_x = _x_at_y(right_points, y_value)
                if left_x is None or right_x is None:
                    continue
                width = abs(right_x - left_x)
                widths.append(width)
                feature[f"{prefix}.band.width_y{int(y_value)}"] = width
            if widths:
                feature[f"{prefix}.band.width_mean"] = sum(widths) / len(widths)
                feature[f"{prefix}.band.width_std"] = float(np.std(widths)) if len(widths) > 1 else 0.0
                feature[f"{prefix}.band.width_min"] = min(widths)
                feature[f"{prefix}.band.width_max"] = max(widths)
                feature[f"{prefix}.band.width_range"] = max(widths) - min(widths)
        if left_points and right_points:
            left_zig = feature.get(f"{prefix}.left.zigzag_geometry_score", 0.0)
            right_zig = feature.get(f"{prefix}.right.zigzag_geometry_score", 0.0)
            feature[f"{prefix}.band.both_zigzag_geometry_score"] = min(float(left_zig), float(right_zig))
            feature[f"{prefix}.band.zigzag_geometry_sum"] = float(left_zig) + float(right_zig)
    return feature


def domain_rule_features(records_by_name: dict[str, dict[tuple[str, str], dict[str, Any]]], key: tuple[str, str]) -> dict[str, Any]:
    feature: dict[str, Any] = {f"rule.score.{label}": 0.0 for label in CLASSES}
    pred_votes = Counter()
    any_relation_apply = False
    any_inside_special = False
    min_dist_by_class: dict[str, float] = defaultdict(lambda: 999.0)
    for name, records in records_by_name.items():
        record = records[key]
        prediction = record.get("prediction") or {}
        pred = record.get("pred_type") or prediction.get("special_lane_type")
        if pred:
            pred_votes[pred] += 1
            feature[f"rule.vote.pred.{pred}"] = feature.get(f"rule.vote.pred.{pred}", 0) + 1
            feature[f"rule.score.{pred}"] += 1.0

        attrs = prediction.get("line_attributes") or {}
        left_text = json.dumps(attrs.get("left") or {}, ensure_ascii=False).lower()
        right_text = json.dumps(attrs.get("right") or {}, ensure_ascii=False).lower()
        left_zigzag = "zigzag" in left_text
        right_zigzag = "zigzag" in right_text
        left_tidal = "double" in left_text and "yellow" in left_text and "dash" in left_text
        right_tidal = "double" in right_text and "yellow" in right_text and "dash" in right_text
        if left_zigzag and right_zigzag:
            feature["rule.both_zigzag"] = feature.get("rule.both_zigzag", 0) + 1
            feature["rule.score.variable"] += 2.0
        elif left_zigzag or right_zigzag:
            feature["rule.one_side_zigzag"] = feature.get("rule.one_side_zigzag", 0) + 1
            feature["rule.score.variable"] += 0.5
        if left_tidal and right_tidal:
            feature["rule.both_tidal"] = feature.get("rule.both_tidal", 0) + 1
            feature["rule.score.tidal"] += 2.0
        elif left_tidal or right_tidal:
            feature["rule.one_side_tidal"] = feature.get("rule.one_side_tidal", 0) + 1
            feature["rule.score.tidal"] += 0.5

        relation_by_id = {
            str(item.get("object_id")): item
            for item in prediction.get("object_relations") or []
            if item.get("object_id") is not None
        }
        packet = record.get("request", {}).get("packet", {})
        for obj in packet.get("objects") or []:
            label_id = obj.get("label_id")
            prior = OBJECT_PRIORS.get(label_id)
            if prior is None:
                continue
            geometry = obj.get("target_lane_geometry") or {}
            relation = str((relation_by_id.get(str(obj.get("object_id"))) or {}).get("relation") or "").lower()
            inside = geometry.get("inside_target_lane_band") is True
            hint = str(geometry.get("relation_hint") or "").lower()
            dist = geometry.get("distance_to_centerline")
            if dist is not None:
                try:
                    min_dist_by_class[prior] = min(min_dist_by_class[prior], float(dist))
                except (TypeError, ValueError):
                    pass
            if relation == "applies_to_target_lane":
                any_relation_apply = True
                feature[f"rule.apply_object.{prior}"] = feature.get(f"rule.apply_object.{prior}", 0) + 1
                feature[f"rule.score.{prior}"] += 2.2
            elif inside or hint == "inside_target_lane_band":
                any_inside_special = True
                feature[f"rule.inside_object.{prior}"] = feature.get(f"rule.inside_object.{prior}", 0) + 1
                feature[f"rule.score.{prior}"] += 1.3
            else:
                feature[f"rule.near_object.{prior}"] = feature.get(f"rule.near_object.{prior}", 0) + 1
                feature[f"rule.score.{prior}"] += 0.15
    for label, dist in min_dist_by_class.items():
        feature[f"rule.min_object_dist.{label}"] = dist
    feature["rule.any_relation_apply"] = int(any_relation_apply)
    feature["rule.any_inside_special"] = int(any_inside_special)
    feature["rule.pred_vote_unique"] = len(pred_votes)
    feature["rule.pred_vote_max"] = max(pred_votes.values(), default=0)
    feature["rule.score.normal"] += 0.4 if not any_relation_apply and not any_inside_special else 0.0
    best_score_label, best_score = max(((label, feature[f"rule.score.{label}"]) for label in CLASSES), key=lambda item: item[1])
    feature[f"rule.best_score_label.{best_score_label}"] = 1
    feature["rule.best_score"] = best_score
    return feature


def build_feature_set(
    records_by_name: dict[str, dict[tuple[str, str], dict[str, Any]]],
    keys: list[tuple[str, str]],
    data_dir: Path,
    variant: str,
) -> list[dict[str, Any]]:
    features = []
    for key in keys:
        base = build_features(records_by_name, key)
        if "clean" in variant:
            base = clean_feature_dict(base)
        if "domain" in variant:
            base.update(domain_rule_features(records_by_name, key))
        if "meta" in variant:
            base.update(lane_meta_features(data_dir, key))
        if "geometry" in variant:
            base.update(boundary_geometry_features(records_by_name, key))
        features.append(base)
    return features


def make_model(config: dict[str, Any], seed: int) -> Pipeline:
    kind = config["model"]
    if kind == "rf":
        clf = RandomForestClassifier(
            n_estimators=config.get("n_estimators", 400),
            max_depth=config.get("max_depth"),
            min_samples_leaf=config.get("min_samples_leaf", 1),
            class_weight=config.get("class_weight"),
            max_features=config.get("max_features", "sqrt"),
            random_state=seed,
            n_jobs=config.get("n_jobs", -1),
        )
    elif kind == "extra_trees":
        clf = ExtraTreesClassifier(
            n_estimators=config.get("n_estimators", 400),
            max_depth=config.get("max_depth"),
            min_samples_leaf=config.get("min_samples_leaf", 1),
            class_weight=config.get("class_weight"),
            max_features=config.get("max_features", "sqrt"),
            random_state=seed,
            n_jobs=config.get("n_jobs", -1),
        )
    elif kind == "gb":
        clf = GradientBoostingClassifier(
            n_estimators=config.get("n_estimators", 120),
            learning_rate=config.get("learning_rate", 0.05),
            max_depth=config.get("max_depth", 2),
            random_state=seed,
        )
    elif kind == "ada":
        clf = AdaBoostClassifier(
            n_estimators=config.get("n_estimators", 120),
            learning_rate=config.get("learning_rate", 0.5),
            random_state=seed,
        )
    elif kind == "logreg":
        clf = LogisticRegression(
            C=config.get("C", 1.0),
            max_iter=3000,
            class_weight=config.get("class_weight"),
            solver="saga",
            n_jobs=config.get("n_jobs", -1),
            random_state=seed,
        )
    elif kind == "linear_svc":
        clf = LinearSVC(
            C=config.get("C", 1.0),
            class_weight=config.get("class_weight"),
            random_state=seed,
            max_iter=5000,
        )
    else:
        raise ValueError(f"unknown model: {kind}")
    return Pipeline([("vectorizer", DictVectorizer(sparse=True)), ("classifier", clf)])


def predict_labels(model: Pipeline, features: list[dict[str, Any]]) -> list[str]:
    return [str(item) for item in model.predict(features)]


def evaluate_config(
    config: dict[str, Any],
    records_by_name: dict[str, dict[tuple[str, str], dict[str, Any]]],
    primary_records: dict[tuple[str, str], dict[str, Any]],
    keys: list[tuple[str, str]],
    data_dir: Path,
    mode: str,
    folds: int,
    seed: int,
) -> dict[str, Any]:
    features = build_feature_set(records_by_name, keys, data_dir, config["feature_variant"])
    labels = np.array([primary_records[key]["gt_type"] for key in keys])
    groups = np.array([key[0] for key in keys])
    splitter = (
        LeaveOneOut().split(features, labels)
        if mode == "loo"
        else GroupKFold(n_splits=folds).split(features, labels, groups)
    )
    preds: list[str | None] = [None] * len(keys)
    fold_metrics = []
    for split_id, (train_idx, test_idx) in enumerate(splitter, start=1):
        model = make_model(config, seed + split_id)
        model.fit([features[idx] for idx in train_idx], labels[train_idx])
        split_preds = predict_labels(model, [features[idx] for idx in test_idx])
        for idx, pred in zip(test_idx, split_preds):
            preds[int(idx)] = pred
        if mode == "oof":
            fold_metrics.append(
                {
                    "fold": split_id,
                    "records": len(test_idx),
                    "accuracy": round(float(accuracy_score(labels[test_idx], split_preds)), 6),
                }
            )
    correct = int(sum(pred == label for pred, label in zip(preds, labels)))
    confusion = {label: Counter() for label in CLASSES}
    for pred, label in zip(preds, labels):
        confusion[str(label)][str(pred)] += 1
    return {
        "accuracy": round(correct / len(labels), 6),
        "correct": correct,
        "total": int(len(labels)),
        "pred_distribution": dict(Counter(preds)),
        "fold_metrics": fold_metrics,
        "confusion_matrix": {label: dict(row) for label, row in confusion.items()},
    }


def config_grid(input_names: list[str], n_jobs: int) -> list[dict[str, Any]]:
    subsets: list[tuple[str, ...]] = []
    for size in range(1, len(input_names) + 1):
        subsets.extend(itertools.combinations(input_names, size))
    preferred = [
        ("alpha_raw", "alpha_fused", "alpha_geom"),
        ("alpha_raw", "alpha_geom"),
        ("alpha_fused", "alpha_geom"),
    ]
    ordered_subsets = []
    for subset in preferred + subsets:
        if subset and all(name in input_names for name in subset) and subset not in ordered_subsets:
            ordered_subsets.append(subset)

    core_variants = ["base", "base+domain", "base+meta", "base+domain+meta"]
    feature_variants = core_variants + [f"{variant}+clean" for variant in core_variants]
    model_configs = [
        {"model": "rf", "n_estimators": 400, "max_depth": 12, "min_samples_leaf": 1, "class_weight": "balanced_subsample", "n_jobs": n_jobs},
        {"model": "rf", "n_estimators": 500, "max_depth": 20, "min_samples_leaf": 1, "class_weight": "balanced", "n_jobs": n_jobs},
        {"model": "rf", "n_estimators": 500, "max_depth": None, "min_samples_leaf": 2, "class_weight": "balanced_subsample", "n_jobs": n_jobs},
        {"model": "extra_trees", "n_estimators": 500, "max_depth": None, "min_samples_leaf": 1, "class_weight": "balanced", "n_jobs": n_jobs},
        {"model": "extra_trees", "n_estimators": 500, "max_depth": 20, "min_samples_leaf": 2, "class_weight": "balanced_subsample", "n_jobs": n_jobs},
        {"model": "gb", "n_estimators": 120, "learning_rate": 0.04, "max_depth": 2},
        {"model": "gb", "n_estimators": 180, "learning_rate": 0.03, "max_depth": 2},
        {"model": "ada", "n_estimators": 120, "learning_rate": 0.5},
        {"model": "logreg", "C": 0.2, "class_weight": "balanced", "n_jobs": n_jobs},
        {"model": "linear_svc", "C": 0.1, "class_weight": "balanced"},
    ]
    configs = []
    for subset in ordered_subsets:
        for variant in feature_variants:
            for model_config in model_configs:
                item = dict(model_config)
                item["subset"] = list(subset)
                item["feature_variant"] = variant
                configs.append(item)
    return configs


def plot_progress(rows: list[dict[str, Any]], target: float, output: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    kept_x = []
    kept_y = []
    best = None
    running = []
    for row in rows:
        if row["kept"]:
            best = row["accuracy"] if best is None else max(best, row["accuracy"])
        running.append(best if best is not None else row["accuracy"])
        if row["kept"]:
            kept_x.append(row["experiment_id"])
            kept_y.append(row["accuracy"])
    xs = [row["experiment_id"] for row in rows]
    ys = [row["accuracy"] for row in rows]
    plt.figure(figsize=(16, 6))
    plt.scatter(xs, ys, color="#c8c8c8", alpha=0.55, label="Discarded")
    if kept_x:
        plt.scatter(kept_x, kept_y, color="#67b87a", edgecolor="#2f7042", s=80, label="Kept", zorder=3)
    plt.step(xs, running, where="post", color="#7fb88a", linewidth=2, label="Running best")
    plt.axhline(target, color="#b77c7c", linestyle="--", linewidth=1.5, label=f"Target {target:.2f}")
    plt.title(
        f"Pipeline Alpha Autoresearch Progress: {len(rows)} Experiments, "
        f"{sum(row['kept'] for row in rows)} Kept Improvements, Best Kept {max(kept_y) if kept_y else max(ys):.6f}"
    )
    plt.xlabel("Experiment #")
    plt.ylabel("Accuracy (higher is better)")
    plt.ylim(max(0.70, min(ys) - 0.02), max(target + 0.02, max(ys) + 0.03))
    plt.grid(True, alpha=0.22)
    plt.legend(loc="best")
    for row in rows:
        if row["kept"]:
            plt.annotate(row["name"][:28], (row["experiment_id"], row["accuracy"]), xytext=(5, 9), textcoords="offset points", rotation=28, fontsize=8, color="#3d7b4d")
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()


def write_markdown(rows: list[dict[str, Any]], target: float, output: Path) -> None:
    kept = [row for row in rows if row["kept"]]
    best = max(kept or rows, key=lambda item: item["accuracy"])
    lines = [
        "# Pipeline Alpha Autoresearch Report",
        "",
        f"- Target accuracy: `{target:.3f}`",
        f"- Experiments completed: `{len(rows)}`",
        f"- Kept improvements: `{len(kept)}`",
        f"- Best valid accuracy: `{best['accuracy']:.6f}` ({best['correct']}/{best['total']})",
        f"- Best experiment: `{best['name']}`",
        "",
        "## Kept Improvements",
        "",
        "| Exp | Accuracy | Correct | Name | Feature Variant | Model | Inputs |",
        "| --- | ---: | ---: | --- | --- | --- | --- |",
    ]
    for row in kept:
        lines.append(
            f"| {row['experiment_id']} | {row['accuracy']:.6f} | {row['correct']}/{row['total']} | "
            f"{row['name']} | {row['feature_variant']} | {row['model']} | {'+'.join(row['subset'])} |"
        )
    lines.extend(["", "## Top 20 Experiments", "", "| Rank | Exp | Accuracy | Correct | Name | Kept |", "| ---: | ---: | ---: | ---: | --- | --- |"])
    for rank, row in enumerate(sorted(rows, key=lambda item: item["accuracy"], reverse=True)[:20], start=1):
        lines.append(f"| {rank} | {row['experiment_id']} | {row['accuracy']:.6f} | {row['correct']}/{row['total']} | {row['name']} | {row['kept']} |")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    inputs = dict(args.input)
    if args.primary not in inputs:
        raise ValueError(f"--primary must be one of {sorted(inputs)}")
    input_names = list(inputs)
    all_configs = config_grid(input_names, args.n_jobs)
    if args.model_kind:
        allowed_models = set(args.model_kind)
        all_configs = [config for config in all_configs if config["model"] in allowed_models]
    if args.feature_variant:
        allowed_variants = set(args.feature_variant)
        all_configs = [config for config in all_configs if config["feature_variant"] in allowed_variants]
    if args.subset:
        allowed_subsets = {tuple(item.split("+")) for item in args.subset}
        all_configs = [config for config in all_configs if tuple(config["subset"]) in allowed_subsets]
    if args.max_experiments:
        all_configs = all_configs[: args.max_experiments]

    rows = []
    best = -1.0
    for idx, config in enumerate(all_configs, start=1):
        records_by_name = {name: load_records(inputs[name]) for name in config["subset"]}
        primary = args.primary if args.primary in config["subset"] else config["subset"][0]
        primary_records = records_by_name[primary]
        keys = sorted(primary_records)
        result = evaluate_config(
            config,
            records_by_name,
            primary_records,
            keys,
            Path(args.data_dir),
            args.mode,
            args.folds,
            args.random_state,
        )
        name = (
            f"{config['model']} {config['feature_variant']} "
            f"{'+'.join(config['subset'])}"
        )
        kept = result["accuracy"] > best
        best = max(best, result["accuracy"])
        row = {
            "experiment_id": idx,
            "name": name,
            "mode": args.mode,
            "kept": kept,
            "running_best": round(best, 6),
            "feature_variant": config["feature_variant"],
            "model": config["model"],
            "subset": config["subset"],
            **result,
            "config": config,
        }
        rows.append(row)
        print(json.dumps({k: row[k] for k in ("experiment_id", "accuracy", "correct", "kept", "running_best", "name")}, ensure_ascii=False), flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": args.experiment_name,
        "target": args.target,
        "mode": args.mode,
        "primary": args.primary,
        "input_files": {name: str(path) for name, path in inputs.items()},
        "rows": rows,
    }
    (args.output_dir / f"{args.experiment_name}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with (args.output_dir / f"{args.experiment_name}.csv").open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "experiment_id",
                "accuracy",
                "correct",
                "total",
                "kept",
                "running_best",
                "name",
                "feature_variant",
                "model",
                "subset",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "experiment_id": row["experiment_id"],
                    "accuracy": row["accuracy"],
                    "correct": row["correct"],
                    "total": row["total"],
                    "kept": row["kept"],
                    "running_best": row["running_best"],
                    "name": row["name"],
                    "feature_variant": row["feature_variant"],
                    "model": row["model"],
                    "subset": "+".join(row["subset"]),
                }
            )
    plot_progress(rows, args.target, args.output_dir / f"{args.experiment_name}_progress.png")
    write_markdown(rows, args.target, args.output_dir / f"{args.experiment_name}_report.md")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=parse_named_input, required=True)
    parser.add_argument("--primary", required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--experiment-name", default="alpha_autoresearch")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/alpha_autoresearch"))
    parser.add_argument("--mode", choices=("oof", "loo"), default="oof")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--target", type=float, default=0.90)
    parser.add_argument("--random-state", type=int, default=13)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--max-experiments", type=int, default=None)
    parser.add_argument("--model-kind", action="append", choices=("rf", "extra_trees", "gb", "ada", "logreg", "linear_svc"), help="Filter grid by model kind; repeatable")
    parser.add_argument("--feature-variant", action="append", help="Filter grid by feature variant; repeatable")
    parser.add_argument("--subset", action="append", help="Filter grid by exact input subset, e.g. alpha_raw+alpha_geom")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
