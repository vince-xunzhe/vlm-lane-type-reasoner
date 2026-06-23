"""Apply the Alpha Clean85 production decision head.

This is the production counterpart of the Clean85 OOF experiment: it loads one
fixed sklearn model and applies it to three precomputed alpha evidence files.
It does not train folds and does not run any benchmark.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
RELEASE_DIR = ROOT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.alpha_autoresearch_search import build_feature_set  # noqa: E402
from scripts.calibrate_pipeline_internal import load_records, parse_named_input  # noqa: E402
from special_lane_common.data import dump_json  # noqa: E402
from special_lane_common.taxonomy import SPECIAL_LANE_CLASSES, one_hot  # noqa: E402


CLASSES = list(SPECIAL_LANE_CLASSES)


def predict_with_probabilities(model: Any, features: list[dict[str, Any]]) -> tuple[list[str], list[dict[str, float]]]:
    preds = [str(item) for item in model.predict(features)]
    if not hasattr(model, "predict_proba"):
        return preds, [{label: float(label == pred) for label in CLASSES} for pred in preds]

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
    updated["internal_decision_probabilities"] = {
        key: round(float(value), 6)
        for key, value in probabilities.items()
    }
    updated["pred_type"] = pred_type
    updated["pred_one_hot"] = one_hot(pred_type)

    prediction = dict(updated.get("prediction") or {})
    prediction["special_lane_type"] = pred_type
    prediction["one_hot"] = one_hot(pred_type)
    prediction["internal_decision_probabilities"] = updated["internal_decision_probabilities"]
    prediction["pre_internal_decision_special_lane_type"] = base.get("pred_type")
    updated["prediction"] = prediction

    for key in ("gt_type", "gt_one_hot", "correct"):
        updated.pop(key, None)
    return updated


def run(args: argparse.Namespace) -> None:
    inputs = dict(args.input)
    if args.primary not in inputs:
        raise ValueError(f"--primary must be one of {sorted(inputs)}")

    records_by_name = {name: load_records(path) for name, path in inputs.items()}
    keys = sorted(records_by_name[args.primary])
    for name, records in records_by_name.items():
        missing = set(keys) - set(records)
        if missing:
            raise ValueError(f"{name} is missing {len(missing)} records")

    data_dir = Path(args.data_dir) if args.data_dir else RELEASE_DIR / "data" / "input"
    features = build_feature_set(records_by_name, keys, data_dir, args.feature_variant)
    with Path(args.model).open("rb") as fp:
        model = pickle.load(fp)

    preds, probabilities = predict_with_probabilities(model, features)
    output_records = [
        update_record(records_by_name[args.primary][key], pred, prob, args.pipeline_name)
        for key, pred, prob in zip(keys, preds, probabilities)
    ]
    payload = {
        "pipeline": args.pipeline_name,
        "mode": "apply-clean85-model",
        "primary_input": args.primary,
        "input_files": {name: str(path) for name, path in inputs.items()},
        "model_file": str(args.model),
        "feature_variant": args.feature_variant,
        "num_records": len(output_records),
        "pred_distribution": dict(Counter(record["pred_type"] for record in output_records)),
        "records": output_records,
    }
    dump_json(Path(args.output), payload)
    print(json.dumps({"records": len(output_records), "output": args.output}, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=parse_named_input, required=True, help="name=path; repeatable")
    parser.add_argument("--primary", default="alpha_raw")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--model", required=True)
    parser.add_argument("--pipeline-name", default="pipeline_alpha_clean85_prod")
    parser.add_argument("--feature-variant", default="base+meta+clean")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
