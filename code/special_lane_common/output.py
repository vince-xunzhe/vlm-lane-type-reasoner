"""Unified output schema helpers for alpha/beta pipelines."""

from __future__ import annotations

from .taxonomy import SPECIAL_LANE_CLASSES, canonical_lane_type, one_hot


SCHEMA_VERSION = "special_lane_vlm_result/v1"


def normalize_prediction(payload: dict | None) -> dict | None:
    if not payload:
        return None
    predicted = canonical_lane_type(
        payload.get("special_lane_type")
        or payload.get("predicted_type")
        or payload.get("prediction")
        or payload.get("class")
    )
    result = dict(payload)
    result["special_lane_type"] = predicted
    result["one_hot"] = one_hot(predicted)
    if "confidence" not in result:
        result["confidence"] = None
    return result


def make_result_record(
    *,
    pipeline: str,
    model: str,
    sample,
    lane: dict,
    boundary_match,
    request_payload: dict,
    raw_response: str | None = None,
    parsed_response: dict | None = None,
    status: str = "ok",
    error: str | None = None,
) -> dict:
    prediction = normalize_prediction(parsed_response)
    gt_type = lane.get("gt_type")
    return {
        "schema_version": SCHEMA_VERSION,
        "pipeline": pipeline,
        "model": model,
        "image_id": sample.image_id,
        "image_path": str(sample.image_path),
        "lane_id": lane["lane_id"],
        "source_lane_id": lane["source_id"],
        "gt_type": gt_type,
        "gt_one_hot": lane["gt_one_hot"],
        "prediction": prediction,
        "pred_type": prediction["special_lane_type"] if prediction else None,
        "pred_one_hot": prediction["one_hot"] if prediction else {label: 0 for label in SPECIAL_LANE_CLASSES},
        "correct": (prediction["special_lane_type"] == gt_type) if prediction and gt_type else None,
        "boundary_match": {
            "left_boundary_id": boundary_match.left.get("boundary_id") if boundary_match.left else None,
            "right_boundary_id": boundary_match.right.get("boundary_id") if boundary_match.right else None,
            "left_gap_px": boundary_match.left_score,
            "right_gap_px": boundary_match.right_score,
        },
        "request": request_payload,
        "raw_response": raw_response,
        "status": status,
        "error": error,
    }
