#!/usr/bin/env bash
set -euo pipefail

CODE_DIR="${CODE_DIR:-/nas/nfs/large-model/vince/code/vlm-grounding-reasoner-release}"
INPUT_DATA_DIR="${INPUT_DATA_DIR:-/nas/nfs/large-model/vince/data/vlm-post-train-data-workshop-lane-type}"
RUN_DIR="${RUN_DIR:-/nas/nfs/large-model/vince/data/vlm-grounding-reasoner-release/outputs/workshop_lane_type_allfit_20260612}"
RESOURCE_CONFIG="${VLM_RESOURCE_CONFIG:-/nas/nfs/large-model/vince/data/vlm-grounding-reasoner-release/configs/vlm_resources.example.json}"
RESOURCE_PROFILE="${VLM_RESOURCE_PROFILE:-prod_qwen36_27b}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
WORKERS="${WORKERS:-8}"
TIMEOUT="${TIMEOUT:-660}"
REQUEST_RETRIES="${REQUEST_RETRIES:-2}"
RETRY_SLEEP_SECONDS="${RETRY_SLEEP_SECONDS:-10}"
MAX_OBJECT_CROPS="${MAX_OBJECT_CROPS:-2}"

mkdir -p "$RUN_DIR/panels" "$RUN_DIR/model"
cd "$CODE_DIR"
export CODE_DIR RUN_DIR INPUT_DATA_DIR

run_pass() {
  local name="$1"
  local prompt_mode="$2"
  local output="$RUN_DIR/${name}_results.json"
  local panel_dir="$RUN_DIR/panels/${name}"
  shift 2
  if [[ -s "$output" ]]; then
    echo "[skip] $name exists: $output"
    return 0
  fi
  echo "[run] $name prompt=$prompt_mode output=$output"
  "$PYTHON_BIN" pipeline_alpha/run_qwen_pipeline.py \
    --data-dir "$INPUT_DATA_DIR" \
    --resource-config "$RESOURCE_CONFIG" \
    --resource-profile "$RESOURCE_PROFILE" \
    --workers "$WORKERS" \
    --timeout "$TIMEOUT" \
    --max-object-crops "$MAX_OBJECT_CROPS" \
    --request-retries "$REQUEST_RETRIES" \
    --retry-sleep-seconds "$RETRY_SLEEP_SECONDS" \
    --prompt-mode "$prompt_mode" \
    --output "$output" \
    --panel-dir "$panel_dir" \
    --incremental-jsonl "$RUN_DIR/${name}_records.jsonl" \
    --resume-from-incremental \
    "$@"
}

run_pass alpha_raw standard
run_pass alpha_geom standard --boundary-match-mode overlap_rescue
run_pass alpha_line_recall line_recall

MODEL_PATH="$RUN_DIR/model/pipeline_alpha_clean85_extra_trees_fit_all.pkl"

if [[ ! -s "$MODEL_PATH" ]]; then
  echo "[train] fit-all decision head"
  "$PYTHON_BIN" - <<'PY'
import json
import os
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np

code_dir = Path(os.environ.get("CODE_DIR", "/nas/nfs/large-model/vince/code/vlm-grounding-reasoner-release"))
sys.path.insert(0, str(code_dir))
from scripts.alpha_autoresearch_search import build_feature_set, make_model  # noqa: E402
from scripts.calibrate_pipeline_internal import load_records  # noqa: E402

run_dir = Path(os.environ["RUN_DIR"])
input_data_dir = Path(os.environ["INPUT_DATA_DIR"])
records_by_name = {
    "alpha_raw": load_records(run_dir / "alpha_raw_results.json"),
    "alpha_geom": load_records(run_dir / "alpha_geom_results.json"),
    "alpha_line_recall": load_records(run_dir / "alpha_line_recall_results.json"),
}
keys = sorted(records_by_name["alpha_raw"])
config = {
    "model": "extra_trees",
    "subset": ["alpha_raw", "alpha_geom", "alpha_line_recall"],
    "feature_variant": "base+meta+clean",
    "n_jobs": -1,
    "n_estimators": 500,
    "max_depth": 20,
    "min_samples_leaf": 2,
    "class_weight": "balanced_subsample",
}
features = build_feature_set(records_by_name, keys, input_data_dir, "base+meta+clean")
labels = np.array([records_by_name["alpha_raw"][key]["gt_type"] for key in keys])
model = make_model(config, 13)
model.fit(features, labels)
preds = [str(item) for item in model.predict(features)]
correct = sum(pred == label for pred, label in zip(preds, labels))
model_path = run_dir / "model/pipeline_alpha_clean85_extra_trees_fit_all.pkl"
with model_path.open("wb") as fp:
    pickle.dump(model, fp)
summary = {
    "mode": "fit-all",
    "records": len(keys),
    "correct": int(correct),
    "accuracy": round(correct / len(keys), 6),
    "label_distribution": dict(Counter(labels)),
    "pred_distribution": dict(Counter(preds)),
    "config": config,
    "model_path": str(model_path),
    "runtime": {"numpy": np.__version__},
    "note": "Fit-all replay accuracy is training-set replay, not held-out evaluation.",
}
(run_dir / "model/pipeline_alpha_clean85_extra_trees_fit_all_summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
print(json.dumps(summary, ensure_ascii=False))
PY
else
  echo "[skip] fit-all model exists: $MODEL_PATH"
fi

FINAL_OUTPUT="$RUN_DIR/pipeline_alpha_clean85_fit_all_results.json"
echo "[apply] final inference output=$FINAL_OUTPUT"
"$PYTHON_BIN" scripts/apply_alpha_clean85_model.py \
  --data-dir "$INPUT_DATA_DIR" \
  --model "$MODEL_PATH" \
  --pipeline-name pipeline_alpha_clean85_workshop_fit_all \
  --primary alpha_raw \
  --input "alpha_raw=$RUN_DIR/alpha_raw_results.json" \
  --input "alpha_geom=$RUN_DIR/alpha_geom_results.json" \
  --input "alpha_line_recall=$RUN_DIR/alpha_line_recall_results.json" \
  --output "$FINAL_OUTPUT"

"$PYTHON_BIN" - <<'PY'
import json
import os
from collections import Counter
from pathlib import Path

run_dir = Path(os.environ["RUN_DIR"])
summary = {"run_dir": str(run_dir)}
for name in ("alpha_raw", "alpha_geom", "alpha_line_recall"):
    data = json.load((run_dir / f"{name}_results.json").open(encoding="utf-8"))
    summary[name] = {
        "records": data.get("num_records"),
        "dry_run": data.get("dry_run"),
        "status": dict(Counter(record.get("status") for record in data["records"])),
        "pred_distribution": dict(Counter(record.get("pred_type") for record in data["records"])),
    }
final = json.load((run_dir / "pipeline_alpha_clean85_fit_all_results.json").open(encoding="utf-8"))
summary["final"] = {
    "records": final.get("num_records"),
    "pred_distribution": dict(Counter(record.get("pred_type") for record in final["records"])),
}
summary["fit_all_train"] = json.load(
    (run_dir / "model/pipeline_alpha_clean85_extra_trees_fit_all_summary.json").open(encoding="utf-8")
)
(run_dir / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False))
PY
