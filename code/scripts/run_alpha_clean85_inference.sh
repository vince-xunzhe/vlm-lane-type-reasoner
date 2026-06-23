#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RELEASE_DIR="$(cd "$CODE_DIR/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
INPUT_DATA_DIR="${INPUT_DATA_DIR:-$RELEASE_DIR/data/input}"
OUTPUT_DIR="${OUTPUT_DIR:-$RELEASE_DIR/data/outputs}"
MODEL_PATH="${MODEL_PATH:-$RELEASE_DIR/data/models/pipeline_alpha_clean85_extra_trees.pkl}"
RESOURCE_CONFIG="${VLM_RESOURCE_CONFIG:-$RELEASE_DIR/data/configs/vlm_resources.example.json}"
RESOURCE_PROFILE="${VLM_RESOURCE_PROFILE:-prod_qwen36_27b}"

WORKERS="${WORKERS:-4}"
TIMEOUT="${TIMEOUT:-180}"
MAX_OBJECT_CROPS="${MAX_OBJECT_CROPS:-2}"
REQUEST_RETRIES="${REQUEST_RETRIES:-1}"
RETRY_SLEEP_SECONDS="${RETRY_SLEEP_SECONDS:-5}"
ALPHA_GEOM_BOUNDARY_MATCH_MODE="${ALPHA_GEOM_BOUNDARY_MATCH_MODE:-overlap_rescue}"

mkdir -p "$OUTPUT_DIR/panels"

COMMON_ARGS=(
  --data-dir "$INPUT_DATA_DIR"
  --resource-config "$RESOURCE_CONFIG"
  --resource-profile "$RESOURCE_PROFILE"
  --workers "$WORKERS"
  --timeout "$TIMEOUT"
  --max-object-crops "$MAX_OBJECT_CROPS"
  --request-retries "$REQUEST_RETRIES"
  --retry-sleep-seconds "$RETRY_SLEEP_SECONDS"
)

"$PYTHON_BIN" "$CODE_DIR/pipeline_alpha/run_qwen_pipeline.py" \
  "${COMMON_ARGS[@]}" \
  --prompt-mode standard \
  --output "$OUTPUT_DIR/alpha_raw_results.json" \
  --panel-dir "$OUTPUT_DIR/panels/alpha_raw"

"$PYTHON_BIN" "$CODE_DIR/pipeline_alpha/run_qwen_pipeline.py" \
  "${COMMON_ARGS[@]}" \
  --prompt-mode standard \
  --boundary-match-mode "$ALPHA_GEOM_BOUNDARY_MATCH_MODE" \
  --output "$OUTPUT_DIR/alpha_geom_results.json" \
  --panel-dir "$OUTPUT_DIR/panels/alpha_geom"

"$PYTHON_BIN" "$CODE_DIR/pipeline_alpha/run_qwen_pipeline.py" \
  "${COMMON_ARGS[@]}" \
  --prompt-mode line_recall \
  --output "$OUTPUT_DIR/alpha_line_recall_results.json" \
  --panel-dir "$OUTPUT_DIR/panels/alpha_line_recall"

"$PYTHON_BIN" "$CODE_DIR/scripts/apply_alpha_clean85_model.py" \
  --data-dir "$INPUT_DATA_DIR" \
  --model "$MODEL_PATH" \
  --pipeline-name pipeline_alpha_clean85_prod \
  --primary alpha_raw \
  --input "alpha_raw=$OUTPUT_DIR/alpha_raw_results.json" \
  --input "alpha_geom=$OUTPUT_DIR/alpha_geom_results.json" \
  --input "alpha_line_recall=$OUTPUT_DIR/alpha_line_recall_results.json" \
  --output "$OUTPUT_DIR/pipeline_alpha_clean85_prod_results.json"

echo "Final output: $OUTPUT_DIR/pipeline_alpha_clean85_prod_results.json"
