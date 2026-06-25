#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RELEASE_DIR="$(cd "$CODE_DIR/.." && pwd)"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  echo "Usage: $0 <las_dir>" >&2
  echo "Example: $0 /nas/nfs/large-model/vince/data/xd-online-las-data/3702-1-00L025-260315" >&2
  exit 0
fi

if [[ $# -ge 1 ]]; then
  LAS_DIR="$1"
else
  LAS_DIR="${LAS_DIR:-}"
fi

if [[ -z "$LAS_DIR" ]]; then
  echo "Usage: $0 <las_dir>" >&2
  echo "Example: $0 /nas/nfs/large-model/vince/data/xd-online-las-data/3702-1-00L025-260315" >&2
  exit 2
fi

LAS_DIR="${LAS_DIR%/}"
LAS_NAME="$(basename "$LAS_DIR")"
LANE_TYPE_DIR="$LAS_DIR/${LAS_NAME}_r_LaneCenterLine"
INFERENCE_DIR="${INFERENCE_DIR:-$LANE_TYPE_DIR/inference}"
ROUND_NAME="${ROUND_NAME:-${RUN_ROUND:-}}"
AUTO_ROUND="${AUTO_ROUND:-0}"
if [[ -z "$ROUND_NAME" && "$AUTO_ROUND" == "1" ]]; then
  ROUND_INDEX=1
  while [[ -e "$INFERENCE_DIR/round-$ROUND_INDEX" || -e "$LANE_TYPE_DIR/vis_debug/round-$ROUND_INDEX" || -e "$LANE_TYPE_DIR/output/round-$ROUND_INDEX" ]]; do
    ROUND_INDEX=$((ROUND_INDEX + 1))
  done
  ROUND_NAME="round-$ROUND_INDEX"
fi
if [[ -n "$ROUND_NAME" ]]; then
  ROUND_INFERENCE_DIR="$INFERENCE_DIR/$ROUND_NAME"
  DEFAULT_OUTPUT_DIR="$ROUND_INFERENCE_DIR/output"
  DEFAULT_PANELS_DIR="$ROUND_INFERENCE_DIR/panels"
  DEFAULT_VISUALIZATION_OUTPUT_DIR="$LANE_TYPE_DIR/vis_debug/$ROUND_NAME"
  DEFAULT_FINAL_LANETYPE_OUTPUT_DIR="$LANE_TYPE_DIR/output/$ROUND_NAME"
else
  DEFAULT_OUTPUT_DIR="$INFERENCE_DIR/output"
  DEFAULT_PANELS_DIR="$INFERENCE_DIR/panels"
  DEFAULT_VISUALIZATION_OUTPUT_DIR="$LANE_TYPE_DIR/vis_debug"
  DEFAULT_FINAL_LANETYPE_OUTPUT_DIR="$LANE_TYPE_DIR/output"
fi
OUTPUT_DIR="${OUTPUT_DIR:-$DEFAULT_OUTPUT_DIR}"
PANELS_DIR="${PANELS_DIR:-$DEFAULT_PANELS_DIR}"
ADAPTED_DATA_DIR="${ADAPTED_DATA_DIR:-$OUTPUT_DIR/input}"
MANIFEST_PATH="${MANIFEST_PATH:-$OUTPUT_DIR/input_manifest.json}"

DEFAULT_RELEASE_DATA_DIR="$RELEASE_DIR/data"
if [[ ! -d "$DEFAULT_RELEASE_DATA_DIR" && -d "/nas/nfs/large-model/vince/data/vlm-grounding-reasoner-release" ]]; then
  DEFAULT_RELEASE_DATA_DIR="/nas/nfs/large-model/vince/data/vlm-grounding-reasoner-release"
fi
RELEASE_DATA_DIR="${RELEASE_DATA_DIR:-$DEFAULT_RELEASE_DATA_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_PATH="${MODEL_PATH:-$RELEASE_DATA_DIR/models/pipeline_alpha_clean85_extra_trees.pkl}"
RESOURCE_CONFIG="${VLM_RESOURCE_CONFIG:-$RELEASE_DATA_DIR/configs/vlm_resources.example.json}"
RESOURCE_PROFILE="${VLM_RESOURCE_PROFILE:-prod_qwen36_27b}"

WORKERS="${WORKERS:-4}"
TIMEOUT="${TIMEOUT:-660}"
MAX_OBJECT_CROPS="${MAX_OBJECT_CROPS:-2}"
REQUEST_RETRIES="${REQUEST_RETRIES:-2}"
RETRY_SLEEP_SECONDS="${RETRY_SLEEP_SECONDS:-10}"
ALPHA_GEOM_BOUNDARY_MATCH_MODE="${ALPHA_GEOM_BOUNDARY_MATCH_MODE:-overlap_rescue}"
ATTRIBUTE_FORMAT="${ATTRIBUTE_FORMAT:-canonical}"
LINK_MODE="${LINK_MODE:-symlink}"
FORCE="${FORCE:-0}"
LIMIT_IMAGES="${LIMIT_IMAGES:-}"
RUN_VISUALIZATION="${RUN_VISUALIZATION:-1}"
VISUALIZATION_OUTPUT_DIR="${VISUALIZATION_OUTPUT_DIR:-$DEFAULT_VISUALIZATION_OUTPUT_DIR}"
VISUALIZATION_INSTANCES_YAML="${VISUALIZATION_INSTANCES_YAML:-$LANE_TYPE_DIR/visualize_instances.yaml}"
VISUALIZATION_WORKERS="${VISUALIZATION_WORKERS:-$WORKERS}"
VISUALIZATION_LIMIT="${VISUALIZATION_LIMIT:-0}"
VISUALIZATION_FRAMES="${VISUALIZATION_FRAMES:-}"
VISUALIZATION_OVERWRITE="${VISUALIZATION_OVERWRITE:-1}"
VISUALIZATION_NO_PANELS="${VISUALIZATION_NO_PANELS:-0}"
VISUALIZATION_PANEL_THUMB_WIDTH="${VISUALIZATION_PANEL_THUMB_WIDTH:-320}"
VISUALIZATION_PANEL_MAX="${VISUALIZATION_PANEL_MAX:-0}"
VISUALIZATION_PANEL_PAGE_SIZE="${VISUALIZATION_PANEL_PAGE_SIZE:-40}"
VISUALIZATION_SUMMARY_WIDTH="${VISUALIZATION_SUMMARY_WIDTH:-640}"

mkdir -p "$OUTPUT_DIR" "$PANELS_DIR"
if [[ -n "$ROUND_NAME" ]]; then
  echo "[round] $ROUND_NAME"
fi

"$PYTHON_BIN" "$CODE_DIR/scripts/prepare_las_alpha_input.py" \
  --las-dir "$LAS_DIR" \
  --output-data-dir "$ADAPTED_DATA_DIR" \
  --manifest "$MANIFEST_PATH" \
  --link-mode "$LINK_MODE" \
  --overwrite

COMMON_ARGS=(
  --data-dir "$ADAPTED_DATA_DIR"
  --resource-config "$RESOURCE_CONFIG"
  --resource-profile "$RESOURCE_PROFILE"
  --workers "$WORKERS"
  --timeout "$TIMEOUT"
  --max-object-crops "$MAX_OBJECT_CROPS"
  --request-retries "$REQUEST_RETRIES"
  --retry-sleep-seconds "$RETRY_SLEEP_SECONDS"
)

if [[ -n "$LIMIT_IMAGES" ]]; then
  COMMON_ARGS+=(--limit-images "$LIMIT_IMAGES")
fi

run_pass() {
  local name="$1"
  local prompt_mode="$2"
  local output="$OUTPUT_DIR/${name}_results.json"
  local panel_dir="$PANELS_DIR/${name}"
  shift 2

  if [[ "$FORCE" == "1" ]]; then
    rm -f "$output" "$OUTPUT_DIR/${name}_records.jsonl"
  fi
  if [[ -s "$output" ]]; then
    echo "[skip] $name exists: $output"
    return 0
  fi

  echo "[run] $name prompt=$prompt_mode output=$output"
  "$PYTHON_BIN" "$CODE_DIR/pipeline_alpha/run_qwen_pipeline.py" \
    "${COMMON_ARGS[@]}" \
    --prompt-mode "$prompt_mode" \
    --output "$output" \
    --panel-dir "$panel_dir" \
    --incremental-jsonl "$OUTPUT_DIR/${name}_records.jsonl" \
    --resume-from-incremental \
    "$@"
}

run_pass alpha_raw standard
run_pass alpha_geom standard --boundary-match-mode "$ALPHA_GEOM_BOUNDARY_MATCH_MODE"
run_pass alpha_line_recall line_recall

FINAL_OUTPUT="$OUTPUT_DIR/pipeline_alpha_clean85_prod_results.json"
if [[ "$FORCE" == "1" ]]; then
  rm -f "$FINAL_OUTPUT"
fi

echo "[apply] final inference output=$FINAL_OUTPUT"
"$PYTHON_BIN" "$CODE_DIR/scripts/apply_alpha_clean85_model.py" \
  --data-dir "$ADAPTED_DATA_DIR" \
  --model "$MODEL_PATH" \
  --pipeline-name pipeline_alpha_clean85_las_prod \
  --primary alpha_raw \
  --input "alpha_raw=$OUTPUT_DIR/alpha_raw_results.json" \
  --input "alpha_geom=$OUTPUT_DIR/alpha_geom_results.json" \
  --input "alpha_line_recall=$OUTPUT_DIR/alpha_line_recall_results.json" \
  --output "$FINAL_OUTPUT"

LANE_ATTRIBUTE_OUTPUT_DIR="${LANE_ATTRIBUTE_OUTPUT_DIR:-$OUTPUT_DIR/center_line_2d}"
LANE_ATTRIBUTE_SUMMARY="${LANE_ATTRIBUTE_SUMMARY:-$OUTPUT_DIR/lane_attribute_summary.json}"
FINAL_LANETYPE_OUTPUT_DIR="${FINAL_LANETYPE_OUTPUT_DIR:-$DEFAULT_FINAL_LANETYPE_OUTPUT_DIR}"
FINAL_LANETYPE_OUTPUT_JSON="$FINAL_LANETYPE_OUTPUT_DIR/output_lanes_attr.json"

"$PYTHON_BIN" "$CODE_DIR/scripts/write_las_lane_attributes.py" \
  --manifest "$MANIFEST_PATH" \
  --predictions "$FINAL_OUTPUT" \
  --output-dir "$LANE_ATTRIBUTE_OUTPUT_DIR" \
  --summary "$LANE_ATTRIBUTE_SUMMARY" \
  --attribute-format "$ATTRIBUTE_FORMAT"

"$PYTHON_BIN" "$CODE_DIR/scripts/output_to_final_LaneType_output.py" \
  --vlm-output-dir "$LANE_ATTRIBUTE_OUTPUT_DIR" \
  --output-folder "$FINAL_LANETYPE_OUTPUT_DIR" \
  --exist-ok \
  --no-progress

VISUALIZATION_INDEX_HTML="$VISUALIZATION_OUTPUT_DIR/index.html"
if [[ "$RUN_VISUALIZATION" == "1" ]]; then
  VISUALIZATION_ARGS=(
    --las-dir "$LAS_DIR"
    --output-dir "$VISUALIZATION_OUTPUT_DIR"
    --attr-path "$FINAL_LANETYPE_OUTPUT_JSON"
    --instances-yaml "$VISUALIZATION_INSTANCES_YAML"
    --workers "$VISUALIZATION_WORKERS"
    --limit "$VISUALIZATION_LIMIT"
    --panel-thumb-width "$VISUALIZATION_PANEL_THUMB_WIDTH"
    --panel-max "$VISUALIZATION_PANEL_MAX"
    --panel-page-size "$VISUALIZATION_PANEL_PAGE_SIZE"
    --summary-width "$VISUALIZATION_SUMMARY_WIDTH"
  )
  if [[ "$VISUALIZATION_OVERWRITE" == "1" ]]; then
    VISUALIZATION_ARGS+=(--overwrite)
  fi
  if [[ "$VISUALIZATION_NO_PANELS" == "1" ]]; then
    VISUALIZATION_ARGS+=(--no-panels)
  fi
  if [[ -n "$VISUALIZATION_FRAMES" ]]; then
    VISUALIZATION_ARGS+=(--frames "$VISUALIZATION_FRAMES")
  fi

  echo "[visualize] output=$VISUALIZATION_OUTPUT_DIR"
  "$PYTHON_BIN" "$CODE_DIR/scripts/visualize_lane_modules.py" "${VISUALIZATION_ARGS[@]}"
fi

echo "Final prediction records: $FINAL_OUTPUT"
echo "Final LAS lane attributes: $LANE_ATTRIBUTE_OUTPUT_DIR"
echo "Final downstream LaneType output: $FINAL_LANETYPE_OUTPUT_JSON"
if [[ -n "$ROUND_NAME" ]]; then
  echo "Round: $ROUND_NAME"
fi
if [[ "$RUN_VISUALIZATION" == "1" ]]; then
  echo "Visual debug output: $VISUALIZATION_OUTPUT_DIR"
  echo "Visual debug index: $VISUALIZATION_INDEX_HTML"
fi
