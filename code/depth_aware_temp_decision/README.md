# Depth-Aware Temporary Decision

This folder contains a temporary rule-based decision path for downstream
debugging.  It consumes the depth-aware association probe output and writes the
same LaneType interface format as the production path.

This path intentionally does not retrain or replace the Clean85 decision head.
It is a bridge for testing depth-aware association before the association
features are integrated into the trained decision model.

## Run

```bash
code/depth_aware_temp_decision/run_depth_aware_temp_decision.sh \
  --las-dir /nas/nfs/large-model/vince/data/xd-online-las-data/3702-1-00L025-260316 \
  --round-name round-11-depth-aware-temp-decision \
  --overwrite
```

The standard downstream output is written to:

```text
<las_dir>/<las_name>_r_LaneCenterLine/output/<round_name>/output_lanes_attr.json
```

The per-frame canonical lane-attribute JSON files are written to:

```text
<las_dir>/<las_name>_r_LaneCenterLine/inference/<round_name>/depth_aware_temp_decision/center_line_2d/
```

By default, the script reuses an existing depth-aware association result for the
round, or runs `scripts/probe_3d_lane_association.py` if it is missing.
