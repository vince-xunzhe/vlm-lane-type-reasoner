# Alpha Clean85 Production Inference Release

This package contains only the Alpha Clean85 production inference path.

```text
code/   Python code and run scripts
data/   model, config template, input/output folders
```

Install:

```bash
cd code
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

VLM resource:

```bash
# Default production profile:
# base_url = http://192.167.20.200:8000/v1
# model    = qwen3.6-27b
# thinking = enabled
export VLM_RESOURCE_PROFILE="prod_qwen36_27b"
```

The release still keeps the old EAS profiles in
`data/configs/vlm_resources.example.json` for rollback/testing.

Run:

```bash
cd ..
code/scripts/run_alpha_clean85_inference.sh
```

Main output:

```text
data/outputs/pipeline_alpha_clean85_prod_results.json
```

Downstream should read `records[].pred_type` or
`records[].prediction.special_lane_type`.

LAS production input:

```bash
code/scripts/run_las_alpha_clean85_inference.sh \
  /nas/nfs/large-model/vince/data/xd-online-las-data/3702-1-00L025-260315
```

Docker build:

```bash
docker build -t vlm-grounding-reasoner-alpha-clean85:20260617 .
```

Docker image delivery:

```bash
docker load -i vlm-grounding-reasoner-alpha-clean85-20260617-amd64.docker.tar.gz
docker images | grep vlm-grounding-reasoner-alpha-clean85
```

Docker run:

```bash
docker run --rm \
  --network host \
  -v /nas/nfs/large-model/vince/data:/nas/nfs/large-model/vince/data \
  vlm-grounding-reasoner-alpha-clean85:20260617 \
  /nas/nfs/large-model/vince/data/xd-online-las-data/3702-1-00L025-260315
```

If `--network host` is unavailable, make sure the container can reach:

```text
http://192.167.20.200:8000/v1/chat/completions
```

For a LAS directory `<las_dir>` whose basename is `<las_name>`, the script reads:

```text
<las_dir>/<las_name>/Data/Img/Camera0
<las_dir>/<las_name>_r_LaneCenterLine/center_line_2d/output
<las_dir>/<las_name>_r_LaneCenterLine/lanes
<las_dir>/<las_name>_r_LaneCenterLine/objects/pred
```

All generated inputs, intermediate VLM records, status summaries, and final
outputs are written under:

```text
<las_dir>/<las_name>_r_LaneCenterLine/inference/output/
<las_dir>/<las_name>_r_LaneCenterLine/inference/panels/
```

The LAS-format final output is:

```text
<las_dir>/<las_name>_r_LaneCenterLine/inference/output/center_line_2d/<image_id>.json
```

Each final JSON keeps only the `lane` field and rewrites each valid
`lane[].attribute` to one of `bus`, `tidal`, `variable`, `bicycle`, or `normal`.

The downstream LaneType interface output is generated automatically from that
folder:

```text
<las_dir>/<las_name>_r_LaneCenterLine/output/output_lanes_attr.json
```

Visual debug artifacts are generated automatically after LAS inference:

```text
<las_dir>/<las_name>_r_LaneCenterLine/vis_debug/index.html
```

Set `RUN_VISUALIZATION=0` to skip this step.  For quick checks, use
`VISUALIZATION_LIMIT=<n>` or `VISUALIZATION_FRAMES=<frame_id[,frame_id...]>`.
If `<las_dir>/<las_name>_r_LaneCenterLine/visualize_instances.yaml`
exists, only the frame stems listed there are visualized; if it is missing,
all discovered frames are visualized.

For iterative experiments, set `ROUND_NAME=round-1` or `AUTO_ROUND=1`.
Round mode writes inference and visualization artifacts under:

```text
<las_dir>/<las_name>_r_LaneCenterLine/inference/<round_name>/output/
<las_dir>/<las_name>_r_LaneCenterLine/inference/<round_name>/panels/
<las_dir>/<las_name>_r_LaneCenterLine/output/<round_name>/output_lanes_attr.json
<las_dir>/<las_name>_r_LaneCenterLine/vis_debug/<round_name>/index.html
```
