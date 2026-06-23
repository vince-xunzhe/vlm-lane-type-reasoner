# Data Layout

Put production inputs under:

```text
data/input/
  images/<image_id>.jpg
  jsons/<image_id>.json
  lanes/<image_id>.json
  objects/<image_id>.json or objects/<image_id>_segments.json
```

Required runtime assets already included:

```text
data/models/pipeline_alpha_clean85_extra_trees.pkl
data/configs/vlm_resources.example.json
```

The default VLM profile is `prod_qwen36_27b`:

```text
base_url: http://192.167.20.200:8000/v1
model: qwen3.6-27b
api_key: sk-no-key-required
enable_thinking: true
```

The final prediction file is written to:

```text
data/outputs/pipeline_alpha_clean85_prod_results.json
```

For LAS production deployment, pass the absolute LAS directory to:

```bash
release/code/scripts/run_las_alpha_clean85_inference.sh <las_dir>
```

The expected LAS input layout is:

```text
<las_dir>/<las_name>/Data/Img/Camera0
<las_dir>/<las_name>_r_LaneCenterLine/center_line_2d/output
<las_dir>/<las_name>_r_LaneCenterLine/lanes
<las_dir>/<las_name>_r_LaneCenterLine/objects/pred
```

Generated artifacts are placed in:

```text
<las_dir>/<las_name>_r_LaneCenterLine/inference/output/
<las_dir>/<las_name>_r_LaneCenterLine/inference/panels/
```

The downstream LaneType interface file is generated at:

```text
<las_dir>/<las_name>_r_LaneCenterLine/output/output_lanes_attr.json
```
