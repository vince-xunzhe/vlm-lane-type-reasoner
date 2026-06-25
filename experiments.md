# Experiments

## 2026-06-25 - `round-2-association-probe`

Goal: diagnose camera-consistent 3D soft assignment between detected objects and
2D lane centerlines using DA3 depth and SAM3 road masks, without changing the
existing VLM or decision-head outputs.

Remote project:
`/nas/nfs/large-model/vince/code/vlm-lane-type-reasoner`

Remote data:
`/nas/nfs/large-model/vince/data/xd-online-las-data/3702-1-00L025-260316/3702-1-00L025-260316_r_LaneCenterLine`

Command:

```bash
python3 /nas/nfs/large-model/vince/code/vlm-lane-type-reasoner/code/scripts/probe_3d_lane_association.py \
  --base-dir /nas/nfs/large-model/vince/data/xd-online-las-data/3702-1-00L025-260316/3702-1-00L025-260316_r_LaneCenterLine \
  --round-name round-2-association-probe \
  --workers 4 \
  --overwrite
```

Result:

- Frames from `visualize_instances.yaml`: 14 selected, 0 missing.
- Successful frames: 14.
- Active objects after filtering: 5.
- Filtered sign/signal objects: 11.
- The motivating case `040-0-073637-940-000720` was filtered before lane
  assignment because the bus-related sign depth was about 53.9 m, above the
  20 m sign threshold.

Artifacts:

- Summary JSON:
  `/nas/nfs/large-model/vince/data/xd-online-las-data/3702-1-00L025-260316/3702-1-00L025-260316_r_LaneCenterLine/inference/round-2-association-probe/association_3d/association_results.json`
- Visual index:
  `/nas/nfs/large-model/vince/data/xd-online-las-data/3702-1-00L025-260316/3702-1-00L025-260316_r_LaneCenterLine/vis_debug/round-2-association-probe/association_3d/index.html`

Status: diagnostic-only, kept as a probe for reviewing the association failure
mode before wiring it into the production inference chain.

## 2026-06-25 - `round-3-association-refine`

Goal: refine the 3D association probe after visual inspection.

Changes:

- Increased sign/signal depth threshold from 20 m to 100 m.
- Unconditionally filtered `mixed_lane_signal_candidate` before any downstream
  association.
- Replaced lateral nearest-road rescue for lane depth with strict sampling:
  exact 2D centerline pixels inside SAM3 road mask first, then ego-side
  extension-line samples only.

Command:

```bash
python3 /nas/nfs/large-model/vince/code/vlm-lane-type-reasoner/code/scripts/probe_3d_lane_association.py \
  --base-dir /nas/nfs/large-model/vince/data/xd-online-las-data/3702-1-00L025-260316/3702-1-00L025-260316_r_LaneCenterLine \
  --round-name round-3-association-refine \
  --workers 4 \
  --overwrite
```

Result:

- Frames from `visualize_instances.yaml`: 14 selected, 0 missing.
- Successful frames: 14.
- Active objects after filtering: 9.
- Filtered objects: 7.
- Invalid lanes under strict road-mask sampling: 5.
- Lanes that attempted ego-side extension: 7.
- `040-0-073834-567-000744` is no longer filtered at 22.0 m and assigns top-1
  to lane 150; its fully occluded lane 147 is invalid and does not participate.

Artifacts:

- Summary JSON:
  `/nas/nfs/large-model/vince/data/xd-online-las-data/3702-1-00L025-260316/3702-1-00L025-260316_r_LaneCenterLine/inference/round-3-association-refine/association_3d/association_results.json`
- Visual index:
  `/nas/nfs/large-model/vince/data/xd-online-las-data/3702-1-00L025-260316/3702-1-00L025-260316_r_LaneCenterLine/vis_debug/round-3-association-refine/association_3d/index.html`

## 2026-06-25 - `round-4-sign-anchor`

Goal: replace the misleading sign-to-ground scatter heuristic with sign anchor
matching.

Changes:

- Disabled sign/signal ground scatter search entirely.
- Sign/signal objects now use the DA3 depth median inside the bbox to form a
  camera-space anchor.
- Sign-to-lane matching compares the anchor horizontal bearing against lane 3D
  profile samples, with longitudinal depth gap as a penalty.
- Visualization now draws a white sign anchor and purple top-lane support
  samples instead of yellow ground candidates for sign/signal objects.

Command:

```bash
python3 /nas/nfs/large-model/vince/code/vlm-lane-type-reasoner/code/scripts/probe_3d_lane_association.py \
  --base-dir /nas/nfs/large-model/vince/data/xd-online-las-data/3702-1-00L025-260316/3702-1-00L025-260316_r_LaneCenterLine \
  --round-name round-4-sign-anchor \
  --workers 4 \
  --overwrite
```

Result:

- Frames from `visualize_instances.yaml`: 14 selected, 0 missing.
- Successful frames: 14.
- Active objects after filtering: 9.
- Filtered objects: 7.
- Active sign/signal objects using anchor matching: 4.
- All active sign/signal objects have `ground_candidate_count=0`.
- `040-0-073834-567-000744` changed from broad ground scatter top-1 lane 150
  to anchor-profile top-1 lane 151, with lower confidence `p=0.359`.

Artifacts:

- Summary JSON:
  `/nas/nfs/large-model/vince/data/xd-online-las-data/3702-1-00L025-260316/3702-1-00L025-260316_r_LaneCenterLine/inference/round-4-sign-anchor/association_3d/association_results.json`
- Visual index:
  `/nas/nfs/large-model/vince/data/xd-online-las-data/3702-1-00L025-260316/3702-1-00L025-260316_r_LaneCenterLine/vis_debug/round-4-sign-anchor/association_3d/index.html`

## 2026-06-25 - `round-5-bev-confidence`

Goal: remove the camera-bearing sign heuristic and gate low-confidence sign
depths.

Changes:

- Replaced sign horizontal-bearing scoring with BEV XZ distance between the
  sign depth anchor and lane 3D profile samples.
- Added sign/signal depth confidence filtering using bbox `conf_median` and
  `conf_p75`.
- Kept sign ground scatter search disabled.

Command:

```bash
python3 /nas/nfs/large-model/vince/code/vlm-lane-type-reasoner/code/scripts/probe_3d_lane_association.py \
  --base-dir /nas/nfs/large-model/vince/data/xd-online-las-data/3702-1-00L025-260316/3702-1-00L025-260316_r_LaneCenterLine \
  --round-name round-5-bev-confidence \
  --workers 4 \
  --overwrite
```

Result:

- Frames from `visualize_instances.yaml`: 14 selected, 0 missing.
- Successful frames: 14.
- Active objects after filtering: 5.
- Filtered objects: 11.
- Filter reasons: `mixed_lane_signal_candidate_filtered=6`,
  `sign_depth_confidence_low=4`, `sign_depth_gt_max=1`.
- Current active sign/signal count is 0 because all sign bbox confidence
  medians and p75 values are at the DA3 confidence floor.

Artifacts:

- Summary JSON:
  `/nas/nfs/large-model/vince/data/xd-online-las-data/3702-1-00L025-260316/3702-1-00L025-260316_r_LaneCenterLine/inference/round-5-bev-confidence/association_3d/association_results.json`
- Visual index:
  `/nas/nfs/large-model/vince/data/xd-online-las-data/3702-1-00L025-260316/3702-1-00L025-260316_r_LaneCenterLine/vis_debug/round-5-bev-confidence/association_3d/index.html`

## 2026-06-25 - `round-6-lane-band-bev`

Goal: fix road-surface object association that was overfitting to the drawn 2D
lane centerline rather than the physical lane region.

Changes:

- Road marking/object association now uses BEV lane bands inferred from adjacent
  lane centers at each depth slice.
- Lane centerlines remain only lane indicators; scoring uses whether road-surface
  object candidates fall inside each lane band, plus band-distance and
  center-offset terms.
- Added `bev_debug/` visualizations that show confidence, lane bands, object
  BEV candidates, and the distance lines used by the lane-band calculation.
- Overlay labels now show DA3 confidence as `conf_median/conf_p75`.

Command:

```bash
python3 /nas/nfs/large-model/vince/code/vlm-lane-type-reasoner/code/scripts/probe_3d_lane_association.py \
  --base-dir /nas/nfs/large-model/vince/data/xd-online-las-data/3702-1-00L025-260316/3702-1-00L025-260316_r_LaneCenterLine \
  --round-name round-6-lane-band-bev \
  --workers 4 \
  --overwrite
```

Result:

- Frames from `visualize_instances.yaml`: 14 selected, 0 missing.
- Successful frames: 14.
- Active objects after filtering: 5.
- Filtered objects: 11.
- Motivating frame `040-0-071229-369-000120`: `bicycle_icon` changed from the
  previous lane-centerline-biased top-1 lane `1` to lane `2`.
- For that frame, lane `2` has `p=0.938`, `inside_fraction=0.932`,
  `robust_band_distance=0.0m`; lane `1` has `p=0.062`.

Artifacts:

- Summary JSON:
  `/nas/nfs/large-model/vince/data/xd-online-las-data/3702-1-00L025-260316/3702-1-00L025-260316_r_LaneCenterLine/inference/round-6-lane-band-bev/association_3d/association_results.json`
- Visual index:
  `/nas/nfs/large-model/vince/data/xd-online-las-data/3702-1-00L025-260316/3702-1-00L025-260316_r_LaneCenterLine/vis_debug/round-6-lane-band-bev/association_3d/index.html`
- Motivating BEV debug:
  `/nas/nfs/large-model/vince/data/xd-online-las-data/3702-1-00L025-260316/3702-1-00L025-260316_r_LaneCenterLine/vis_debug/round-6-lane-band-bev/association_3d/bev_debug/040-0-071229-369-000120.jpg`

## 2026-06-25 - `round-7-forward-lane-extension`

Goal: avoid false invalid lanes when the visible 2D lane indicator is occluded
but its forward collinear extension intersects SAM3 road mask.

Changes:

- Lane depth sampling fallback is now `centerline_exact -> forward_extension ->
  ego_extension`.
- `forward_extension` starts from the image-near lane endpoint and samples along
  the lane direction toward the image-far endpoint and beyond it.
- The fallback still samples strictly on the lane line direction; it does not
  perform lateral nearest-road search.
- Per-lane JSON now records extension attempts, selected extension source, and
  forward/ego extension road-hit counts.

Command:

```bash
python3 /nas/nfs/large-model/vince/code/vlm-lane-type-reasoner/code/scripts/probe_3d_lane_association.py \
  --base-dir /nas/nfs/large-model/vince/data/xd-online-las-data/3702-1-00L025-260316/3702-1-00L025-260316_r_LaneCenterLine \
  --round-name round-7-forward-lane-extension \
  --workers 4 \
  --overwrite
```

Result:

- Frames from `visualize_instances.yaml`: 14 selected, 0 missing.
- Successful frames: 14.
- Active objects after filtering: 5.
- Filtered objects: 11.
- Motivating frame `040-0-073637-940-000720`: lane `134` changed from invalid
  to valid using `forward_extension`.
- Lane `134` now has `depth_count=23`, `depth_median=44.85m`,
  `forward_extension_road_hit_count=23`, `ego_extension_road_hit_count=0`.
- The previous `040-0-071229-369-000120` `bicycle_icon` case remains assigned
  to lane `2` with `p=0.938`.

Artifacts:

- Summary JSON:
  `/nas/nfs/large-model/vince/data/xd-online-las-data/3702-1-00L025-260316/3702-1-00L025-260316_r_LaneCenterLine/inference/round-7-forward-lane-extension/association_3d/association_results.json`
- Visual index:
  `/nas/nfs/large-model/vince/data/xd-online-las-data/3702-1-00L025-260316/3702-1-00L025-260316_r_LaneCenterLine/vis_debug/round-7-forward-lane-extension/association_3d/index.html`
- Motivating overlay:
  `/nas/nfs/large-model/vince/data/xd-online-las-data/3702-1-00L025-260316/3702-1-00L025-260316_r_LaneCenterLine/vis_debug/round-7-forward-lane-extension/association_3d/association_overlay/040-0-073637-940-000720.jpg`

## 2026-06-25 - `round-8-confidence-visualization`

Goal: expose DA3 confidence evidence used by sign/signal filtering before
changing the gate.

Changes:

- Added `confidence_overlay/` per-frame visualizations.
- The confidence overlay shows a DA3 confidence heatmap with the normal lane and
  object overlays.
- Each confidence image includes global confidence percentiles and the current
  sign gate thresholds.
- Per-frame JSON now records full-scene confidence stats.

Command:

```bash
python3 /nas/nfs/large-model/vince/code/vlm-lane-type-reasoner/code/scripts/probe_3d_lane_association.py \
  --base-dir /nas/nfs/large-model/vince/data/xd-online-las-data/3702-1-00L025-260316/3702-1-00L025-260316_r_LaneCenterLine \
  --round-name round-8-confidence-visualization \
  --workers 4 \
  --overwrite
```

Result:

- Frames from `visualize_instances.yaml`: 14 selected, 0 missing.
- Successful frames: 14.
- Active objects after filtering: 5.
- Filtered objects: 11.
- Filter reasons are unchanged: `mixed_lane_signal_candidate_filtered=6`,
  `sign_depth_confidence_low=4`, `sign_depth_gt_max=1`.
- The inspected `bus_related_time_restriction_sign` examples filtered by
  confidence all have bbox `conf_median=1.00` and `conf_p75=1.00` under the
  current `1.01/1.01` sign gate.

Artifacts:

- Summary JSON:
  `/nas/nfs/large-model/vince/data/xd-online-las-data/3702-1-00L025-260316/3702-1-00L025-260316_r_LaneCenterLine/inference/round-8-confidence-visualization/association_3d/association_results.json`
- Visual index:
  `/nas/nfs/large-model/vince/data/xd-online-las-data/3702-1-00L025-260316/3702-1-00L025-260316_r_LaneCenterLine/vis_debug/round-8-confidence-visualization/association_3d/index.html`
- Confidence example:
  `/nas/nfs/large-model/vince/data/xd-online-las-data/3702-1-00L025-260316/3702-1-00L025-260316_r_LaneCenterLine/vis_debug/round-8-confidence-visualization/association_3d/confidence_overlay/040-0-073637-940-000720.jpg`

## 2026-06-25 - `round-9-no-sign-confidence-filter`

Goal: stop dropping sign/signal detections solely because DA3 confidence is at
the floor value.

Changes:

- Disabled sign/signal depth-confidence filtering by default.
- Kept `conf_median`, `conf_p75`, and the stored threshold gate in JSON for
  diagnostics.
- Kept `confidence_overlay/` visualizations; the legend now reports
  `sign conf filter off`.
- Existing filters still apply: `mixed_lane_signal_candidate` is always
  filtered, unreliable/no-depth boxes are filtered, and signs beyond
  `sign_max_depth_m=100m` are filtered.

Command:

```bash
python3 /nas/nfs/large-model/vince/code/vlm-lane-type-reasoner/code/scripts/probe_3d_lane_association.py \
  --base-dir /nas/nfs/large-model/vince/data/xd-online-las-data/3702-1-00L025-260316/3702-1-00L025-260316_r_LaneCenterLine \
  --round-name round-9-no-sign-confidence-filter \
  --workers 4 \
  --overwrite
```

Result:

- Frames from `visualize_instances.yaml`: 14 selected, 0 missing.
- Successful frames: 14.
- Active objects after filtering: 9, up from 5 in `round-8`.
- Filtered objects: 7, down from 11 in `round-8`.
- Filter reasons: `mixed_lane_signal_candidate_filtered=6`,
  `sign_depth_gt_max=1`.
- `sign_depth_confidence_low` no longer appears as a filter reason.
- Motivating frame `040-0-073637-940-000720`: the
  `bus_related_time_restriction_sign` is active and assigns to lane `134` with
  `p=0.708`, despite bbox confidence `1.00/1.00`.

Artifacts:

- Summary JSON:
  `/nas/nfs/large-model/vince/data/xd-online-las-data/3702-1-00L025-260316/3702-1-00L025-260316_r_LaneCenterLine/inference/round-9-no-sign-confidence-filter/association_3d/association_results.json`
- Visual index:
  `/nas/nfs/large-model/vince/data/xd-online-las-data/3702-1-00L025-260316/3702-1-00L025-260316_r_LaneCenterLine/vis_debug/round-9-no-sign-confidence-filter/association_3d/index.html`
- Motivating overlay:
  `/nas/nfs/large-model/vince/data/xd-online-las-data/3702-1-00L025-260316/3702-1-00L025-260316_r_LaneCenterLine/vis_debug/round-9-no-sign-confidence-filter/association_3d/association_overlay/040-0-073637-940-000720.jpg`

## 2026-06-25 - `round-10-sign-lateral-band`

Goal: make sign/signal lane association prioritize BEV lateral ownership instead
of nearest sampled lane profile distance.

Changes:

- Replaced sign/signal nearest-profile BEV distance scoring with lateral
  lane-band scoring at the object's DA3 anchor depth.
- Lane profile extrapolation is still allowed, but it only estimates each
  lane's lateral band at the object depth; it no longer wins purely because an
  extrapolated/sampled segment is closer in longitudinal distance.
- Lane depth quality remains a weak auxiliary factor.
- BEV debug tables now show `in`, `d`, `c`, and `ex`: inside-band flag,
  band-boundary distance, center offset, and lane extrapolation length.
- Sign/signal depth-confidence filtering remains disabled from `round-9`.

Command:

```bash
python3 /nas/nfs/large-model/vince/code/vlm-lane-type-reasoner/code/scripts/probe_3d_lane_association.py \
  --base-dir /nas/nfs/large-model/vince/data/xd-online-las-data/3702-1-00L025-260316/3702-1-00L025-260316_r_LaneCenterLine \
  --round-name round-10-sign-lateral-band \
  --workers 4 \
  --overwrite
```

Result:

- Frames from `visualize_instances.yaml`: 14 selected, 0 missing.
- Successful frames: 14.
- Active objects after filtering: 9.
- Filtered objects: 7.
- Motivating frame `040-0-073340-889-000687`: the
  `bus_related_time_restriction_sign` changes from lane `107` in `round-9`
  (`p=0.418`) to lane `108` (`p=0.855`).
- In that frame, lane `108` is inside the lateral band at the sign anchor depth
  with `band_distance=0.000m`, `center_offset=0.158m`, and
  `z_extrapolation=20.301m`; lane `107` is outside with
  `band_distance=1.082m` and `center_offset=2.322m`.
- Frame `040-0-073637-940-000720` remains associated to lane `134` (`p=0.544`).

Artifacts:

- Summary JSON:
  `/nas/nfs/large-model/vince/data/xd-online-las-data/3702-1-00L025-260316/3702-1-00L025-260316_r_LaneCenterLine/inference/round-10-sign-lateral-band/association_3d/association_results.json`
- Visual index:
  `/nas/nfs/large-model/vince/data/xd-online-las-data/3702-1-00L025-260316/3702-1-00L025-260316_r_LaneCenterLine/vis_debug/round-10-sign-lateral-band/association_3d/index.html`
- Motivating overlay:
  `/nas/nfs/large-model/vince/data/xd-online-las-data/3702-1-00L025-260316/3702-1-00L025-260316_r_LaneCenterLine/vis_debug/round-10-sign-lateral-band/association_3d/association_overlay/040-0-073340-889-000687.jpg`
- Motivating BEV debug:
  `/nas/nfs/large-model/vince/data/xd-online-las-data/3702-1-00L025-260316/3702-1-00L025-260316_r_LaneCenterLine/vis_debug/round-10-sign-lateral-band/association_3d/bev_debug/040-0-073340-889-000687.jpg`
