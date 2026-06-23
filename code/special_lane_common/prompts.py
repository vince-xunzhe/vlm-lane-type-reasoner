"""Prompt builders shared by Qwen special-lane pipelines."""

from __future__ import annotations

import json


SYSTEM_PROMPT = """你是自动驾驶场景里的车道属性与特殊车道分类器。
你只处理用户给定的一条目标车道，不做开放式检测，不新增坐标。
图像坐标统一为 Qwen-VL 常用的 norm_1000 坐标：左上角为 (0,0)，右下角为 (1000,1000)。
你必须输出严格 JSON，不要输出解释性正文。"""


OUTPUT_SCHEMA = {
    "lane_id": "<target_lane_id>",
    "object_semantics": [
        {
            "object_id": "<object_id>",
            "semantic_type": "bus_text_gong_or_bus_text_jiao_or_bus_sign_or_bus_restriction_sign_or_variable_text_ke_or_variable_text_bian_or_variable_lane_signal_or_red_x_signal_or_tidal_text_chao_or_tidal_text_xi_or_bicycle_sign_or_bicycle_icon_or_other_or_irrelevant",
            "visible_text": "<text_if_any>",
            "target_class_prior": "bus_or_tidal_or_variable_or_bicycle_or_normal_or_none",
            "is_relevant_traffic_cue": True,
            "confidence": 0.0,
        }
    ],
    "line_attributes": {
        "left": {
            "color": "white_or_yellow_or_unknown",
            "pattern": "solid_or_dashed_or_unknown",
            "multiplicity": "single_or_double_or_unknown",
            "shape": "straight_or_curve_or_zigzag_or_channelizing_area_or_others",
            "label": "single_white_solid_or_double_yellow_dash_or_zigzag_or_others_or_unknown",
            "is_variable_zigzag_boundary": False,
            "is_tidal_double_yellow_dashed_boundary": False,
            "confidence": 0.0,
        },
        "right": {
            "color": "white_or_yellow_or_unknown",
            "pattern": "solid_or_dashed_or_unknown",
            "multiplicity": "single_or_double_or_unknown",
            "shape": "straight_or_curve_or_zigzag_or_channelizing_area_or_others",
            "label": "single_white_solid_or_double_yellow_dash_or_zigzag_or_others_or_unknown",
            "is_variable_zigzag_boundary": False,
            "is_tidal_double_yellow_dashed_boundary": False,
            "confidence": 0.0,
        },
    },
    "object_relations": [
        {
            "object_id": "<object_id>",
            "label_name": "<detector_label_or_visual_label>",
            "relation": "applies_to_target_lane_or_nearby_or_irrelevant",
            "evidence_type": "road_text_or_road_symbol_or_sign_or_signal_or_other",
            "confidence": 0.0,
        }
    ],
    "special_lane_type": "bus_or_tidal_or_variable_or_bicycle_or_normal",
    "one_hot": {"bus": 0, "tidal": 0, "variable": 0, "bicycle": 0, "normal": 1},
    "confidence": 0.0,
    "supporting_objects": ["<object_id>"],
    "supporting_line_sides": ["left", "right"],
}


def _json_block(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def alpha_prompt(packet: dict) -> str:
    view_lines = []
    for idx, view in enumerate(packet.get("views", []), start=1):
        view_lines.append(f"- view_{idx}: {view.get('id')} - {view.get('desc')}")
    if not view_lines:
        view_lines.append("- view_1: original frame")
    return "\n".join(
        [
            "任务：判断目标车道是否为特殊车道，并输出业务 one-hot 类型。",
            "",
            "输入说明：",
            "- 你会收到多张图像视图，这些视图共同组成 alpha region packet。",
            "- target_lane 来自人工标注 jsons，是本次唯一要判断的指代车道。",
            "- left_boundary/right_boundary 来自车道线检测 single_line；它们是目标车道的左右边界候选。",
            "- objects 来自感知检测 bbox，可作为公交/自行车/潮汐/可变车道的文字、标志牌或信号证据。",
            "- objects 中的 target_lane_geometry 是预先计算的几何提示：anchor_point_2d 为对象关联锚点；inside_target_lane_band 表示该锚点是否落在左右边界夹出的目标车道带内。",
            "- 所有坐标均为 norm_1000，不是原始像素。",
            "",
            "图像视图顺序：",
            *view_lines,
            "",
            "重要约束：",
            "- global_overlay 用来看全局位置和 id。",
            "- lane_focus_crop 用来看目标 lane、左右边界和附近对象的真实图像细节。",
            "- lane_mask_whiteboard 用来看结构化细线、目标车道和左右边界关系。",
            "- left_boundary_raw_zoom/right_boundary_raw_zoom 用来看未被 overlay 遮挡的左右边界真实线型；判断颜色、虚实、单双和 zigzag 时优先使用这两张图。",
            "- object_crops/contact_sheet 用来看检测对象的局部语义。",
            "- 不要创建新 id；只能引用 packet 中已有 lane_id / boundary_id / object_id。",
            "",
            "请完成：",
            "1. 对每个 object bbox 做细语义判断，输出 object_semantics。尤其区分 mixed_lane_signal_candidate 中的 variable_lane_signal 与 red_x_signal。",
            "2. 观察左右边界线，分别判断颜色 yellow/white/unknown、线型 solid/dashed/unknown、单双 single/double/unknown、形态 zigzag/channelizing_area/others/straight_or_curve。",
            "3. 判断 objects 与 target_lane 的空间关联，只能在给定 object_id 中选择证据。",
            "4. 输出 special_lane_type，只能为 bus/tidal/variable/bicycle/normal；若证据不足，选择 normal。",
            "",
            "空间关联硬约束：",
            "- 对路面文字/路面图标类 object（如 公、交、可、变、自行车图标），只有当 inside_target_lane_band=true，或视觉上完整落在目标车道带内，才能判 applies_to_target_lane。",
            "- 若 object 的 relation_hint 是 left_adjacent_or_outside 或 right_adjacent_or_outside，默认 relation=nearby/irrelevant；不要因为它本身是特殊车道符号就套到目标车道。",
            "- 对标志牌/信号灯类 object，只有在其箭头/安装位置/投影明显约束目标车道时才判 applies_to_target_lane；否则为 nearby。",
            "- 最终分类必须只使用 relation=applies_to_target_lane 的 object 证据，以及左右边界自身线型证据。",
            "",
            "业务判别重点：",
            "- 先做视觉证据抽取，再做类别判断；不要因为想输出某类而反推线型或 object 语义。",
            "- 普通车道线是连续/虚线的直线或曲线，应输出 shape=straight_or_curve，is_variable_zigzag_boundary=false，is_tidal_double_yellow_dashed_boundary=false。",
            "- zigzag 只指沿目标车道边界连续锯齿状、折线状的可变车道边线；导流区/渠化区斜纹、宽三角区、阴影区必须输出 channelizing_area 或 others，不算 variable 边界。",
            "- double_yellow_dashed 只指同一侧边界清楚可见双黄虚线；不要把黄实线、白虚线、普通车道分隔线或远处相邻线判成 true。",
            "- mixed_lane_signal_candidate 不是天然等于 variable_lane_signal；若看不清是可变车道信号/红叉，semantic_type=other 或 irrelevant。",
            "- variable: 需要强证据。通常要求左右边界为真正 zigzag，或目标车道内有“可”“变”文字，或明确约束目标车道的可变车道信号。",
            "- tidal: 需要强证据。通常要求左右边界为 double + yellow + dashed，或目标车道关联红叉/潮汐文字。",
            "- bus: “公”“交”、公交专用/时段限制标志是强证据。",
            "- bicycle: 自行车图标或自行车标志牌是强证据。",
            "- normal: 只有在特殊车道证据都不成立或与目标车道不关联时输出。",
            "- 当只有疑似信号、疑似特殊线型，但无法确认属于目标车道时，优先输出 normal。",
            "",
            "region_packet:",
            "```json",
            _json_block(packet),
            "```",
            "",
            "严格按以下 JSON schema 输出，字段名不要改变：",
            "```json",
            _json_block(OUTPUT_SCHEMA),
            "```",
        ]
    )


def alpha_compact_prompt(packet: dict) -> str:
    view_lines = []
    for idx, view in enumerate(packet.get("views", []), start=1):
        view_lines.append(f"- view_{idx}: {view.get('id')} - {view.get('desc')}")
    if not view_lines:
        view_lines.append("- view_1: original frame")
    return "\n".join(
        [
            "任务：对唯一指定的 target_lane 做一次完整 evidence extraction，并给出特殊车道 one-hot 类型。",
            "",
            "输入说明：",
            "- 你会收到 alpha multiview packet：全局叠加图、目标车道 crop、结构白板图、object crops/contact sheet。",
            "- target_lane 是唯一要判断的车道；left_boundary/right_boundary 是该目标车道的左右边界候选。",
            "- objects 是检测到的道路文字、符号、标志牌、信号灯等候选证据。",
            "- object.target_lane_geometry 是几何先验，不是最终答案；它帮助你判断 object 是否约束 target_lane。",
            "- 坐标为 Qwen norm_1000。",
            "",
            "图像视图顺序：",
            *view_lines,
            "",
            "推理顺序要求：",
            "1. 先看 lane_focus_crop 和真实图像，判断左右边界的真实线型。",
            "2. 再看 left_boundary_raw_zoom/right_boundary_raw_zoom；如果 overlay 与 raw zoom 冲突，以 raw zoom 的真实路面纹理为准。",
            "3. 再看 lane_mask_whiteboard，确认目标车道、左右边界和 object 的空间关系。",
            "4. 再看 object crops/contact sheet，判断每个 object 的语义。",
            "5. 最后综合证据输出 special_lane_type。",
            "",
            "线型判定细则：",
            "- 普通连续/虚线的直线或曲线：shape=straight_or_curve，zigzag=false，tidal_double_yellow_dashed=false。",
            "- variable zigzag 只指目标车道左右边界上连续锯齿/折线状的可变车道边线。",
            "- 导流区、渠化区、斜纹填充区、三角禁行区：shape=channelizing_area 或 others，不算 variable zigzag。",
            "- tidal double yellow dashed 只指目标车道边界上清楚的双黄虚线，不要把黄实线/白虚线/远处相邻线当作潮汐边界。",
            "",
            "object 关联细则：",
            "- road text / road symbol 必须位于目标车道带内，或视觉上完整属于目标车道，才能 relation=applies_to_target_lane。",
            "- sign / signal 必须箭头、投影、安装位置或车道上方位置明确约束目标车道，才能 applies_to_target_lane。",
            "- 如果 object.target_lane_geometry.relation_hint 是 left_adjacent_or_outside 或 right_adjacent_or_outside，默认 nearby/irrelevant，除非图像明确显示它约束 target_lane。",
            "- mixed_lane_signal_candidate 需要视觉确认；看不清时 semantic_type=other 或 irrelevant。",
            "",
            "类别判定细则：",
            "- bus: 目标车道内“公/交”文字或明确公交专用/公交时段标志。",
            "- bicycle: 目标车道内自行车图标或明确自行车道标志。",
            "- variable: 目标车道左右边界是真 zigzag，或目标车道内“可/变”文字，或明确约束目标车道的可变车道信号。",
            "- tidal: 目标车道左右边界是双黄虚线，或明确约束目标车道的红叉/潮汐文字。",
            "- normal: 上述特殊证据不足、相邻车道证据、或证据冲突无法确认时输出 normal。",
            "",
            "输出要求：",
            "- 只能引用 packet 中已有 object_id / boundary_id / lane_id。",
            "- object_semantics、line_attributes、object_relations 必须与最终 special_lane_type 一致。",
            "- 不要为了匹配某类别反推线型或 object 语义。",
            "",
            "region_packet:",
            "```json",
            _json_block(packet),
            "```",
            "",
            "严格按以下 JSON schema 输出，字段名不要改变：",
            "```json",
            _json_block(OUTPUT_SCHEMA),
            "```",
        ]
    )


def alpha_line_prompt(packet: dict) -> str:
    view_lines = []
    for idx, view in enumerate(packet.get("views", []), start=1):
        view_lines.append(f"- view_{idx}: {view.get('id')} - {view.get('desc')}")
    if not view_lines:
        view_lines.append("- view_1: original frame")
    return "\n".join(
        [
            "任务：只做目标车道左右边界线型证据抽取，重点判断 variable/tidal/normal 的线型差异。",
            "",
            "重要：本轮不是 object 语义任务。objects 只用于空间参照，最终分类主要基于 left_boundary/right_boundary 的线型证据。",
            "",
            "输入：",
            "- target_lane 是唯一目标车道。",
            "- left_boundary/right_boundary 是目标车道左右边界候选。",
            "- 多张视图包括真实图像 crop 和结构白板图；请优先看 lane_focus_crop 里的真实路面线型，再用 whiteboard 确认对应边界。",
            "- 如果存在 left_boundary_raw_zoom/right_boundary_raw_zoom，请把它们作为左右边界颜色、虚实、单双和 zigzag 的最高优先级证据；这些 crop 尽量不覆盖真实道路标线。",
            "- 坐标为 norm_1000。",
            "",
            "图像视图顺序：",
            *view_lines,
            "",
            "请严格区分：",
            "- normal 普通线：直线/曲线的白线或黄线，实线/虚线均可；shape=straight_or_curve。",
            "- variable zigzag：目标车道边界自身呈连续锯齿/折线/拉链状，且沿车道方向延伸；is_variable_zigzag_boundary=true。",
            "- channelizing_area：导流区/渠化区/斜纹填充/三角禁行区，不属于目标车道边界 zigzag；shape=channelizing_area，is_variable_zigzag_boundary=false。",
            "- tidal double yellow dashed：目标车道边界自身是清楚的双黄虚线；is_tidal_double_yellow_dashed_boundary=true。",
            "- 如果线条被遮挡、太远、或只看到相邻线，不要猜；输出 unknown/others 并降低 confidence。",
            "",
            "输出 special_lane_type 的规则：",
            "- 如果左右边界均明确为 variable zigzag，输出 variable。",
            "- 如果左右边界均明确为 double yellow dashed，输出 tidal。",
            "- 如果线型证据不满足 variable/tidal，输出 normal。",
            "- bus/bicycle 不是本轮任务；除非线型视图和目标车道内物体证据极明确，否则不要输出 bus/bicycle。",
            "",
            "object 字段要求：",
            "- object_semantics 可以为空列表，或仅输出你非常确定且与目标车道关联的对象。",
            "- object_relations 可以为空列表；不要为了支持线型结论而强行关联 object。",
            "",
            "region_packet:",
            "```json",
            _json_block(packet),
            "```",
            "",
            "严格按以下 JSON schema 输出，字段名不要改变：",
            "```json",
            _json_block(OUTPUT_SCHEMA),
            "```",
        ]
    )


def alpha_line_recall_prompt(packet: dict) -> str:
    view_lines = []
    for idx, view in enumerate(packet.get("views", []), start=1):
        view_lines.append(f"- view_{idx}: {view.get('id')} - {view.get('desc')}")
    if not view_lines:
        view_lines.append("- view_1: original frame")
    return "\n".join(
        [
            "任务：高召回抽取目标车道左右边界的特殊线型证据，并输出候选特殊车道类型。",
            "",
            "本轮目标不是保守判最终业务类型，而是尽量不要漏掉 variable/tidal 的线型线索。下游会再用分类器校准误报。",
            "",
            "输入：",
            "- target_lane 是唯一目标车道。",
            "- left_boundary/right_boundary 是目标车道左右边界候选。",
            "- 如果存在 left_boundary_raw_zoom/right_boundary_raw_zoom，必须优先用它们判断真实道路标线，不要被 overlay 颜色影响。",
            "- lane_mask_whiteboard 只用于确认哪条线是 left/right，不用于判断真实颜色。",
            "- 坐标为 norm_1000。",
            "",
            "图像视图顺序：",
            *view_lines,
            "",
            "高召回线型规则：",
            "- 看到边界沿车道方向呈连续折线、锯齿、拉链形、周期性横向摆动，即标记 shape=zigzag，并置 is_variable_zigzag_boundary=true。",
            "- 如果 raw zoom 中有部分遮挡，但可见段已经呈 zigzag，也应标记 true，并降低 confidence，而不是直接 normal。",
            "- 看到同一侧边界由两条并行黄色虚线组成，即标记 color=yellow、pattern=dashed、multiplicity=double，并置 is_tidal_double_yellow_dashed_boundary=true。",
            "- 如果只是一大片导流区/三角区/斜纹填充，shape=channelizing_area，variable=false。",
            "- 如果边界明显是普通白/黄直线或曲线，shape=straight_or_curve。",
            "",
            "候选 special_lane_type 规则：",
            "- 任一侧或双侧存在 zigzag 候选时，优先输出 variable；supporting_line_sides 写入对应 side。",
            "- 任一侧或双侧存在 double yellow dashed 候选时，优先输出 tidal；supporting_line_sides 写入对应 side。",
            "- 如果 variable 与 tidal 同时存在，选择证据更强、confidence 更高者。",
            "- 只有当左右边界都清楚是普通线且没有特殊线型候选时，才输出 normal。",
            "- bus/bicycle 不是本轮重点；除非 object crop 中极明确且属于目标车道，否则不要输出。",
            "",
            "输出要求：",
            "- line_attributes 是最重要字段，必须完整填写 left/right。",
            "- object_semantics 和 object_relations 可以为空或只填非常确定的对象。",
            "- confidence 反映可见证据强弱；不要因为高召回而伪造看不见的线。",
            "",
            "region_packet:",
            "```json",
            _json_block(packet),
            "```",
            "",
            "严格按以下 JSON schema 输出，字段名不要改变：",
            "```json",
            _json_block(OUTPUT_SCHEMA),
            "```",
        ]
    )


def beta_prompt(packet: dict) -> str:
    return "\n".join(
        [
            "任务：基于点/框 grounding 判断目标车道的左右车道线属性、标志关联关系与特殊车道类型。",
            "",
            "Qwen-VL grounding 约定：",
            "- 车道线用 `points_2d` 指代，这些点落在检测到的车道线 polygon/polyline 上；请沿这些点连接的可见车道线观察线型。",
            "- 标志牌/文字/信号等对象用 `bbox_2d` 指代。",
            "- 每个 object 还有 target_lane_geometry：anchor_point_2d 为对象关联锚点；inside_target_lane_band 表示锚点是否落在目标车道左右边界夹出的车道带内。",
            "- 坐标为 norm_1000：左上角 (0,0)，右下角 (1000,1000)。",
            "- 不要重新检测新区域，只判断 packet 内已有 left_line_points/right_line_points 与 objects。",
            "",
            "请完成：",
            "1. 对每个 object bbox 做细语义判断，输出 object_semantics。尤其区分 mixed_lane_signal_candidate 中的 variable_lane_signal 与 red_x_signal。",
            "2. 对 left_line_points 和 right_line_points 所指代车道线分别判断 yellow/white、solid/dashed、single/double、zigzag/channelizing_area/others。",
            "3. 对每个 bbox object 判断是否约束 target_lane。",
            "4. 输出 bus/tidal/variable/bicycle/normal 的 one-hot 分类；证据不足时输出 normal。",
            "",
            "空间关联硬约束：",
            "- 对路面文字/路面图标类 object（如 公、交、可、变、自行车图标），只有当 inside_target_lane_band=true，或视觉上完整落在目标车道带内，才能判 applies_to_target_lane。",
            "- 若 object 的 relation_hint 是 left_adjacent_or_outside 或 right_adjacent_or_outside，默认 relation=nearby/irrelevant；不要因为它本身是特殊车道符号就套到目标车道。",
            "- 对标志牌/信号灯类 object，只有在其箭头/安装位置/投影明显约束目标车道时才判 applies_to_target_lane；否则为 nearby。",
            "- 最终分类必须只使用 relation=applies_to_target_lane 的 object 证据，以及 left_line_points/right_line_points 所指示的左右边界线型证据。",
            "",
            "业务判别重点：",
            "- 先做视觉证据抽取，再做类别判断；不要因为想输出某类而反推线型或 object 语义。",
            "- 普通车道线是连续/虚线的直线或曲线，应输出 shape=straight_or_curve，is_variable_zigzag_boundary=false，is_tidal_double_yellow_dashed_boundary=false。",
            "- zigzag 只指沿 target_lane 左右边界连续锯齿状、折线状的可变车道边线；导流区/渠化区斜纹、宽三角区、阴影区必须输出 channelizing_area 或 others，不算 variable 边界。",
            "- double_yellow_dashed 只指 left_line_points/right_line_points 所指的同一侧边界清楚可见双黄虚线；不要把黄实线、白虚线、普通车道分隔线或远处相邻线判成 true。",
            "- mixed_lane_signal_candidate 不是天然等于 variable_lane_signal；若看不清是可变车道信号/红叉，semantic_type=other 或 irrelevant。",
            "- variable: 需要强证据。通常要求左右边界为真正 zigzag，或目标车道内有“可”“变”文字，或明确约束目标车道的可变车道信号。",
            "- tidal: 需要强证据。通常要求左右边界为 double + yellow + dashed，或目标车道关联红叉/潮汐文字。",
            "- bus: “公”“交”、公交专用/时段限制标志是强证据。",
            "- bicycle: 自行车图标或自行车标志牌是强证据。",
            "- normal: 只有在特殊车道证据都不成立或与目标车道不关联时输出。",
            "- 当只有疑似信号、疑似特殊线型，但无法确认属于目标车道时，优先输出 normal。",
            "",
            "grounding_packet:",
            "```json",
            _json_block(packet),
            "```",
            "",
            "严格按以下 JSON schema 输出，字段名不要改变：",
            "```json",
            _json_block(OUTPUT_SCHEMA),
            "```",
        ]
    )
