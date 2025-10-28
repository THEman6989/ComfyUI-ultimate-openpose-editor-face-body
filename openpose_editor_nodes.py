import json
import math
from copy import deepcopy

import torch
import numpy as np
from .util import draw_pose_json, draw_pose, extend_scalelist, pose_normalized

OpenposeJSON = dict


def _normalize_pose_source(pose_json: str) -> list:
    if not pose_json:
        return []
    if pose_json.startswith('{'):
        pose_json = f"[{pose_json}]"
    return json.loads(pose_json)


def _apply_confidence_smoothing(confs: list, alpha: float) -> None:
    alpha = min(max(alpha, 0.0), 1.0)
    last_conf = None
    for idx, value in enumerate(confs):
        if value is None:
            continue
        if last_conf is None:
            last_conf = value
        else:
            last_conf = alpha * value + (1.0 - alpha) * last_conf
        confs[idx] = last_conf


def _interpolate_missing(xs: list, ys: list, confs: list) -> None:
    n = len(xs)
    for idx in range(n):
        if xs[idx] is not None and ys[idx] is not None:
            continue
        prev_idx = next((j for j in range(idx - 1, -1, -1) if xs[j] is not None and ys[j] is not None), None)
        next_idx = next((j for j in range(idx + 1, n) if xs[j] is not None and ys[j] is not None), None)
        if prev_idx is not None and next_idx is not None:
            span = next_idx - prev_idx
            if span == 0:
                continue
            ratio = (idx - prev_idx) / span
            xs[idx] = xs[prev_idx] + (xs[next_idx] - xs[prev_idx]) * ratio
            ys[idx] = ys[prev_idx] + (ys[next_idx] - ys[prev_idx]) * ratio
            if confs[prev_idx] is not None and confs[next_idx] is not None:
                confs[idx] = confs[prev_idx] + (confs[next_idx] - confs[prev_idx]) * ratio
            else:
                confs[idx] = confs[prev_idx] if confs[prev_idx] is not None else confs[next_idx]
        elif prev_idx is not None:
            xs[idx] = xs[prev_idx]
            ys[idx] = ys[prev_idx]
            confs[idx] = confs[prev_idx]
        elif next_idx is not None:
            xs[idx] = xs[next_idx]
            ys[idx] = ys[next_idx]
            confs[idx] = confs[next_idx]


def _suppress_outliers(xs: list, ys: list, confs: list, diag_per_frame: list, jump_ratio: float) -> None:
    if jump_ratio <= 0.0:
        return
    n = len(xs)
    for idx in range(n):
        if xs[idx] is None or ys[idx] is None:
            continue
        prev_idx = next((j for j in range(idx - 1, -1, -1) if xs[j] is not None and ys[j] is not None), None)
        next_idx = next((j for j in range(idx + 1, n) if xs[j] is not None and ys[j] is not None), None)
        if prev_idx is None or next_idx is None:
            continue
        diag = max(diag_per_frame[idx], diag_per_frame[prev_idx], diag_per_frame[next_idx]) or 1.0
        max_jump = diag * jump_ratio
        prev_dist = math.hypot(xs[idx] - xs[prev_idx], ys[idx] - ys[prev_idx])
        next_dist = math.hypot(xs[idx] - xs[next_idx], ys[idx] - ys[next_idx])
        direct_dist = math.hypot(xs[next_idx] - xs[prev_idx], ys[next_idx] - ys[prev_idx])
        if prev_dist > max_jump and next_dist > max_jump and direct_dist < max_jump * 2.0:
            xs[idx] = None
            ys[idx] = None
            confs[idx] = None


def _smooth_keypoint_sequence(sequence: list, diag_per_frame: list, smooth_conf: bool, alpha: float,
                               filter_outliers: bool, jump_ratio: float, confidence_floor: float) -> list:
    if not any(sequence):
        return sequence
    template = next((kp for kp in sequence if kp), None)
    if template is None:
        return sequence
    num_points = len(template) // 3
    result = [None if kp is None else kp[:] for kp in sequence]
    for point_idx in range(num_points):
        xs = [None] * len(sequence)
        ys = [None] * len(sequence)
        confs = [None] * len(sequence)
        for frame_idx, kp in enumerate(sequence):
            if kp is None or len(kp) < (point_idx + 1) * 3:
                continue
            base = point_idx * 3
            conf_value = kp[base + 2]
            if conf_value is None or conf_value < confidence_floor:
                continue
            xs[frame_idx] = kp[base]
            ys[frame_idx] = kp[base + 1]
            confs[frame_idx] = conf_value
        if filter_outliers:
            _suppress_outliers(xs, ys, confs, diag_per_frame, jump_ratio)
        _interpolate_missing(xs, ys, confs)
        if smooth_conf:
            _apply_confidence_smoothing(confs, alpha)
        for frame_idx in range(len(sequence)):
            if result[frame_idx] is None:
                continue
            base = point_idx * 3
            if xs[frame_idx] is not None and ys[frame_idx] is not None:
                result[frame_idx][base] = xs[frame_idx]
                result[frame_idx][base + 1] = ys[frame_idx]
            if confs[frame_idx] is not None:
                result[frame_idx][base + 2] = max(confidence_floor, min(1.0, confs[frame_idx]))
    return result


def _smooth_pose_frames(frames: list, smooth_conf: bool, alpha: float, filter_outliers: bool,
                        jump_ratio: float, confidence_floor: float) -> list:
    processed = deepcopy(frames)
    if not processed:
        return processed
    diag_per_frame = []
    for image in processed:
        width = image.get('canvas_width') or 1.0
        height = image.get('canvas_height') or 1.0
        diag_per_frame.append(math.hypot(width, height))
    max_people = max((len(image.get('people', [])) for image in processed), default=0)
    key_sets = [
        'pose_keypoints_2d',
        'face_keypoints_2d',
        'hand_left_keypoints_2d',
        'hand_right_keypoints_2d'
    ]
    for person_idx in range(max_people):
        for key in key_sets:
            sequence = []
            for image in processed:
                figures = image.get('people', [])
                if person_idx < len(figures):
                    keypoints = figures[person_idx].get(key)
                    sequence.append(keypoints[:] if keypoints else None)
                else:
                    sequence.append(None)
            smoothed = _smooth_keypoint_sequence(sequence, diag_per_frame, smooth_conf, alpha,
                                                 filter_outliers, jump_ratio, confidence_floor)
            for frame_idx, new_values in enumerate(smoothed):
                figures = processed[frame_idx].get('people', [])
                if person_idx < len(figures) and new_values is not None:
                    figures[person_idx][key] = new_values
    return processed


def _prepare_pose_source(pose_json: str, smooth_conf: bool, alpha: float, filter_outliers: bool,
                         jump_ratio: float, confidence_floor: float) -> str:
    frames = _normalize_pose_source(pose_json)
    if not frames:
        return pose_json
    processed = _smooth_pose_frames(frames, smooth_conf, alpha, filter_outliers, jump_ratio, confidence_floor)
    return json.dumps(processed)

class OpenposeEditorNode10:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "show_body": ("BOOLEAN", {"default": True}),
                "show_face": ("BOOLEAN", {"default": True}),
                "show_hands": ("BOOLEAN", {"default": True}),
                "resolution_x": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 12800,
                    "tooltip": "Resolution X. -1 means use the original resolution."
                }),
                "pose_marker_size": ("INT", {
                    "default": 4,
                    "min": 0,
                    "max": 100
                }),
                "face_marker_size": ("INT", {
                    "default": 3,
                    "min": 0,
                    "max": 100
                }),
                "hand_marker_size": ("INT", {
                    "default": 2,
                    "min": 0,
                    "max": 100
                }),
                "hands_scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.05
                }),
                "body_scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.05
                }),
                "head_scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.05
                }),
                "overall_scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.05
                }),
                "scalelist_behavior": (["poses", "images"], {"default": "poses", "tooltip": "When the scale input is a list, this determines how the scale list takes effect, the differences appear when there are multiple persons(poses) in one image."}),
                "match_scalelist_method": (["no extend", "loop extend", "clamp extend"], {"default": "loop extend", "tooltip": "Match the scale list to the input poses or images when the scale list length is shorter. No extend: Beyound the scale list will be 1.0. Loop: Loop the scale list to match the poses or images length. Clamp: Use the last scale value to extend the scale list."}),
                "only_scale_pose_index": ("INT", {
                    "default": 99,
                    "min": -100,
                    "max": 100,
                    "tooltip": "For multiple poses in one image, the scale will be only applied at desired index. If set to a number larger than the number of poses in the image, the scale will be applied to all poses. Negative number will apply to the pose from the end."
                }),
                "POSE_JSON": ("STRING", {"multiline": True}),
                "POSE_KEYPOINT": ("POSE_KEYPOINT",{"default": None}),
            },
        }

    RETURN_NAMES = ("POSE_IMAGE", "POSE_KEYPOINT", "POSE_JSON")
    RETURN_TYPES = ("IMAGE", "POSE_KEYPOINT", "STRING")
    OUTPUT_NODE = True
    FUNCTION = "load_pose"
    CATEGORY = "ultimate-openpose"

    def load_pose(self, show_body, show_face, show_hands, resolution_x, pose_marker_size, face_marker_size, hand_marker_size, hands_scale, body_scale, head_scale, overall_scale, scalelist_behavior, match_scalelist_method, only_scale_pose_index, POSE_JSON: str, POSE_KEYPOINT=None) -> tuple[OpenposeJSON]:
        '''
        priority output is: POSE_JSON > POSE_KEYPOINT
        priority edit is: POSE_KEYPOINT > POSE_JSON
        '''
        if POSE_JSON:
            POSE_JSON = POSE_JSON.replace("'",'"').replace('None','[]')
            POSE_PASS = POSE_JSON
            if POSE_KEYPOINT is not None:
                POSE_PASS = json.dumps(POSE_KEYPOINT,indent=4).replace("'",'"').replace('None','[]')

            # parse the JSON
            hands_scalelist, body_scalelist, head_scalelist, overall_scalelist = extend_scalelist(
                scalelist_behavior, POSE_PASS, hands_scale, body_scale, head_scale, overall_scale,
                match_scalelist_method, only_scale_pose_index)
            normalized_pose_json = pose_normalized(POSE_PASS)
            pose_imgs, POSE_PASS_SCALED = draw_pose_json(normalized_pose_json, resolution_x, show_body, show_face, show_hands, pose_marker_size, face_marker_size, hand_marker_size, hands_scalelist, body_scalelist, head_scalelist, overall_scalelist)
            if pose_imgs:
                pose_imgs_np = np.array(pose_imgs).astype(np.float32) / 255
                return {
                    "ui": {"POSE_JSON": [json.dumps(POSE_PASS_SCALED, indent=4)]},
                    "result": (torch.from_numpy(pose_imgs_np), POSE_PASS_SCALED, json.dumps(POSE_PASS_SCALED,indent=4))
                }
        elif POSE_KEYPOINT is not None:
            POSE_JSON = json.dumps(POSE_KEYPOINT,indent=4).replace("'",'"').replace('None','[]')
            hands_scalelist, body_scalelist, head_scalelist, overall_scalelist = extend_scalelist(
                scalelist_behavior, POSE_JSON, hands_scale, body_scale, head_scale, overall_scale,
                match_scalelist_method, only_scale_pose_index)
            normalized_pose_json = pose_normalized(POSE_JSON)
            pose_imgs, POSE_SCALED = draw_pose_json(normalized_pose_json, resolution_x, show_body, show_face, show_hands, pose_marker_size, face_marker_size, hand_marker_size, hands_scalelist, body_scalelist, head_scalelist, overall_scalelist)
            if pose_imgs:
                pose_imgs_np = np.array(pose_imgs).astype(np.float32) / 255
                return {
                    "ui": {"POSE_JSON": [json.dumps(POSE_SCALED, indent=4)]},
                    "result": (torch.from_numpy(pose_imgs_np), POSE_SCALED, json.dumps(POSE_SCALED, indent=4))
                }

        # otherwise output blank images
        W=512
        H=768
        pose_draw = dict(bodies={'candidate':[], 'subset':[]}, faces=[], hands=[])
        pose_out = dict(pose_keypoints_2d=[], face_keypoints_2d=[], hand_left_keypoints_2d=[], hand_right_keypoints_2d=[])
        people=[dict(people=[pose_out], canvas_height=H, canvas_width=W)]

        W_scaled = resolution_x
        if resolution_x < 64:
            W_scaled = W
        H_scaled = int(H*(W_scaled*1.0/W))
        pose_img = [draw_pose(pose_draw, H_scaled, W_scaled, pose_marker_size, face_marker_size, hand_marker_size)]
        pose_img_np = np.array(pose_img).astype(np.float32) / 255

        return {
                "ui": {"POSE_JSON": people},
                "result": (torch.from_numpy(pose_img_np), people, json.dumps(people))
        }


class OpenposeEditorNode10V2(OpenposeEditorNode10):
    @classmethod
    def INPUT_TYPES(cls):
        base_optional = dict(super().INPUT_TYPES()["optional"])
        pose_json_cfg = base_optional.pop("POSE_JSON")
        pose_keypoint_cfg = base_optional.pop("POSE_KEYPOINT")
        base_optional.update({
            "enable_confidence_smoothing": ("BOOLEAN", {"default": True, "tooltip": "Smooth per-keypoint confidence across frames to reduce flicker."}),
            "confidence_smoothing_alpha": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Blend factor for the confidence exponential moving average."}),
            "enable_outlier_suppression": ("BOOLEAN", {"default": True, "tooltip": "Detect sudden jumps and replace them with interpolated positions."}),
            "outlier_jump_threshold": ("FLOAT", {"default": 0.12, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Maximum allowed motion as a fraction of the canvas diagonal before treating a point as an outlier."}),
            "confidence_floor": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Minimum confidence value to keep when smoothing and interpolating keypoints."}),
        })
        base_optional["POSE_JSON"] = pose_json_cfg
        base_optional["POSE_KEYPOINT"] = pose_keypoint_cfg
        return {"optional": base_optional}

    def load_pose(self, show_body, show_face, show_hands, resolution_x, pose_marker_size, face_marker_size,
                  hand_marker_size, hands_scale, body_scale, head_scale, overall_scale, scalelist_behavior,
                  match_scalelist_method, only_scale_pose_index, enable_confidence_smoothing,
                  confidence_smoothing_alpha, enable_outlier_suppression, outlier_jump_threshold,
                  confidence_floor, POSE_JSON: str, POSE_KEYPOINT=None) -> tuple[OpenposeJSON]:
        if POSE_JSON:
            POSE_JSON = POSE_JSON.replace("'", '"').replace('None', '[]')
            pose_source = POSE_JSON
            if POSE_KEYPOINT is not None:
                pose_source = json.dumps(POSE_KEYPOINT, indent=4).replace("'", '"').replace('None', '[]')
            if (enable_confidence_smoothing or enable_outlier_suppression) and pose_source:
                pose_source = _prepare_pose_source(
                    pose_source,
                    enable_confidence_smoothing,
                    confidence_smoothing_alpha,
                    enable_outlier_suppression,
                    outlier_jump_threshold,
                    confidence_floor,
                )

            hands_scalelist, body_scalelist, head_scalelist, overall_scalelist = extend_scalelist(
                scalelist_behavior, pose_source, hands_scale, body_scale, head_scale, overall_scale,
                match_scalelist_method, only_scale_pose_index)
            normalized_pose_json = pose_normalized(pose_source)
            pose_imgs, pose_scaled = draw_pose_json(
                normalized_pose_json, resolution_x, show_body, show_face, show_hands,
                pose_marker_size, face_marker_size, hand_marker_size, hands_scalelist, body_scalelist,
                head_scalelist, overall_scalelist)
            if pose_imgs:
                pose_imgs_np = np.array(pose_imgs).astype(np.float32) / 255
                return {
                    "ui": {"POSE_JSON": [json.dumps(pose_scaled, indent=4)]},
                    "result": (torch.from_numpy(pose_imgs_np), pose_scaled, json.dumps(pose_scaled, indent=4))
                }
        elif POSE_KEYPOINT is not None:
            pose_source = json.dumps(POSE_KEYPOINT, indent=4).replace("'", '"').replace('None', '[]')
            if enable_confidence_smoothing or enable_outlier_suppression:
                pose_source = _prepare_pose_source(
                    pose_source,
                    enable_confidence_smoothing,
                    confidence_smoothing_alpha,
                    enable_outlier_suppression,
                    outlier_jump_threshold,
                    confidence_floor,
                )
            hands_scalelist, body_scalelist, head_scalelist, overall_scalelist = extend_scalelist(
                scalelist_behavior, pose_source, hands_scale, body_scale, head_scale, overall_scale,
                match_scalelist_method, only_scale_pose_index)
            normalized_pose_json = pose_normalized(pose_source)
            pose_imgs, pose_scaled = draw_pose_json(
                normalized_pose_json, resolution_x, show_body, show_face, show_hands,
                pose_marker_size, face_marker_size, hand_marker_size, hands_scalelist, body_scalelist,
                head_scalelist, overall_scalelist)
            if pose_imgs:
                pose_imgs_np = np.array(pose_imgs).astype(np.float32) / 255
                return {
                    "ui": {"POSE_JSON": [json.dumps(pose_scaled, indent=4)]},
                    "result": (torch.from_numpy(pose_imgs_np), pose_scaled, json.dumps(pose_scaled, indent=4))
                }

        return super().load_pose(
            show_body, show_face, show_hands, resolution_x, pose_marker_size, face_marker_size,
            hand_marker_size, hands_scale, body_scale, head_scale, overall_scale, scalelist_behavior,
            match_scalelist_method, only_scale_pose_index, POSE_JSON, POSE_KEYPOINT)
