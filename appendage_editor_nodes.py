import json
import copy
import math
import torch
import numpy as np
from .util import scale

class AppendageEditorNode10:
    @staticmethod
    def normalize_scale_parameter(scale_param, target_length, behavior):
        """
        Normalize a scale parameter to a list of the target length.

        Args:
            scale_param: Either a single float or list of floats
            target_length: Desired length of output list
            behavior: "truncate", "loop", or "repeat"

        Returns:
            List of floats with length determined by behavior
        """
        # Convert single value to list
        if not isinstance(scale_param, (list, tuple)):
            scale_list = [scale_param]
        else:
            scale_list = list(scale_param)

        if len(scale_list) == target_length:
            return scale_list

        if behavior == "truncate":
            return scale_list[:target_length]
        elif behavior == "loop":
            if len(scale_list) == 0:
                return [1.0] * target_length
            result = []
            for i in range(target_length):
                result.append(scale_list[i % len(scale_list)])
            return result
        elif behavior == "repeat":
            if len(scale_list) == 0:
                return [1.0] * target_length
            if len(scale_list) >= target_length:
                return scale_list[:target_length]
            else:
                result = scale_list[:]
                last_value = scale_list[-1]
                while len(result) < target_length:
                    result.append(last_value)
                return result
        else:
            raise ValueError(f"Unknown behavior: {behavior}")

    @staticmethod
    def determine_output_length(scale_params, pose_count, behavior):
        """
        Determine the output length based on scale parameters and behavior.
        """
        # Get all list lengths
        lengths = [pose_count]
        for param in scale_params:
            if isinstance(param, (list, tuple)):
                lengths.append(len(param))

        if behavior == "truncate":
            return min(lengths)
        else:  # loop or repeat
            return max(lengths)

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "POSE_KEYPOINT": ("POSE_KEYPOINT",),
                "appendage_type": ([
                    "left_upper_arm", "left_forearm", "left_full_arm",
                    "right_upper_arm", "right_forearm", "right_full_arm",
                    "left_upper_leg", "left_lower_leg", "left_full_leg",
                    "right_upper_leg", "right_lower_leg", "right_full_leg",
                    "left_hand", "right_hand", "left_foot", "right_foot",
                    "torso", "shoulders", "head", "face", "full_person"
                ], {
                    "default": "left_upper_arm"
                }),
            },
            "optional": {
                "scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.05
                }),
                "x_offset": ("FLOAT", {
                    "default": 0.0,
                    "min": -2.0,
                    "max": 2.0,
                    "step": 0.01
                }),
                "y_offset": ("FLOAT", {
                    "default": 0.0,
                    "min": -2.0,
                    "max": 2.0,
                    "step": 0.01
                }),
                "rotation": ("FLOAT", {
                    "default": 0.0,
                    "min": -180.0,
                    "max": 180.0,
                    "step": 1.0
                }),
                "bidirectional_scale": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "If true, scales in both directions from pivot. If false, only scales away from body to prevent cannibalizing adjacent parts."
                }),
                "person_index": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 100,
                    "tooltip": "Person to edit (-1 for all people)"
                }),
                "list_mismatch_behavior": (["truncate", "loop", "repeat"], {"default": "loop", "tooltip": "Truncate: Truncate the list to the shortest length. Loop: Loop the list to the longest length. Repeat: Repeat the list to the longest length."}),
            },
        }

    RETURN_NAMES = ("POSE_KEYPOINT",)
    RETURN_TYPES = ("POSE_KEYPOINT",)
    FUNCTION = "edit_appendage"
    CATEGORY = "ultimate-openpose"

    def edit_appendage(self, POSE_KEYPOINT, appendage_type, scale=1.0, x_offset=0.0, y_offset=0.0, rotation=0.0, bidirectional_scale=False, person_index=-1, list_mismatch_behavior="loop"):
        if POSE_KEYPOINT is None:
            return (None,)

        # Deep copy to avoid modifying the original
        pose_data = copy.deepcopy(POSE_KEYPOINT)
        if not isinstance(pose_data, list):
            pose_data = [pose_data]

        pose_count = len(pose_data)

        # Normalize scale parameters to handle lists vs single floats using the original node's methods
        scale_params = [scale, x_offset, y_offset, rotation]
        output_length = self.determine_output_length(scale_params, pose_count, list_mismatch_behavior)

        scale_list = self.normalize_scale_parameter(scale, output_length, list_mismatch_behavior)
        x_offset_list = self.normalize_scale_parameter(x_offset, output_length, list_mismatch_behavior)
        y_offset_list = self.normalize_scale_parameter(y_offset, output_length, list_mismatch_behavior)
        rotation_list = self.normalize_scale_parameter(rotation, output_length, list_mismatch_behavior)

        # Process each frame with its corresponding parameter values
        output_pose_data = []

        for i in range(output_length):
            # Get the pose data for this index
            pose_idx = i if i < pose_count else pose_count - 1
            if list_mismatch_behavior == "loop" and pose_count > 0:
                pose_idx = i % pose_count

            # Get current frame and parameter values
            current_frame = copy.deepcopy(pose_data[pose_idx])
            current_scale = scale_list[i]
            current_x_offset = x_offset_list[i]
            current_y_offset = y_offset_list[i]
            current_rotation = rotation_list[i]

            # Apply transformations to this frame
            if 'people' in current_frame:
                people_to_edit = range(len(current_frame['people'])) if person_index == -1 else [person_index]

                for person_idx in people_to_edit:
                    if person_idx >= len(current_frame['people']):
                        continue

                    person = current_frame['people'][person_idx]

                    if appendage_type in ["left_hand", "right_hand"]:
                        self._edit_hand_appendage(person, appendage_type, current_scale, current_x_offset, current_y_offset, current_rotation, bidirectional_scale)
                    elif appendage_type == "face":
                        self._edit_face_appendage(person, current_scale, current_x_offset, current_y_offset, current_rotation, bidirectional_scale)
                    elif appendage_type == "full_person":
                        self._edit_full_person(person, current_scale, current_x_offset, current_y_offset, current_rotation, bidirectional_scale)
                    else:
                        self._edit_body_appendage(person, appendage_type, current_scale, current_x_offset, current_y_offset, current_rotation, bidirectional_scale)

            output_pose_data.append(current_frame)

        return (output_pose_data,)

    def _edit_hand_appendage(self, person, appendage_type, scale_factor, x_offset, y_offset, rotation, bidirectional_scale):
        """Edit hand appendages using hand keypoints."""
        keypoint_field = "hand_left_keypoints_2d" if appendage_type == "left_hand" else "hand_right_keypoints_2d"

        if keypoint_field not in person or not person[keypoint_field]:
            return

        keypoints = person[keypoint_field]

        # Use wrist (first point) as pivot for hands
        if len(keypoints) >= 3 and keypoints[2] > 0:
            pivot = [keypoints[0], keypoints[1]]
        else:
            # Calculate center of mass if wrist not available
            pivot = self._calculate_center_of_mass(keypoints)
            if pivot is None:
                return

        # Apply transformations
        new_keypoints = self._apply_transformations(keypoints, scale_factor, x_offset, y_offset, rotation, pivot, bidirectional_scale, pivot_index=0)
        person[keypoint_field] = new_keypoints

    def _edit_body_appendage(self, person, appendage_type, scale_factor, x_offset, y_offset, rotation, bidirectional_scale):
        """Edit body appendages (arms, legs, feet) using body pose keypoints."""
        if 'pose_keypoints_2d' not in person or not person['pose_keypoints_2d']:
            return

        keypoints = person['pose_keypoints_2d']

        # Get keypoint indices for the specific appendage
        appendage_indices, pivot_index = self._get_appendage_indices(appendage_type)
        if not appendage_indices:
            return

        # Calculate pivot point for the appendage
        pivot = self._calculate_appendage_pivot(keypoints, appendage_indices, pivot_index)
        if pivot is None:
            return

        # Apply transformations only to the appendage keypoints
        new_keypoints = keypoints[:]

        for i in range(0, len(keypoints), 3):
            keypoint_idx = i // 3
            if keypoint_idx in appendage_indices and len(keypoints) > i+2:
                x, y, conf = keypoints[i], keypoints[i+1], keypoints[i+2]

                if conf > 0:
                    # Apply rotation
                    if rotation != 0.0:
                        rad = math.radians(rotation)
                        cos_r, sin_r = math.cos(rad), math.sin(rad)
                        rel_x, rel_y = x - pivot[0], y - pivot[1]
                        x = rel_x * cos_r - rel_y * sin_r + pivot[0]
                        y = rel_x * sin_r + rel_y * cos_r + pivot[1]

                    # Apply scaling with directional control
                    if scale_factor != 1.0:
                        if bidirectional_scale:
                            scaled_point = scale([x, y], scale_factor, pivot)
                            x, y = scaled_point[0], scaled_point[1]
                        else:
                            # Unidirectional scaling - only scale away from body
                            x, y = self._apply_unidirectional_scale([x, y], scale_factor, pivot, keypoint_idx, pivot_index)

                    # Apply offset
                    x += x_offset
                    y += y_offset

                new_keypoints[i] = x
                new_keypoints[i+1] = y

        person['pose_keypoints_2d'] = new_keypoints

    def _edit_face_appendage(self, person, scale_factor, x_offset, y_offset, rotation, bidirectional_scale):
        """Edit facial keypoints stored in face_keypoints_2d."""
        keypoint_field = "face_keypoints_2d"

        if keypoint_field not in person or not person[keypoint_field]:
            return

        keypoints = person[keypoint_field]
        pivot = self._calculate_center_of_mass(keypoints)
        if pivot is None:
            return

        new_keypoints = self._apply_transformations(keypoints, scale_factor, x_offset, y_offset, rotation, pivot, bidirectional_scale)
        person[keypoint_field] = new_keypoints

    def _edit_full_person(self, person, scale_factor, x_offset, y_offset, rotation, bidirectional_scale):
        """Edit all available keypoints for a person (body, face, and hands)."""
        pivot = self._calculate_global_pivot(person)
        if pivot is None:
            return

        for field in ["pose_keypoints_2d", "face_keypoints_2d", "hand_left_keypoints_2d", "hand_right_keypoints_2d"]:
            if field not in person or not person[field]:
                continue

            keypoints = person[field]
            new_keypoints = self._apply_transformations(keypoints, scale_factor, x_offset, y_offset, rotation, pivot, bidirectional_scale)
            person[field] = new_keypoints

    def _get_appendage_indices(self, appendage_type):
        """Get OpenPose keypoint indices for specific appendages and their pivot points."""
        # COCO 18-keypoint format (0-based) used by ComfyUI ControlNet Aux OpenPose Pose node:
        # 0: Nose, 1: Neck, 2: RShoulder, 3: RElbow, 4: RWrist, 5: LShoulder, 6: LElbow, 7: LWrist,
        # 8: RHip, 9: RKnee, 10: RAnkle, 11: LHip, 12: LKnee, 13: LAnkle, 14: REye, 15: LEye, 16: REar, 17: LEar

        appendage_map = {
            # Arms - COCO format
            "left_upper_arm": ([5, 6], 5),          # LShoulder, LElbow (pivot: shoulder)
            "left_forearm": ([6, 7], 6),            # LElbow, LWrist (pivot: elbow)
            "left_full_arm": ([5, 6, 7], 5),        # LShoulder, LElbow, LWrist (pivot: shoulder)
            "right_upper_arm": ([2, 3], 2),         # RShoulder, RElbow (pivot: shoulder)
            "right_forearm": ([3, 4], 3),           # RElbow, RWrist (pivot: elbow)
            "right_full_arm": ([2, 3, 4], 2),       # RShoulder, RElbow, RWrist (pivot: shoulder)

            # Legs - COCO format (FIXED!)
            "left_upper_leg": ([11, 12], 11),       # LHip, LKnee (pivot: hip)
            "left_lower_leg": ([12, 13], 12),       # LKnee, LAnkle (pivot: knee) - FIXED: was [13,14] which was LAnkle,REye!
            "left_full_leg": ([11, 12, 13], 11),    # LHip, LKnee, LAnkle (pivot: hip)
            "right_upper_leg": ([8, 9], 8),         # RHip, RKnee (pivot: hip)
            "right_lower_leg": ([9, 10], 9),        # RKnee, RAnkle (pivot: knee)
            "right_full_leg": ([8, 9, 10], 8),      # RHip, RKnee, RAnkle (pivot: hip)

            # Feet - COCO format (no foot keypoints in COCO, use ankle only)
            "left_foot": ([13], 13),                # LAnkle only (pivot: ankle)
            "right_foot": ([10], 10),               # RAnkle only (pivot: ankle)

            # Torso and Shoulders - COCO format
            "torso": ([1, 2, 5, 8, 11], 1),         # Neck, RShoulder, LShoulder, RHip, LHip (pivot: neck)
            "shoulders": ([2, 5], 1),               # RShoulder, LShoulder (pivot: neck)
            "head": ([0, 1, 14, 15, 16, 17], 1),    # Nose, Neck, Eyes, Ears (pivot: neck)
        }

        result = appendage_map.get(appendage_type, ([], None))
        return result[0], result[1]

    def _calculate_appendage_pivot(self, keypoints, appendage_indices, pivot_index):
        """Calculate pivot point for body appendage using specified pivot index."""
        if pivot_index is not None:
            # Use specific pivot point (e.g., shoulder for upper arm, elbow for forearm)
            i = pivot_index * 3
            if len(keypoints) > i+2 and keypoints[i+2] > 0:
                return [keypoints[i], keypoints[i+1]]

        # Fallback to center of mass if pivot point not available
        valid_points = []
        for idx in appendage_indices:
            i = idx * 3
            if len(keypoints) > i+2 and keypoints[i+2] > 0:
                valid_points.append([keypoints[i], keypoints[i+1]])

        if not valid_points:
            return None

        pivot_x = sum(p[0] for p in valid_points) / len(valid_points)
        pivot_y = sum(p[1] for p in valid_points) / len(valid_points)
        return [pivot_x, pivot_y]


class AppendageEditorNode10V2(AppendageEditorNode10):
    @classmethod
    def INPUT_TYPES(cls):
        base = copy.deepcopy(super().INPUT_TYPES())
        base["optional"]["limit_scale_to_canvas"] = ("BOOLEAN", {
            "default": True,
            "tooltip": "Prevent scaling beyond the canvas boundary by clamping growth when feet would exit the frame."
        })
        base["optional"]["only_scale_up"] = ("BOOLEAN", {
            "default": False,
            "tooltip": "When enabled, avoid shrink operations and shift the full body upward so feet remain grounded during scaling."
        })
        return base

    def edit_appendage(self, POSE_KEYPOINT, appendage_type, scale=1.0, x_offset=0.0, y_offset=0.0, rotation=0.0,
                       bidirectional_scale=False, person_index=-1, list_mismatch_behavior="loop",
                       limit_scale_to_canvas=True, only_scale_up=False):
        if POSE_KEYPOINT is None:
            return (None,)

        pose_data = copy.deepcopy(POSE_KEYPOINT)
        if not isinstance(pose_data, list):
            pose_data = [pose_data]

        pose_count = len(pose_data)

        scale_params = [scale, x_offset, y_offset, rotation]
        output_length = self.determine_output_length(scale_params, pose_count, list_mismatch_behavior)

        scale_list = self.normalize_scale_parameter(scale, output_length, list_mismatch_behavior)
        x_offset_list = self.normalize_scale_parameter(x_offset, output_length, list_mismatch_behavior)
        y_offset_list = self.normalize_scale_parameter(y_offset, output_length, list_mismatch_behavior)
        rotation_list = self.normalize_scale_parameter(rotation, output_length, list_mismatch_behavior)
        limit_list = self.normalize_scale_parameter(limit_scale_to_canvas, output_length, list_mismatch_behavior)
        only_up_list = self.normalize_scale_parameter(only_scale_up, output_length, list_mismatch_behavior)

        output_pose_data = []

        for i in range(output_length):
            pose_idx = i if i < pose_count else pose_count - 1
            if list_mismatch_behavior == "loop" and pose_count > 0:
                pose_idx = i % pose_count

            current_frame = copy.deepcopy(pose_data[pose_idx])
            current_scale = scale_list[i]
            current_x_offset = x_offset_list[i]
            current_y_offset = y_offset_list[i]
            current_rotation = rotation_list[i]
            limit_flag = bool(limit_list[i])
            only_up_flag = bool(only_up_list[i])

            if 'people' in current_frame:
                people_to_edit = range(len(current_frame['people'])) if person_index == -1 else [person_index]
                canvas_height = current_frame.get('canvas_height') if isinstance(current_frame, dict) else None

                for person_idx_iter in people_to_edit:
                    if person_idx_iter >= len(current_frame['people']):
                        continue

                    person = current_frame['people'][person_idx_iter]
                    original_person_state = copy.deepcopy(person)
                    original_bottom = self._get_person_bottom(original_person_state)

                    effective_scale = current_scale
                    if only_up_flag and isinstance(effective_scale, (int, float)) and effective_scale < 1.0:
                        effective_scale = 1.0

                    if (limit_flag and canvas_height is not None and isinstance(effective_scale, (int, float))
                            and effective_scale > 1.0):
                        effective_scale = self._limit_scale_for_canvas(
                            original_person_state,
                            appendage_type,
                            effective_scale,
                            current_x_offset,
                            current_y_offset,
                            current_rotation,
                            bidirectional_scale,
                            only_up_flag,
                            canvas_height,
                            original_bottom,
                        )

                    self._apply_appendage_edit(
                        person,
                        appendage_type,
                        effective_scale,
                        current_x_offset,
                        current_y_offset,
                        current_rotation,
                        bidirectional_scale,
                    )

                    if only_up_flag and original_bottom is not None:
                        self._apply_only_scale_up_shift(person, original_bottom)

                    if limit_flag and canvas_height is not None:
                        self._enforce_canvas_bounds(person, canvas_height)

            output_pose_data.append(current_frame)

        return (output_pose_data,)

    def _apply_appendage_edit(self, person, appendage_type, scale_factor, x_offset, y_offset, rotation, bidirectional_scale):
        if appendage_type in ["left_hand", "right_hand"]:
            self._edit_hand_appendage(person, appendage_type, scale_factor, x_offset, y_offset, rotation, bidirectional_scale)
        elif appendage_type == "face":
            self._edit_face_appendage(person, scale_factor, x_offset, y_offset, rotation, bidirectional_scale)
        elif appendage_type == "full_person":
            self._edit_full_person(person, scale_factor, x_offset, y_offset, rotation, bidirectional_scale)
        else:
            self._edit_body_appendage(person, appendage_type, scale_factor, x_offset, y_offset, rotation, bidirectional_scale)

    def _limit_scale_for_canvas(self, person_snapshot, appendage_type, target_scale, x_offset, y_offset, rotation,
                               bidirectional_scale, only_scale_up, canvas_height, original_bottom):
        if target_scale <= 1.0 or canvas_height is None:
            return target_scale

        if original_bottom is None:
            original_bottom = self._get_person_bottom(person_snapshot)

        if original_bottom is None:
            return target_scale

        bottom_at_target, _ = self._simulate_bottom(
            person_snapshot,
            appendage_type,
            target_scale,
            x_offset,
            y_offset,
            rotation,
            bidirectional_scale,
            only_scale_up,
            original_bottom,
        )

        if bottom_at_target is None or bottom_at_target <= canvas_height:
            return target_scale

        baseline_scale = 1.0
        bottom_at_baseline, _ = self._simulate_bottom(
            person_snapshot,
            appendage_type,
            baseline_scale,
            x_offset,
            y_offset,
            rotation,
            bidirectional_scale,
            only_scale_up,
            original_bottom,
        )

        if bottom_at_baseline is None or bottom_at_baseline > canvas_height:
            return baseline_scale

        low = baseline_scale
        high = target_scale
        best = baseline_scale

        for _ in range(25):
            mid = (low + high) / 2.0
            bottom_mid, _ = self._simulate_bottom(
                person_snapshot,
                appendage_type,
                mid,
                x_offset,
                y_offset,
                rotation,
                bidirectional_scale,
                only_scale_up,
                original_bottom,
            )

            if bottom_mid is None:
                best = mid
                low = mid
                continue

            if bottom_mid <= canvas_height:
                best = mid
                low = mid
            else:
                high = mid

        return best

    def _simulate_bottom(self, person_snapshot, appendage_type, scale_factor, x_offset, y_offset, rotation, bidirectional_scale,
                         only_scale_up, original_bottom):
        simulated = self._simulate_basic_transform(
            person_snapshot,
            appendage_type,
            scale_factor,
            x_offset,
            y_offset,
            rotation,
            bidirectional_scale,
        )

        bottom_after = self._get_person_bottom(simulated)
        shift = 0.0

        if only_scale_up and original_bottom is not None and bottom_after is not None and bottom_after > original_bottom:
            shift = original_bottom - bottom_after
            self._apply_vertical_shift(simulated, shift)
            bottom_after = self._get_person_bottom(simulated)

        return bottom_after, shift

    def _simulate_basic_transform(self, person_snapshot, appendage_type, scale_factor, x_offset, y_offset, rotation,
                                  bidirectional_scale):
        simulated = copy.deepcopy(person_snapshot)
        self._apply_appendage_edit(simulated, appendage_type, scale_factor, x_offset, y_offset, rotation, bidirectional_scale)
        return simulated

    def _apply_only_scale_up_shift(self, person, original_bottom):
        new_bottom = self._get_person_bottom(person)
        if new_bottom is None or original_bottom is None:
            return

        if new_bottom > original_bottom:
            shift = original_bottom - new_bottom
            self._apply_vertical_shift(person, shift)

    def _enforce_canvas_bounds(self, person, canvas_height):
        if canvas_height is None:
            return

        bottom = self._get_person_bottom(person)
        if bottom is None:
            return

        if bottom > canvas_height:
            shift = canvas_height - bottom
            self._apply_vertical_shift(person, shift)

    def _get_person_bottom(self, person):
        max_y = None
        for field in ["pose_keypoints_2d", "face_keypoints_2d", "hand_left_keypoints_2d", "hand_right_keypoints_2d"]:
            keypoints = person.get(field)
            if not keypoints:
                continue

            for i in range(1, len(keypoints), 3):
                if len(keypoints) > i + 1:
                    conf = keypoints[i + 1]
                    if conf > 0:
                        y_val = keypoints[i]
                        if max_y is None or y_val > max_y:
                            max_y = y_val

        return max_y

    def _apply_vertical_shift(self, person, shift):
        if shift == 0.0:
            return

        for field in ["pose_keypoints_2d", "face_keypoints_2d", "hand_left_keypoints_2d", "hand_right_keypoints_2d"]:
            keypoints = person.get(field)
            if not keypoints:
                continue

            for i in range(1, len(keypoints), 3):
                if len(keypoints) > i + 1:
                    conf = keypoints[i + 1]
                    if conf > 0:
                        keypoints[i] += shift

    def _apply_unidirectional_scale(self, point, scale_factor, pivot, keypoint_idx, pivot_index):
        """Apply scaling only in the direction away from the body/pivot."""
        x, y = point

        if keypoint_idx == pivot_index:
            # Don't scale the pivot point itself
            return x, y

        # Calculate direction vector from pivot to point
        dx = x - pivot[0]
        dy = y - pivot[1]

        # Scale only the distance, keeping direction
        distance = math.sqrt(dx*dx + dy*dy)
        if distance > 0:
            new_distance = distance * scale_factor
            scale_ratio = new_distance / distance

            new_x = pivot[0] + dx * scale_ratio
            new_y = pivot[1] + dy * scale_ratio
            return new_x, new_y

        return x, y

    def _calculate_center_of_mass(self, keypoints):
        """Calculate center of mass from valid keypoints."""
        valid_points = []
        for i in range(0, len(keypoints), 3):
            if len(keypoints) > i+2 and keypoints[i+2] > 0:
                valid_points.append([keypoints[i], keypoints[i+1]])

        if not valid_points:
            return None

        pivot_x = sum(p[0] for p in valid_points) / len(valid_points)
        pivot_y = sum(p[1] for p in valid_points) / len(valid_points)
        return [pivot_x, pivot_y]

    def _apply_transformations(self, keypoints, scale_factor, x_offset, y_offset, rotation, pivot, bidirectional_scale, pivot_index=None):
        """Apply transformations to all keypoints."""
        new_keypoints = []
        for i in range(0, len(keypoints), 3):
            if len(keypoints) > i+2:
                x, y, conf = keypoints[i], keypoints[i+1], keypoints[i+2]

                if conf > 0:
                    # Apply rotation
                    if rotation != 0.0:
                        rad = math.radians(rotation)
                        cos_r, sin_r = math.cos(rad), math.sin(rad)
                        rel_x, rel_y = x - pivot[0], y - pivot[1]
                        x = rel_x * cos_r - rel_y * sin_r + pivot[0]
                        y = rel_x * sin_r + rel_y * cos_r + pivot[1]

                    # Apply scaling
                    if scale_factor != 1.0:
                        if bidirectional_scale:
                            scaled_point = scale([x, y], scale_factor, pivot)
                            x, y = scaled_point[0], scaled_point[1]
                        else:
                            # For hands and other appendages, use unidirectional scaling from pivot
                            x, y = self._apply_unidirectional_scale([x, y], scale_factor, pivot, i//3, pivot_index)

                    # Apply offset
                    x += x_offset
                    y += y_offset

                new_keypoints.extend([x, y, conf])

        return new_keypoints

    def _calculate_global_pivot(self, person):
        """Calculate a pivot using all available keypoints for a person."""
        aggregated_points = []
        for field in ["pose_keypoints_2d", "face_keypoints_2d", "hand_left_keypoints_2d", "hand_right_keypoints_2d"]:
            keypoints = person.get(field)
            if not keypoints:
                continue

            for i in range(0, len(keypoints), 3):
                if len(keypoints) > i + 2 and keypoints[i + 2] > 0:
                    aggregated_points.append([keypoints[i], keypoints[i + 1]])

        if not aggregated_points:
            return None

        pivot_x = sum(p[0] for p in aggregated_points) / len(aggregated_points)
        pivot_y = sum(p[1] for p in aggregated_points) / len(aggregated_points)
        return [pivot_x, pivot_y]
