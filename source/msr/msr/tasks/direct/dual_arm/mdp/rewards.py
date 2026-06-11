# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import os
from bisect import bisect_left
from collections import deque

import numpy as np
import torch
from typing import TYPE_CHECKING, Any

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms, quat_error_magnitude, quat_mul

try:
    import cv2 as _cv2
    _HAS_CV2 = True
except ImportError:
    _cv2 = None
    _HAS_CV2 = False

try:
    from skimage.morphology import skeletonize as _skimage_skeletonize
    _HAS_SKIMAGE = True
except Exception:
    _skimage_skeletonize = None
    _HAS_SKIMAGE = False

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


CONNECTIVITY_CONNECTED = 1
CONNECTIVITY_NOT_CONNECTED = 2


def _resolve_debug_log_path(env: ManagerBasedRLEnv, debug_log_file: str | None = None) -> str:
    candidate_log_dirs: list[str] = []
    for candidate in (
        getattr(env, "log_dir", None),
        getattr(getattr(env, "cfg", None), "log_dir", None),
        getattr(getattr(env, "unwrapped", None), "log_dir", None),
        getattr(getattr(getattr(env, "unwrapped", None), "cfg", None), "log_dir", None),
    ):
        if isinstance(candidate, str) and len(candidate) > 0:
            candidate_log_dirs.append(candidate)

    base_log_dir = candidate_log_dirs[0] if len(candidate_log_dirs) > 0 else None

    if debug_log_file:
        if os.path.isabs(debug_log_file):
            return debug_log_file
        if base_log_dir is not None:
            return os.path.join(base_log_dir, debug_log_file)
        return os.path.abspath(debug_log_file)

    if base_log_dir is not None:
        return os.path.join(base_log_dir, "vessel_reward_debug.txt")
    return "/tmp/vessel_reward_debug.txt"


def _emit_debug_line(
    env: ManagerBasedRLEnv,
    line: str,
    write_to_file: bool = False,
    debug_log_file: str | None = None,
    write_to_terminal: bool = False,
):
    if write_to_terminal:
        print(line, flush=True)
    if not write_to_file:
        return
    try:
        file_path = _resolve_debug_log_path(env, debug_log_file)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_rgba(color: Any) -> tuple[int, int, int, int] | None:
    if isinstance(color, (list, tuple)) and len(color) >= 3:
        try:
            r = int(color[0])
            g = int(color[1])
            b = int(color[2])
            a = int(color[3]) if len(color) > 3 else 255
            return (r, g, b, a)
        except (TypeError, ValueError):
            return None
    if isinstance(color, dict):
        keys = {k.lower(): v for k, v in color.items()}
        if all(k in keys for k in ("r", "g", "b")):
            try:
                r = int(keys["r"])
                g = int(keys["g"])
                b = int(keys["b"])
                a = int(keys.get("a", 255))
                return (r, g, b, a)
            except (TypeError, ValueError):
                return None
    return None


def _extract_label_text(meta: Any) -> str:
    if isinstance(meta, str):
        return meta
    if isinstance(meta, dict):
        preferred_keys = ("class", "label", "name", "semanticLabel", "semantic_label", "data")
        for key in preferred_keys:
            if key in meta and isinstance(meta[key], str):
                return meta[key]
        text_parts = [str(v) for v in meta.values() if isinstance(v, str)]
        return "|".join(text_parts)
    if isinstance(meta, (list, tuple)):
        text_parts = [str(v) for v in meta if isinstance(v, str)]
        return "|".join(text_parts)
    return str(meta)


def _extract_id_to_labels(sem_info: Any) -> dict[int, str]:
    if not isinstance(sem_info, dict):
        return {}
    result: dict[int, str] = {}
    for key, value in sem_info.items():
        if not isinstance(value, dict):
            continue
        key_lower = str(key).lower()
        if "label" not in key_lower and "semantic" not in key_lower:
            continue
        for raw_id, meta in value.items():
            sem_id = _safe_int(raw_id)
            if sem_id is None:
                continue
            label_text = _extract_label_text(meta)
            if label_text:
                result[sem_id] = label_text
    return result


def _extract_id_to_colors(sem_info: Any) -> dict[int, tuple[int, int, int, int]]:
    if not isinstance(sem_info, dict):
        return {}
    result: dict[int, tuple[int, int, int, int]] = {}
    for key, value in sem_info.items():
        if not isinstance(value, dict):
            continue
        if "color" not in str(key).lower():
            continue
        for raw_id, raw_color in value.items():
            sem_id = _safe_int(raw_id)
            if sem_id is None:
                continue
            rgba = _normalize_rgba(raw_color)
            if rgba is not None:
                result[sem_id] = rgba
    return result


def get_semantic_class_color_mapping(
    env: ManagerBasedRLEnv,
    camera_cfg_name: str = "camera",
) -> dict[str, tuple[int, int, int, int] | None]:
    """Return best-effort mapping from semantic class label to RGBA color.

    Priority:
    1) Explicit camera config mapping (`semantic_segmentation_mapping`)
    2) Runtime `semantic_segmentation` info (if it provides ID->label and ID->color)

    Returns `None` for labels that are known but have no resolved color.
    """
    camera = env.scene[camera_cfg_name]
    sem_info = camera.data.info.get("semantic_segmentation", {})

    label_to_color: dict[str, tuple[int, int, int, int] | None] = {}

    cfg_mapping = getattr(camera.cfg, "semantic_segmentation_mapping", {}) or {}
    if isinstance(cfg_mapping, dict):
        for raw_label, raw_color in cfg_mapping.items():
            label = str(raw_label)
            if ":" in label:
                label = label.split(":", 1)[1]
            rgba = _normalize_rgba(raw_color)
            if rgba is not None:
                label_to_color[label] = rgba

    id_to_labels = _extract_id_to_labels(sem_info)
    id_to_colors = _extract_id_to_colors(sem_info)
    for sem_id, label_text in id_to_labels.items():
        label_to_color.setdefault(label_text, id_to_colors.get(sem_id))

    return label_to_color


def _semantic_label_mask_from_info(
    sem_data: torch.Tensor,
    sem_info: Any,
    target_label: str,
) -> torch.Tensor | None:
    id_to_labels = _extract_id_to_labels(sem_info)
    if not id_to_labels:
        return None

    target_lower = target_label.lower()
    target_ids = [sem_id for sem_id, text in id_to_labels.items() if target_lower in text.lower()]
    if not target_ids:
        return None

    # Non-colorized semantic segmentation: int32 IDs with shape [..., 1]
    if sem_data.shape[-1] == 1 and sem_data.dtype in (torch.int16, torch.int32, torch.int64):
        semantic_ids = sem_data[..., 0].to(dtype=torch.int64)
        mask = torch.zeros_like(semantic_ids, dtype=torch.bool)
        for sem_id in target_ids:
            mask |= semantic_ids == sem_id
        return mask

    # Colorized semantic segmentation: need ID->color map from info
    id_to_colors = _extract_id_to_colors(sem_info)
    target_colors = [id_to_colors[sem_id] for sem_id in target_ids if sem_id in id_to_colors]
    if not target_colors:
        return None

    rgb_data = sem_data[..., :3].to(dtype=torch.int16)
    mask = torch.zeros(rgb_data.shape[:-1], dtype=torch.bool, device=rgb_data.device)
    for r, g, b, _ in target_colors:
        mask |= (rgb_data[..., 0] == r) & (rgb_data[..., 1] == g) & (rgb_data[..., 2] == b)
    return mask


def _semantic_label_color_area_stats_env0(
    sem_data: torch.Tensor,
    sem_info: Any,
) -> list[tuple[str, tuple[int, int, int, int] | None, float]]:
    """Compute per-label area ratio on env-0 using semantic info when possible."""
    if sem_data.ndim != 4:
        return []

    id_to_labels = _extract_id_to_labels(sem_info)
    if not id_to_labels:
        return []

    total_pixels = float(sem_data.shape[1] * sem_data.shape[2])
    if total_pixels <= 0:
        return []

    id_to_colors = _extract_id_to_colors(sem_info)
    stats: list[tuple[str, tuple[int, int, int, int] | None, float]] = []

    if sem_data.shape[-1] == 1 and sem_data.dtype in (torch.int16, torch.int32, torch.int64):
        semantic_ids = sem_data[0, ..., 0].to(dtype=torch.int64)
        for sem_id, label_text in id_to_labels.items():
            ratio = float((semantic_ids == sem_id).sum().item() / total_pixels)
            if ratio > 0.0:
                stats.append((label_text, id_to_colors.get(sem_id), ratio))
        stats.sort(key=lambda x: x[2], reverse=True)
        return stats

    if sem_data.shape[-1] >= 3:
        rgb_data = sem_data[0, ..., :3].to(dtype=torch.int16)
        for sem_id, label_text in id_to_labels.items():
            rgba = id_to_colors.get(sem_id)
            if rgba is None:
                continue
            r, g, b, _ = rgba
            ratio = float(((rgb_data[..., 0] == r) & (rgb_data[..., 1] == g) & (rgb_data[..., 2] == b)).sum().item() / total_pixels)
            if ratio > 0.0:
                stats.append((label_text, rgba, ratio))
        stats.sort(key=lambda x: x[2], reverse=True)
        return stats

    return []


def position_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize tracking of the position error using L2-norm.

    The function computes the position error between the desired position (from the command) and the
    current position of the asset's body (in world frame). The position error is computed as the L2-norm
    of the difference between the desired and current positions.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_state_w[:, :3], asset.data.root_state_w[:, 3:7], des_pos_b)
    curr_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3]  # type: ignore
    return torch.norm(curr_pos_w - des_pos_w, dim=1)


def position_command_error_tanh(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward tracking of the position using the tanh kernel.

    The function computes the position error between the desired position (from the command) and the
    current position of the asset's body (in world frame) and maps it with a tanh kernel.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_state_w[:, :3], asset.data.root_state_w[:, 3:7], des_pos_b)
    curr_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3]  # type: ignore
    distance = torch.norm(curr_pos_w - des_pos_w, dim=1)
    return 1 - torch.tanh(distance / std)

def orientation_command_error_tanh(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward tracking of the orientation using the tanh kernel.

    The function computes the orientation error between the desired orientation (from the command) and the
    current orientation of the asset's body (in world frame) and maps it with a tanh kernel.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current orientations
    des_quat_b = command[:, 3:7]
    des_quat_w = quat_mul(asset.data.root_state_w[:, 3:7], des_quat_b)
    curr_quat_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], 3:7]  # type: ignore
    orientation_error = quat_error_magnitude(curr_quat_w, des_quat_w)
    return 1 - torch.tanh(orientation_error / std)

def orientation_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize tracking orientation error using shortest path.

    The function computes the orientation error between the desired orientation (from the command) and the
    current orientation of the asset's body (in world frame). The orientation error is computed as the shortest
    path between the desired and current orientations.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current orientations
    des_quat_b = command[:, 3:7]
    des_quat_w = quat_mul(asset.data.root_state_w[:, 3:7], des_quat_b)
    curr_quat_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], 3:7]  # type: ignore
    return quat_error_magnitude(curr_quat_w, des_quat_w)

def adaptive_action_smoothness(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg, 
    command_name: str,
    activation_distance: float = 0.1,  # 开始激活平滑限制的距离阈值
    max_weight: float = 1.0,           # 最大权重系数
    smoothness_std: float = 0.05       # 平滑过渡的标准差
) -> torch.Tensor:
    """
    距离自适应的动作平滑奖励函数。
    当机械臂接近目标时，逐渐增强对动作变化率的惩罚。
    
    Args:
        env: 环境实例
        asset_cfg: 资产配置
        command_name: 命令名称
        activation_distance: 开始激活平滑限制的距离阈值
        max_weight: 距离为0时的最大权重系数
        smoothness_std: 距离权重函数的平滑参数
    """
    # 获取资产和命令
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    
    # 计算当前位置与目标位置的距离
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(
        asset.data.root_state_w[:, :3], 
        asset.data.root_state_w[:, 3:7], 
        des_pos_b
    )
    curr_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3]
    distance = torch.norm(curr_pos_w - des_pos_w, dim=1)
    
    # 计算动作变化率
    if len(env.action_manager.action) > 1:
        action_diff = env.action_manager.action - env.action_manager.prev_action
        action_rate = torch.sum(action_diff**2, dim=1)
    else:
        action_rate = torch.zeros_like(distance)
    
    # 计算距离相关的权重系数
    # 使用高斯函数，距离越近权重越大
    distance_weight = max_weight * torch.exp(
        -((distance - 0.0) ** 2) / (2 * smoothness_std ** 2)
    )
    
    # 只在激活距离内应用惩罚
    active_mask = distance < activation_distance
    distance_weight = distance_weight * active_mask.float()
    
    # 返回加权动作率
    return distance_weight * action_rate


def vessel_semantic_coverage_reward(
    env: ManagerBasedRLEnv,
    camera_cfg_name: str = "camera",
    vessel_color: tuple = (25, 82, 255),  # Blue-ish Vessel color from observed semantic images
    color_tolerance: int = 10,  # Tolerance for color matching
    vessel_label: str = "vessel",
    prefer_semantic_info: bool = True,
    debug: bool = False,
    debug_every: int = 500,
    debug_to_file: bool = False,
    debug_log_file: str | None = None,
    debug_to_terminal: bool = False,
) -> torch.Tensor:
    """
    Reward based on the visible area of 'Vessel' class in semantic segmentation image.
    
    The reward is computed as the ratio of Vessel pixels (blue) to total image pixels.
    Higher coverage = higher reward, encouraging the robot to expose more vessel area.
    
    Args:
        env: The environment instance
        camera_cfg_name: Name of the camera sensor in the scene
        vessel_color: RGB color tuple for Vessel class (default blue)
        color_tolerance: Tolerance for color matching (in case of compression artifacts)
    
    Returns:
        torch.Tensor: Reward tensor with shape (num_envs,)
    """
    call_counter = getattr(env, "_vessel_reward_call_counter", 0) + 1
    setattr(env, "_vessel_reward_call_counter", call_counter)

    if debug:
        debug_counter = getattr(env, "_vessel_reward_debug_counter", 0) + 1
        setattr(env, "_vessel_reward_debug_counter", debug_counter)
        should_debug = debug_counter % max(1, debug_every) == 0
    else:
        debug_counter = getattr(env, "_vessel_reward_debug_counter", 0)
        should_debug = False

    try:
        # Get camera sensor from scene
        camera = env.scene[camera_cfg_name]
        
        # Get semantic segmentation output
        # The output shape is typically [num_envs, height, width, channels]
        sem_data = camera.data.output.get("semantic_segmentation")
        sem_info = camera.data.info.get("semantic_segmentation", {})
        
        if sem_data is None:
            # If semantic segmentation not available, return zero reward
            if should_debug:
                available_outputs = list(camera.data.output.keys())
                _emit_debug_line(
                    env,
                    f"[vessel_reward][debug] semantic_segmentation is None. "
                    f"camera='{camera_cfg_name}', available_outputs={available_outputs}"
                    ,
                    write_to_file=debug_to_file,
                    debug_log_file=debug_log_file,
                    write_to_terminal=debug_to_terminal,
                )
            return torch.zeros(env.num_envs, device=env.device)
        
        method = "color_fallback"
        vessel_mask = None

        # Prefer semantic-info based matching by label/ID when available.
        if prefer_semantic_info:
            vessel_mask = _semantic_label_mask_from_info(sem_data, sem_info, vessel_label)
            if vessel_mask is not None:
                method = "semantic_info"

        # Fallback: color matching (less robust across runs/mappings)
        if vessel_mask is None:
            if sem_data.shape[-1] == 4:
                rgb_data = sem_data[..., :3]  # [num_envs, H, W, 3]
            else:
                rgb_data = sem_data

            vessel_r, vessel_g, vessel_b = vessel_color
            r_match = torch.abs(rgb_data[..., 0].float() - vessel_r) < color_tolerance
            g_match = torch.abs(rgb_data[..., 1].float() - vessel_g) < color_tolerance
            b_match = torch.abs(rgb_data[..., 2].float() - vessel_b) < color_tolerance
            vessel_mask = r_match & g_match & b_match
        
        # Count vessel pixels per environment
        vessel_pixel_count = vessel_mask.sum(dim=(-1, -2)).float()  # [num_envs]
        
        # Total pixels per image
        total_pixels = vessel_mask.shape[-1] * vessel_mask.shape[-2]
        
        # Compute coverage ratio (0 to 1)
        coverage_ratio = vessel_pixel_count / total_pixels

        if should_debug:
            if sem_data.shape[-1] >= 3:
                rgb_sample = sem_data[0, ..., :3].reshape(-1, 3).to(dtype=torch.int32)
                unique_colors, counts = torch.unique(rgb_sample, dim=0, return_counts=True)
                top_k = min(8, unique_colors.shape[0])
                sorted_idx = torch.argsort(counts, descending=True)[:top_k]
                top_colors = unique_colors[sorted_idx].detach().cpu().tolist()
                top_counts = counts[sorted_idx].detach().cpu().tolist()
                top_colors_text = list(zip(top_colors, top_counts))
            else:
                top_colors_text = []

            mean_coverage = float(coverage_ratio.mean().item())
            env0_coverage = float(coverage_ratio[0].item())
            max_coverage = float(coverage_ratio.max().item())
            non_zero_envs = int((coverage_ratio > 0).sum().item())
            info_keys = list(sem_info.keys()) if isinstance(sem_info, dict) else []
            class_color_map = get_semantic_class_color_mapping(env, camera_cfg_name)
            vessel_related_items = [
                (k, v) for k, v in class_color_map.items() if vessel_label.lower() in str(k).lower()
            ]
            label_area_stats = _semantic_label_color_area_stats_env0(sem_data, sem_info)
            top_label_area_stats = [
                (label, rgba, round(ratio, 6)) for label, rgba, ratio in label_area_stats[:8]
            ]

            _emit_debug_line(
                env,
                f"[vessel_reward][debug] step={debug_counter}, camera='{camera_cfg_name}', "
                f"shape={tuple(sem_data.shape)}, dtype={sem_data.dtype}, "
                f"method={method}, vessel_label='{vessel_label}', "
                f"vessel_color={vessel_color}, tolerance={color_tolerance}, "
                f"env0={env0_coverage:.6f}, mean={mean_coverage:.6f}, "
                f"max={max_coverage:.6f}, non_zero_envs={non_zero_envs}/{env.num_envs}"
                ,
                write_to_file=debug_to_file,
                debug_log_file=debug_log_file,
                write_to_terminal=debug_to_terminal,
            )
            _emit_debug_line(
                env,
                f"[vessel_reward][debug] semantic_info_keys={info_keys[:10]}, "
                f"contains_vessel_text={'vessel' in str(sem_info).lower()}"
                ,
                write_to_file=debug_to_file,
                debug_log_file=debug_log_file,
                write_to_terminal=debug_to_terminal,
            )
            _emit_debug_line(
                env,
                f"[vessel_reward][debug] top_colors_env0={top_colors_text}",
                write_to_file=debug_to_file,
                debug_log_file=debug_log_file,
                write_to_terminal=debug_to_terminal,
            )
            _emit_debug_line(
                env,
                f"[vessel_reward][debug] vessel_label_color_candidates={vessel_related_items[:10]}",
                write_to_file=debug_to_file,
                debug_log_file=debug_log_file,
                write_to_terminal=debug_to_terminal,
            )
            _emit_debug_line(
                env,
                f"[vessel_reward][debug] label_color_area_ratio_env0(top)={top_label_area_stats}",
                write_to_file=debug_to_file,
                debug_log_file=debug_log_file,
                write_to_terminal=debug_to_terminal,
            )
        
        return coverage_ratio
        
    except (KeyError, AttributeError, ValueError, RuntimeError) as e:
        # Camera not found, data not available, or Warp buffer error (e.g. resolution too large)
        if should_debug:
            _emit_debug_line(
                env,
                f"[vessel_reward][debug] exception={type(e).__name__}: {e}. "
                f"camera='{camera_cfg_name}'"
                ,
                write_to_file=debug_to_file,
                debug_log_file=debug_log_file,
                write_to_terminal=debug_to_terminal,
            )
        return torch.zeros(env.num_envs, device=env.device)


def vessel_semantic_coverage_reward_tanh(
    env: ManagerBasedRLEnv,
    camera_cfg_name: str = "camera",
    vessel_color: tuple = (25, 82, 255),
    color_tolerance: int = 10,
    vessel_label: str = "vessel",
    prefer_semantic_info: bool = True,
    std: float = 0.1,  # Standard deviation for tanh scaling
) -> torch.Tensor:
    """
    Reward based on Vessel coverage with tanh shaping for smoother gradients.
    
    This applies a tanh transformation to the coverage ratio to provide
    smoother gradients when coverage is low.
    
    Args:
        env: The environment instance
        camera_cfg_name: Name of the camera sensor in the scene
        vessel_color: RGB color tuple for Vessel class
        color_tolerance: Tolerance for color matching
        std: Standard deviation for tanh scaling
    
    Returns:
        torch.Tensor: Shaped reward tensor with shape (num_envs,)
    """
    coverage = vessel_semantic_coverage_reward(
        env,
        camera_cfg_name,
        vessel_color,
        color_tolerance,
        vessel_label,
        prefer_semantic_info,
    )
    # Apply tanh shaping: higher coverage gives reward closer to 1
    return torch.tanh(coverage / std)


def gripper_edge_penalty(
    env: ManagerBasedRLEnv,
    camera_cfg_name: str = "camera",
    gripper_label: str = "robot,gripper",
    gripper_color: tuple = (0, 0, 255),
    color_tolerance: int = 10,
    prefer_semantic_info: bool = True,
    edge_safe_margin_px: float = 5.0,
) -> torch.Tensor:
    """Penalty when gripper mask gets too close to image boundary.

    For each environment, compute the minimum pixel distance ``d_min`` from the
    gripper mask to the image border and return ``-ReLU(d0 - d_min)`` where
    ``d0=edge_safe_margin_px``.
    """
    camera = env.scene[camera_cfg_name]
    sem_data = camera.data.output.get("semantic_segmentation")
    if sem_data is None:
        return torch.zeros(env.num_envs, device=env.device)

    sem_info = camera.data.info.get("semantic_segmentation", {})
    gripper_mask = _get_label_mask_batch(
        sem_data,
        sem_info,
        gripper_label,
        gripper_color,
        color_tolerance,
        prefer_semantic_info,
    )

    n, h, w = gripper_mask.shape
    safe_margin = max(float(edge_safe_margin_px), 0.0)
    penalty = torch.zeros(n, device=env.device)
    for i in range(n):
        ys, xs = torch.nonzero(gripper_mask[i], as_tuple=True)
        if ys.numel() == 0:
            continue

        dist_to_border = torch.stack(
            (
                xs.to(dtype=torch.float32),
                (w - 1 - xs).to(dtype=torch.float32),
                ys.to(dtype=torch.float32),
                (h - 1 - ys).to(dtype=torch.float32),
            ),
            dim=1,
        )
        d_min = torch.min(dist_to_border)
        penalty[i] = -torch.clamp(safe_margin - d_min, min=0.0)
    return penalty


def action_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize L2 norm of action."""
    return torch.sum(env.action_manager.action ** 2, dim=1)


def action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize L2 norm of action rate (difference between current and previous actions)."""
    action_diff = env.action_manager.action - env.action_manager.prev_action
    return torch.sum(action_diff ** 2, dim=1)


def joint_vel_l2_limited(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot_1"),
    clip: float = 2000.0,
    deadzone: float = 0.0,
    reset_grace_steps: int = 8,
    warmup_grace_steps: int = 100,
    ema_alpha: float = 0.9,
) -> torch.Tensor:
    """Joint-velocity penalty with clipping, reset grace and optional EMA smoothing.

    This is a safer alternative to plain ``joint_vel_l2`` when occasional reset spikes
    or scene settling can dominate the reward.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    penalty = torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)

    if deadzone > 0.0:
        penalty = torch.clamp(penalty - float(deadzone), min=0.0)
    if clip > 0.0:
        penalty = torch.clamp(penalty, max=float(clip))

    if reset_grace_steps > 0 and hasattr(env, "episode_length_buf"):
        reset_mask = env.episode_length_buf < int(reset_grace_steps)
        penalty = torch.where(reset_mask, torch.zeros_like(penalty), penalty)

    if warmup_grace_steps > 0 and hasattr(env, "_warmup_step_buf"):
        warmup_mask = getattr(env, "_warmup_step_buf") < int(warmup_grace_steps)
        penalty = torch.where(warmup_mask, torch.zeros_like(penalty), penalty)

    if 0.0 < ema_alpha < 1.0:
        state_shape = (env.num_envs,)
        if not hasattr(env, "_joint_vel_penalty_ema") or getattr(env, "_joint_vel_penalty_ema").shape != state_shape:
            env._joint_vel_penalty_ema = torch.zeros(state_shape, device=env.device)
        env._joint_vel_penalty_ema = ema_alpha * env._joint_vel_penalty_ema + (1.0 - ema_alpha) * penalty
        penalty = env._joint_vel_penalty_ema

    return penalty


def step_penalty(env: ManagerBasedRLEnv, value: float = 1.0) -> torch.Tensor:
    """Constant per-step penalty magnitude."""
    return torch.full((env.num_envs,), float(value), device=env.device)


def end_effector_workspace_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot_1", body_names="psm_tool_tip_Link"),
    x_bounds: tuple[float, float] = (-0.15, 0.15),
    y_bounds: tuple[float, float] = (0.2, 0.5),
    z_bounds: tuple[float, float] = (0.01, 0.2),
    squared: bool = True,
) -> torch.Tensor:
    """Penalize end-effector position when it goes outside a 3D workspace box.

    The end-effector position is read from ``asset.data.body_state_w[..., :3]`` in
    world coordinates. Inside the box the penalty is zero.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    tip_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3]  # type: ignore

    x, y, z = tip_pos_w[:, 0], tip_pos_w[:, 1], tip_pos_w[:, 2]

    dx = torch.clamp(x_bounds[0] - x, min=0.0) + torch.clamp(x - x_bounds[1], min=0.0)
    dy = torch.clamp(y_bounds[0] - y, min=0.0) + torch.clamp(y - y_bounds[1], min=0.0)
    dz = torch.clamp(z_bounds[0] - z, min=0.0) + torch.clamp(z - z_bounds[1], min=0.0)

    penalty = dx * dx + dy * dy + dz * dz if squared else dx + dy + dz
    return penalty


def end_effector_workspace_boundary_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot_1", body_names="psm_tool_tip_Link"),
    x_bounds: tuple[float, float] = (-0.15, 0.15),
    y_bounds: tuple[float, float] = (0.2, 0.5),
    z_bounds: tuple[float, float] = (0.01, 0.2),
    margin: float = 0.02,
    squared: bool = True,
) -> torch.Tensor:
    """Penalize being close to workspace boundaries while still inside the workspace.

    This term is intended as a soft safety shaping signal near the boundary.
    Outside-workspace penalty should be handled by ``end_effector_workspace_penalty``.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    tip_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3]  # type: ignore

    x, y, z = tip_pos_w[:, 0], tip_pos_w[:, 1], tip_pos_w[:, 2]
    inside = (
        (x >= x_bounds[0])
        & (x <= x_bounds[1])
        & (y >= y_bounds[0])
        & (y <= y_bounds[1])
        & (z >= z_bounds[0])
        & (z <= z_bounds[1])
    )

    dists = torch.stack(
        (
            x - x_bounds[0],
            x_bounds[1] - x,
            y - y_bounds[0],
            y_bounds[1] - y,
            z - z_bounds[0],
            z_bounds[1] - z,
        ),
        dim=1,
    )
    min_dist = torch.min(dists, dim=1).values

    safe_margin = max(float(margin), 1e-6)
    proximity = torch.clamp((safe_margin - min_dist) / safe_margin, min=0.0)
    proximity = proximity * inside.float()
    return proximity * proximity if squared else proximity


def _episode_start_mask(env: ManagerBasedRLEnv) -> torch.Tensor:
    if hasattr(env, "episode_length_buf"):
        return env.episode_length_buf <= 1
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)


def reset_vessel_connectivity_state(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
) -> None:
    """Reset per-episode state used by vessel connectivity reward/termination.

    This prevents connectivity streak counters from leaking across episode
    boundaries, which can otherwise cause immediate success after reset.
    """
    if env_ids.numel() == 0:
        return

    # Collect diagnostics as pending episode logs before wiping connectivity state.
    from .metrics import collect_and_reset_vessel_connectivity_diagnostics

    collect_and_reset_vessel_connectivity_diagnostics(env, env_ids)

    def _reset_or_init_tensor(attr_name: str, dtype: torch.dtype) -> None:
        tensor = getattr(env, attr_name, None)
        needs_reinit = not torch.is_tensor(tensor)

        if torch.is_tensor(tensor):
            if tensor.shape[0] != env.num_envs:
                needs_reinit = True
            elif hasattr(tensor, "is_inference") and tensor.is_inference():
                # Inference tensors cannot be updated in-place outside inference mode.
                needs_reinit = True

        if needs_reinit:
            tensor = torch.zeros(env.num_envs, device=env.device, dtype=dtype)
            setattr(env, attr_name, tensor)

        tensor[env_ids] = 0

    # Strict completion streak m_t.
    _reset_or_init_tensor("_vessel_cd_gall_counter", torch.long)
    # Shaping counter n_t and cached previous n_{t-1}.
    _reset_or_init_tensor("_vessel_cd_gall_reward_counter", torch.long)
    _reset_or_init_tensor("_vessel_cd_gall_reward_counter_prev", torch.long)
    # Connectivity masks c_t / c_{t-1} used for break penalty.
    _reset_or_init_tensor("_vessel_cd_gall_connected_mask", torch.bool)
    _reset_or_init_tensor("_vessel_cd_gall_prev_connected_mask", torch.bool)

    # Force state updates on next frame after reset.
    setattr(env, "_vessel_cd_gall_counter_fp", None)
    setattr(env, "_vessel_cd_gall_reward_counter_fp", None)
    setattr(env, "_vessel_cd_gall_connected_mask_fp", None)


def vessel_depth_weighted_area_delta_reward(
    env: ManagerBasedRLEnv,
    camera_cfg_name: str = "camera",
    depth_camera_cfg_name: str = "depth_camera",
    vessel_label: str = "vessel",
    vessel_color: tuple = (25, 82, 255),
    color_tolerance: int = 10,
    prefer_semantic_info: bool = True,
    normalizer_alpha: float = 0.99,
    normalizer_clip: float = 5.0,
    depth_exponent: float = 2.0,
) -> torch.Tensor:
    """EMA-normalized frame-to-frame change of depth-weighted vessel area.

    This mirrors ``scripts/saved/scripts/reward.py``: the metric is
    ``sum(vessel_mask * depth**depth_exponent)`` and the returned reward is the
    clipped EMA-normalized delta of that metric for each environment.
    """
    camera = env.scene[camera_cfg_name]
    depth_camera = env.scene[depth_camera_cfg_name]

    sem_data = camera.data.output.get("semantic_segmentation")
    depth_data = depth_camera.data.output.get("depth")
    if sem_data is None or depth_data is None:
        return torch.zeros(env.num_envs, device=env.device)

    if sem_data.shape[0] != env.num_envs or depth_data.shape[0] != env.num_envs:
        return torch.zeros(env.num_envs, device=env.device)

    sem_info = camera.data.info.get("semantic_segmentation", {})
    vessel_mask = _get_label_mask_batch(
        sem_data,
        sem_info,
        vessel_label,
        vessel_color,
        color_tolerance,
        prefer_semantic_info,
    )

    depth_values = depth_data[..., 0] if depth_data.ndim == 4 and depth_data.shape[-1] == 1 else depth_data
    depth_values = torch.nan_to_num(depth_values.float(), nan=0.0, posinf=0.0, neginf=0.0)
    weighted_area = torch.sum(vessel_mask.float() * depth_values.pow(depth_exponent), dim=(1, 2))

    state_shape = (env.num_envs,)
    if not hasattr(env, "_vessel_area_prev") or getattr(env, "_vessel_area_prev").shape != state_shape:
        env._vessel_area_prev = torch.zeros(state_shape, device=env.device)
        env._vessel_area_mean = torch.zeros(state_shape, device=env.device)
        env._vessel_area_var = torch.ones(state_shape, device=env.device)

    prev_metric = env._vessel_area_prev
    ema_mean = env._vessel_area_mean
    ema_var = env._vessel_area_var

    episode_start = _episode_start_mask(env)
    delta = torch.zeros_like(weighted_area)
    continuing = ~episode_start
    delta[continuing] = weighted_area[continuing] - prev_metric[continuing]

    ema_mean[episode_start] = 0.0
    ema_var[episode_start] = 1.0
    prev_metric[episode_start] = weighted_area[episode_start]

    if continuing.any():
        ema_mean[continuing] = normalizer_alpha * ema_mean[continuing] + (1.0 - normalizer_alpha) * delta[continuing]
        ema_var[continuing] = (
            normalizer_alpha * ema_var[continuing]
            + (1.0 - normalizer_alpha) * (delta[continuing] - ema_mean[continuing]).pow(2)
        )
        prev_metric[continuing] = weighted_area[continuing]

    std = torch.sqrt(torch.clamp(ema_var, min=1e-6))
    return torch.clamp(delta / std, -normalizer_clip, normalizer_clip)


def gall_depth_weighted_area_delta_reward(
    env: ManagerBasedRLEnv,
    camera_cfg_name: str = "camera",
    depth_camera_cfg_name: str = "depth_camera",
    gall_label: str = "gall",
    gall_color: tuple = (255, 105, 180),
    color_tolerance: int = 10,
    prefer_semantic_info: bool = True,
    normalizer_alpha: float = 0.99,
    normalizer_clip: float = 5.0,
    depth_exponent: float = 2.0,
    penalize_decrease: bool = True,
) -> torch.Tensor:
    """EMA-normalized frame-to-frame change of depth-weighted gall area."""
    camera = env.scene[camera_cfg_name]
    depth_camera = env.scene[depth_camera_cfg_name]

    sem_data = camera.data.output.get("semantic_segmentation")
    depth_data = depth_camera.data.output.get("depth")
    if sem_data is None or depth_data is None:
        return torch.zeros(env.num_envs, device=env.device)

    if sem_data.shape[0] != env.num_envs or depth_data.shape[0] != env.num_envs:
        return torch.zeros(env.num_envs, device=env.device)

    sem_info = camera.data.info.get("semantic_segmentation", {})
    gall_mask = _get_label_mask_batch(
        sem_data,
        sem_info,
        gall_label,
        gall_color,
        color_tolerance,
        prefer_semantic_info,
    )

    depth_values = depth_data[..., 0] if depth_data.ndim == 4 and depth_data.shape[-1] == 1 else depth_data
    depth_values = torch.nan_to_num(depth_values.float(), nan=0.0, posinf=0.0, neginf=0.0)
    weighted_area = torch.sum(gall_mask.float() * depth_values.pow(depth_exponent), dim=(1, 2))

    state_shape = (env.num_envs,)
    if not hasattr(env, "_gall_area_prev") or getattr(env, "_gall_area_prev").shape != state_shape:
        env._gall_area_prev = torch.zeros(state_shape, device=env.device)
        env._gall_area_mean = torch.zeros(state_shape, device=env.device)
        env._gall_area_var = torch.ones(state_shape, device=env.device)

    prev_metric = env._gall_area_prev
    ema_mean = env._gall_area_mean
    ema_var = env._gall_area_var

    episode_start = _episode_start_mask(env)
    delta = torch.zeros_like(weighted_area)
    continuing = ~episode_start
    delta[continuing] = weighted_area[continuing] - prev_metric[continuing]

    ema_mean[episode_start] = 0.0
    ema_var[episode_start] = 1.0
    prev_metric[episode_start] = weighted_area[episode_start]

    if continuing.any():
        ema_mean[continuing] = normalizer_alpha * ema_mean[continuing] + (1.0 - normalizer_alpha) * delta[continuing]
        ema_var[continuing] = (
            normalizer_alpha * ema_var[continuing]
            + (1.0 - normalizer_alpha) * (delta[continuing] - ema_mean[continuing]).pow(2)
        )
        prev_metric[continuing] = weighted_area[continuing]

    std = torch.sqrt(torch.clamp(ema_var, min=1e-6))
    reward_delta = delta if penalize_decrease else torch.clamp(delta, min=0.0)
    return torch.clamp(reward_delta / std, -normalizer_clip, normalizer_clip)


# ──────────────────────────────────────────────────────────────────────────────
# Vessel skeleton trisection analysis helpers (adapted from process.py)
# ──────────────────────────────────────────────────────────────────────────────


def _skel_skeletonize(mask_bool: np.ndarray) -> np.ndarray:
    """Skeletonize a 2D boolean numpy mask using the best available backend."""
    mb = mask_bool.astype(bool)
    if _HAS_SKIMAGE:
        return _skimage_skeletonize(mb)
    if _HAS_CV2 and hasattr(_cv2, "ximgproc") and hasattr(_cv2.ximgproc, "thinning"):
        return _cv2.ximgproc.thinning(
            mb.astype(np.uint8) * 255,
            thinningType=_cv2.ximgproc.THINNING_ZHANGSUEN,
        ) > 0
    if not _HAS_CV2:
        return mb  # no-op fallback
    # Morphological fallback
    img = (mb.astype(np.uint8) * 255).copy()
    skel = np.zeros_like(img)
    kernel = _cv2.getStructuringElement(_cv2.MORPH_CROSS, (3, 3))
    while True:
        eroded = _cv2.erode(img, kernel)
        opened = _cv2.dilate(eroded, kernel)
        skel = _cv2.bitwise_or(skel, _cv2.subtract(img, opened))
        img = eroded
        if _cv2.countNonZero(img) == 0:
            break
    return skel > 0


def _skel_keep_largest(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest connected component in a boolean mask."""
    if not _HAS_CV2:
        return mask
    u8 = mask.astype(np.uint8)
    n, labels, stats, _ = _cv2.connectedComponentsWithStats(u8, connectivity=8)
    if n <= 1:
        return mask
    best = int(np.argmax(stats[1:, _cv2.CC_STAT_AREA]) + 1)
    return (labels == best).astype(bool)


def _skel_filter_small(mask: np.ndarray, min_px: int = 20) -> np.ndarray:
    """Remove connected components smaller than *min_px* pixels."""
    if not _HAS_CV2:
        return mask
    u8 = mask.astype(np.uint8)
    n, labels, stats, _ = _cv2.connectedComponentsWithStats(u8, connectivity=8)
    out = np.zeros_like(u8)
    for i in range(1, n):
        if stats[i, _cv2.CC_STAT_AREA] >= min_px:
            out[labels == i] = 1
    return out.astype(bool)


def _skel_build_graph(skel: np.ndarray):
    """Build an adjacency-list graph from skeleton pixels.

    Returns ``(graph, endpoints)`` where *graph* maps ``(y, x)`` to a list of
    neighbour coordinates and *endpoints* are nodes with degree 1.
    """
    pts = np.column_stack(np.where(skel))
    if pts.size == 0:
        return {}, []
    ps = {(int(y), int(x)) for y, x in pts}
    _D = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    g: dict = {}
    for y, x in ps:
        g[(y, x)] = [(y + dy, x + dx) for dy, dx in _D if (y + dy, x + dx) in ps]
    eps = [p for p, nb in g.items() if len(nb) == 1]
    return g, eps


def _skel_cc(g: dict) -> list:
    """Connected components of a skeleton graph."""
    vis: set = set()
    comps: list = []
    for n in g:
        if n in vis:
            continue
        q = deque([n])
        vis.add(n)
        c: list = []
        while q:
            cur = q.popleft()
            c.append(cur)
            for nb in g[cur]:
                if nb not in vis:
                    vis.add(nb)
                    q.append(nb)
        comps.append(c)
    return comps


def _skel_bfs(g: dict, s, t) -> list:
    """BFS shortest path from *s* to *t* in graph *g*."""
    q = deque([s])
    par = {s: None}
    while q:
        c = q.popleft()
        if c == t:
            break
        for nx in g.get(c, []):
            if nx not in par:
                par[nx] = c
                q.append(nx)
    if t not in par:
        return []
    p: list = []
    c = t
    while c is not None:
        p.append(c)
        c = par[c]
    p.reverse()
    return p


def _skel_bfs_far(g: dict, s, allowed=None):
    """BFS to find the farthest node from *s*. Returns ``(farthest, dist, parent)``."""
    al = set(g.keys()) if allowed is None else set(allowed)
    q = deque([s])
    d = {s: 0}
    par = {s: None}
    last = s
    while q:
        c = q.popleft()
        last = c
        for nb in g.get(c, []):
            if nb in al and nb not in d:
                d[nb] = d[c] + 1
                par[nb] = c
                q.append(nb)
    return last, d, par


def _skel_path_len(p: list) -> float:
    """Euclidean path length of a sequence of ``(y, x)`` points."""
    if len(p) < 2:
        return 0.0
    return sum(
        math.hypot(p[i][1] - p[i - 1][1], p[i][0] - p[i - 1][0])
        for i in range(1, len(p))
    )


def _skel_straightness(seg: list) -> float:
    """Straightness = endpoint_distance / path_length (1.0 = perfectly straight)."""
    if len(seg) < 2:
        return 0.0
    ed = math.hypot(seg[-1][1] - seg[0][1], seg[-1][0] - seg[0][0])
    cl = _skel_path_len(seg)
    return float(ed / cl) if cl > 1e-6 else 0.0


def _skel_straightness_pca(seg: list) -> float:
    """PCA-based straightness = lambda_max / (lambda1 + lambda2)."""
    if len(seg) < 2:
        return 0.0
    pts = np.array([[float(x), float(y)] for y, x in seg], dtype=np.float32)
    pts_centered = pts - np.mean(pts, axis=0, keepdims=True)
    cov = (pts_centered.T @ pts_centered) / max(pts.shape[0] - 1, 1)
    eigvals = np.linalg.eigvalsh(cov)
    total = float(np.sum(eigvals))
    if total <= 1e-9:
        return 0.0
    return float(np.clip(np.max(eigvals) / total, 0.0, 1.0))


def _arm_direction_vector(arm_path: list[tuple[int, int]], junction: tuple[int, int] | None) -> np.ndarray:
    """Compute normalized arm direction using PCA, oriented away from junction."""
    if junction is None or len(arm_path) < 2:
        return np.array([0.0, 0.0], dtype=np.float32)
    pts = np.array([[float(x), float(y)] for y, x in arm_path], dtype=np.float32)
    pts_centered = pts - np.mean(pts, axis=0, keepdims=True)
    cov = (pts_centered.T @ pts_centered) / max(pts.shape[0] - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    direction = eigvecs[:, int(np.argmax(eigvals))]

    jx, jy = float(junction[1]), float(junction[0])
    vec_to_end = pts[-1] - np.array([jx, jy], dtype=np.float32)
    if float(np.dot(direction, vec_to_end)) < 0.0:
        direction = -direction

    norm = float(np.linalg.norm(direction))
    if norm <= 1e-9:
        return np.array([0.0, 0.0], dtype=np.float32)
    return (direction / norm).astype(np.float32)


def _compute_angle_between_arms_deg(
    arm_a: list[tuple[int, int]],
    arm_b: list[tuple[int, int]],
    junction: tuple[int, int] | None,
) -> float:
    """Compute angle (degrees) between two arm directions, in [0, 180]."""
    da = _arm_direction_vector(arm_a, junction)
    db = _arm_direction_vector(arm_b, junction)
    na = float(np.linalg.norm(da))
    nb = float(np.linalg.norm(db))
    if na <= 1e-9 or nb <= 1e-9:
        return -1.0
    cos_val = float(np.clip(np.dot(da, db), -1.0, 1.0))
    return float(math.degrees(math.acos(cos_val)))


def _skel_find_junction(g: dict):
    """Find the main Y-junction node (degree >= 3) in the skeleton graph.

    Prefers nodes whose removal splits the graph into exactly 3 components.
    """
    bns = [n for n, nb in g.items() if len(nb) >= 3]
    if not bns:
        return None
    if len(bns) == 1:
        return bns[0]
    best, bs = None, -1
    for bn in bns:
        sub = {k: [x for x in v if x != bn] for k, v in g.items() if k != bn}
        nc = len(_skel_cc(sub))
        sc = (nc == 3) * 1000 + len(g[bn])
        if sc > bs:
            bs, best = sc, bn
    return best


def _skel_prune(g: dict, min_len: int = 12) -> dict:
    """Prune short spur branches (< *min_len* pixels) from skeleton graph."""
    adj: dict = {k: list(v) for k, v in g.items()}
    changed = True
    while changed:
        changed = False
        for ep in [n for n, nb in adj.items() if len(nb) == 1]:
            if ep not in adj or len(adj[ep]) != 1:
                continue
            path, cur, prev = [ep], ep, None
            while True:
                fwd = [n for n in adj.get(cur, []) if n != prev]
                if len(fwd) != 1:
                    at_j = len(fwd) >= 2
                    break
                prev, cur = cur, fwd[0]
                path.append(cur)
                if len(path) > min_len:
                    at_j = False
                    break
            if len(path) <= min_len and at_j:
                for nd in path[:-1]:
                    for nb in list(adj.get(nd, [])):
                        if nb in adj:
                            try:
                                adj[nb].remove(nd)
                            except ValueError:
                                pass
                    adj.pop(nd, None)
                changed = True
    return adj


def _skel_est_min_arm(g: dict) -> int:
    """Estimate minimum arm length for spur pruning threshold."""
    j = _skel_find_junction(g)
    if j is None:
        return 6
    lens: list = []
    for nb in g[j]:
        p, cur, prev = [nb], nb, j
        while True:
            fwd = [x for x in g.get(cur, []) if x != prev]
            if len(fwd) != 1:
                break
            prev, cur = cur, fwd[0]
            p.append(cur)
        lens.append(len(p))
    return max(3, min(lens) // 3) if lens else 6


def _skel_split_arms(g: dict, junc) -> list:
    """Split skeleton into 3 arms from a junction, choosing the most angularly spread set."""
    eps = [n for n, nb in g.items() if len(nb) == 1]
    paths = [p for ep in eps for p in [_skel_bfs(g, junc, ep)] if len(p) >= 2]
    if len(paths) <= 3:
        return paths
    angs = [math.atan2(p[-1][0] - junc[0], p[-1][1] - junc[1]) for p in paths]
    from itertools import combinations
    best_c, best_s = None, -1.0
    for c in combinations(range(len(paths)), 3):
        a = sorted(angs[i] for i in c)
        ds = [a[1] - a[0], a[2] - a[1], a[0] + 2 * math.pi - a[2]]
        s = min(ds)
        if s > best_s:
            best_s, best_c = s, c
    if best_c is None:
        return paths[:3]
    sel = [paths[i] for i in best_c]
    sel.sort(key=lambda a: a[-1][1])
    right = sorted(sel[1:], key=lambda a: a[-1][0])
    return [sel[0], right[0], right[1]]


def _skel_split_equal(path: list) -> list:
    """Split a path into 3 equal-length segments."""
    if len(path) < 3:
        return [path, [], []]
    cum = [0.0]
    for i in range(1, len(path)):
        cum.append(cum[-1] + math.hypot(
            path[i][1] - path[i - 1][1], path[i][0] - path[i - 1][0]
        ))
    tot = cum[-1]
    if tot <= 1e-6:
        i1, i2 = len(path) // 3, 2 * len(path) // 3
    else:
        i1 = bisect_left(cum, tot / 3)
        i2 = bisect_left(cum, 2 * tot / 3)
    i1 = max(1, min(len(path) - 2, i1))
    i2 = max(i1 + 1, min(len(path) - 1, i2))
    return [path[: i1 + 1], path[i1: i2 + 1], path[i2:]]


def _order_arms_by_anatomical_position(arms: list[list[tuple[int, int]]]) -> list[list[tuple[int, int]]]:
    """Return arms ordered as ``[CD, CHD, CBD]``.

    - ``CBD``: endpoint with the largest ``y`` coordinate (lowest branch)
    - Remaining upper two arms: smaller endpoint ``x`` is ``CD`` and larger is ``CHD``
    """
    if len(arms) != 3 or any(len(arm) == 0 for arm in arms):
        return arms

    endpoints = [arm[-1] for arm in arms]
    cbd_idx = max(range(3), key=lambda idx: endpoints[idx][0])
    upper = [idx for idx in range(3) if idx != cbd_idx]
    cd_idx, chd_idx = sorted(upper, key=lambda idx: endpoints[idx][1])
    return [arms[cd_idx], arms[chd_idx], arms[cbd_idx]]


def _vessel_trisect(vessel_mask: np.ndarray) -> dict:
    """Perform vessel skeleton trisection on a 2D boolean mask.

    Returns a dict with:
        - ``success``: whether 3 valid (>= 2 points) segments were found
        - ``segments``: list of 3 lists of ``(y, x)`` tuples
        - ``straightness``: list of 3 floats in ``[0, 1]``
        - ``junction``: ``(y, x)`` or ``None``
    """
    _empty: dict = {
        "success": False,
        "segments": [[], [], []],
        "straightness": [0.0, 0.0, 0.0],
        "straightness_length_ratio": [0.0, 0.0, 0.0],
        "straightness_pca": [0.0, 0.0, 0.0],
        "cd_chd_angle_deg": -1.0,
        "junction": None,
    }
    if not _HAS_CV2 or not vessel_mask.any():
        return _empty

    main = _skel_keep_largest(vessel_mask)
    if not main.any():
        return _empty
    skel = _skel_skeletonize(main)
    skel = _skel_filter_small(skel, 20)
    if not skel.any():
        return _empty

    g, eps = _skel_build_graph(skel)
    if not g:
        return _empty
    thresh = _skel_est_min_arm(g)
    g = _skel_prune(g, thresh)
    eps = [n for n, nb in g.items() if len(nb) == 1]

    junc = _skel_find_junction(g)
    if junc is not None:
        segs = _skel_split_arms(g, junc)
    else:
        nodes = max(_skel_cc(g), key=len) if g else []
        if not nodes:
            return _empty
        start = max(nodes, key=lambda p: (p[0], -p[1]))
        u, _, _ = _skel_bfs_far(g, start, nodes)
        v, _, par = _skel_bfs_far(g, u, nodes)
        path: list = []
        c = v
        while c is not None:
            path.append(c)
            c = par.get(c)
        path.reverse()
        segs = _skel_split_equal(path)

    valid = [s for s in segs if len(s) >= 2]
    ok = len(valid) == 3
    if ok:
        valid = _order_arms_by_anatomical_position(valid[:3])
    while len(valid) < 3:
        valid.append([])
    strs_length_ratio = [_skel_straightness(s) for s in valid[:3]]
    strs_pca = [_skel_straightness_pca(s) for s in valid[:3]]
    cd_chd_angle_deg = _compute_angle_between_arms_deg(valid[0], valid[1], junc) if ok else -1.0
    return {
        "success": ok,
        "segments": valid[:3],
        "straightness": strs_length_ratio,
        "straightness_length_ratio": strs_length_ratio,
        "straightness_pca": strs_pca,
        "cd_chd_angle_deg": cd_chd_angle_deg,
        "junction": junc,
    }


def _check_gall_adjacency(
    segs: list,
    gall_mask: np.ndarray,
    radius: int = 5,
) -> list:
    """Check which vessel segments have endpoints adjacent to the gall mask.

    A segment is considered connected if any of its endpoint pixels
    (first/last ``check_count`` points) fall within a dilated gall region.

    Returns a list of 3 booleans.
    """
    if not _HAS_CV2 or not gall_mask.any():
        return [False, False, False]
    u8 = gall_mask.astype(np.uint8) * 255
    k = _cv2.getStructuringElement(_cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    zone = _cv2.dilate(u8, k, iterations=1) > 0
    h, w = gall_mask.shape
    conns: list = []
    for seg in segs[:3]:
        found = False
        if len(seg) >= 2:
            check = max(1, min(len(seg), 10))
            # Check both ends of the segment
            for pt in list(seg[-check:]) + list(seg[:check]):
                y, x = pt
                if 0 <= y < h and 0 <= x < w and zone[y, x]:
                    found = True
                    break
        conns.append(found)
    while len(conns) < 3:
        conns.append(False)
    return conns


def _draw_arm_mask(shape: tuple[int, int], arm_path: list[tuple[int, int]], thickness: int = 2) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    if not arm_path:
        return mask.astype(bool)
    if len(arm_path) == 1:
        y, x = arm_path[0]
        _cv2.circle(mask, (x, y), max(1, thickness // 2), 255, -1)
        return mask.astype(bool)
    for index in range(1, len(arm_path)):
        y0, x0 = arm_path[index - 1]
        y1, x1 = arm_path[index]
        _cv2.line(mask, (x0, y0), (x1, y1), 255, thickness, lineType=_cv2.LINE_AA)
    return mask.astype(bool)


def _extract_boundary(mask_bool: np.ndarray) -> np.ndarray:
    mask_u8 = mask_bool.astype(np.uint8) * 255
    if np.count_nonzero(mask_u8) == 0:
        return np.zeros_like(mask_bool, dtype=bool)
    kernel = _cv2.getStructuringElement(_cv2.MORPH_ELLIPSE, (3, 3))
    eroded = _cv2.erode(mask_u8, kernel, iterations=1)
    boundary = _cv2.subtract(mask_u8, eroded)
    return boundary > 0


def _compute_min_distance_between_masks(source_mask: np.ndarray, target_boundary_mask: np.ndarray) -> float:
    if not np.any(source_mask) or not np.any(target_boundary_mask):
        return -1.0
    target_inv = np.where(target_boundary_mask, 0, 255).astype(np.uint8)
    distance_map = _cv2.distanceTransform(target_inv, _cv2.DIST_L2, 5)
    return float(np.min(distance_map[source_mask]))


def _compute_centroid(mask_bool: np.ndarray) -> tuple[int, int] | None:
    coords = np.column_stack(np.where(mask_bool))
    if coords.size == 0:
        return None
    cy, cx = np.mean(coords, axis=0)
    return int(round(cy)), int(round(cx))


def _same_component(binary_mask: np.ndarray, point_a: tuple[int, int] | None, point_b: tuple[int, int] | None) -> bool:
    if point_a is None or point_b is None:
        return False
    if not binary_mask[point_a[0], point_a[1]] or not binary_mask[point_b[0], point_b[1]]:
        return False
    num_labels, labels = _cv2.connectedComponents(binary_mask.astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        return False
    label_a = labels[point_a[0], point_a[1]]
    label_b = labels[point_b[0], point_b[1]]
    return bool(label_a != 0 and label_a == label_b)


def _ensure_point_inside_mask(point: tuple[int, int] | None, mask_bool: np.ndarray) -> tuple[int, int] | None:
    if point is None:
        return None
    y, x = point
    h, w = mask_bool.shape
    if 0 <= y < h and 0 <= x < w and mask_bool[y, x]:
        return point
    coords = np.column_stack(np.where(mask_bool))
    if coords.size == 0:
        return None
    dists = np.sum((coords - np.array([[y, x]])) ** 2, axis=1)
    best = coords[int(np.argmin(dists))]
    return int(best[0]), int(best[1])


def _morphological_connectivity(
    cd_mask: np.ndarray,
    gall_mask: np.ndarray,
    cd_endpoint: tuple[int, int] | None,
    close_radius: int,
) -> tuple[int, tuple[int, int] | None, tuple[int, int] | None, np.ndarray]:
    combined = np.logical_or(cd_mask, gall_mask)
    if not np.any(combined):
        return CONNECTIVITY_NOT_CONNECTED, None, None, combined
    kernel_size = max(1, 2 * int(close_radius) + 1)
    kernel = _cv2.getStructuringElement(_cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    closed = _cv2.morphologyEx(combined.astype(np.uint8) * 255, _cv2.MORPH_CLOSE, kernel, iterations=1) > 0
    gall_centroid = _ensure_point_inside_mask(_compute_centroid(gall_mask), closed)
    cd_anchor = _ensure_point_inside_mask(cd_endpoint, closed)
    connected = _same_component(closed, cd_anchor, gall_centroid)
    return (CONNECTIVITY_CONNECTED if connected else CONNECTIVITY_NOT_CONNECTED), cd_anchor, gall_centroid, closed


def _segment_gall_connected_by_merge(
    seg_path: list[tuple[int, int]],
    gall_mask: np.ndarray,
    close_radius: int,
) -> tuple[bool, tuple[int, int] | None, tuple[int, int] | None]:
    """Check whether one vessel segment is connected to gall via merged-pixel connectivity."""
    if not _HAS_CV2 or len(seg_path) == 0 or not np.any(gall_mask):
        return False, None, None
    seg_mask = _draw_arm_mask(gall_mask.shape, seg_path, thickness=2)
    connectivity, seg_anchor, gall_centroid, _ = _morphological_connectivity(
        cd_mask=seg_mask,
        gall_mask=gall_mask,
        cd_endpoint=seg_path[-1],
        close_radius=close_radius,
    )
    return connectivity == CONNECTIVITY_CONNECTED, seg_anchor, gall_centroid


def _is_only_cd_connected(result: dict[str, Any]) -> bool:
    """Return True when exactly the CD segment is connected to gall."""
    if not bool(result.get("success", False)):
        return False
    conns = result.get("gall_connections", [False, False, False])
    if not isinstance(conns, (list, tuple)) or len(conns) < 3:
        return False
    return bool(conns[0]) and (not bool(conns[1])) and (not bool(conns[2]))


def _get_cd_only_connectivity_counter(
    env: ManagerBasedRLEnv,
    camera_cfg_name: str = "camera",
    vessel_label: str = "vessel",
    gall_label: str = "gall",
    vessel_color: tuple = (25, 82, 255),
    gall_color: tuple = (255, 105, 180),
    color_tolerance: int = 10,
    prefer_semantic_info: bool = True,
    gall_dilation_radius: int = 8,
    compute_every: int = 1,
    disconnect_decay: int = 1,
) -> torch.Tensor:
    """Update-and-return strict per-env CD-only connectivity streak ``m_t``.

    Counter update rule:
    - CD-only connected: ``m_t = m_{t-1} + 1``
    - Otherwise: ``m_t = 0``

    Notes:
    - ``disconnect_decay`` is kept only for backward compatibility with old
      configs and is intentionally ignored in strict mode.
    """
    _ = disconnect_decay

    connected_mask, _ = _get_cd_only_connected_masks(
        env,
        camera_cfg_name,
        vessel_label,
        gall_label,
        vessel_color,
        gall_color,
        color_tolerance,
        prefer_semantic_info,
        gall_dilation_radius,
        compute_every,
    )

    counter = getattr(env, "_vessel_cd_gall_counter", None)
    needs_reinit = not torch.is_tensor(counter)
    if torch.is_tensor(counter):
        if counter.shape[0] != env.num_envs:
            needs_reinit = True
        elif hasattr(counter, "is_inference") and counter.is_inference():
            needs_reinit = True

    if needs_reinit:
        counter = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        setattr(env, "_vessel_cd_gall_counter", counter)
        setattr(env, "_vessel_cd_gall_counter_fp", None)

    current_fp = getattr(env, "_trisect_fp", None)
    last_fp = getattr(env, "_vessel_cd_gall_counter_fp", None)
    if current_fp == last_fp:
        return counter

    if connected_mask.any():
        counter[connected_mask] = counter[connected_mask] + 1

    disconnected_mask = ~connected_mask
    if disconnected_mask.any():
        counter[disconnected_mask] = 0

    setattr(env, "_vessel_cd_gall_counter_fp", current_fp)
    return counter


def _get_cd_only_connected_masks(
    env: ManagerBasedRLEnv,
    camera_cfg_name: str = "camera",
    vessel_label: str = "vessel",
    gall_label: str = "gall",
    vessel_color: tuple = (25, 82, 255),
    gall_color: tuple = (255, 105, 180),
    color_tolerance: int = 10,
    prefer_semantic_info: bool = True,
    gall_dilation_radius: int = 8,
    compute_every: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return current and previous CD-only connectivity masks per env.

    The state is updated at most once per distinct semantic frame. Returned
    masks correspond to ``(c_t, c_{t-1})`` used by shaping reward terms.
    """
    results = _get_trisection_cached(
        env,
        camera_cfg_name,
        vessel_label,
        gall_label,
        prefer_semantic_info,
        vessel_color,
        gall_color,
        color_tolerance,
        gall_dilation_radius,
        compute_every,
    )

    current_mask = getattr(env, "_vessel_cd_gall_connected_mask", None)
    prev_mask = getattr(env, "_vessel_cd_gall_prev_connected_mask", None)

    needs_reinit = (not torch.is_tensor(current_mask)) or (not torch.is_tensor(prev_mask))
    if torch.is_tensor(current_mask) and torch.is_tensor(prev_mask):
        if current_mask.shape[0] != env.num_envs or prev_mask.shape[0] != env.num_envs:
            needs_reinit = True
        elif hasattr(current_mask, "is_inference") and current_mask.is_inference():
            needs_reinit = True
        elif hasattr(prev_mask, "is_inference") and prev_mask.is_inference():
            needs_reinit = True

    if needs_reinit:
        current_mask = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        prev_mask = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        setattr(env, "_vessel_cd_gall_connected_mask", current_mask)
        setattr(env, "_vessel_cd_gall_prev_connected_mask", prev_mask)
        setattr(env, "_vessel_cd_gall_connected_mask_fp", None)

    current_fp = getattr(env, "_trisect_fp", None)
    last_fp = getattr(env, "_vessel_cd_gall_connected_mask_fp", None)
    if current_fp == last_fp:
        return current_mask, prev_mask

    new_mask = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    for i, result in enumerate(results):
        new_mask[i] = _is_only_cd_connected(result)

    prev_mask.copy_(current_mask)
    current_mask.copy_(new_mask)
    setattr(env, "_vessel_cd_gall_connected_mask_fp", current_fp)
    return current_mask, prev_mask


def _get_cd_only_connectivity_shaping_counter(
    env: ManagerBasedRLEnv,
    camera_cfg_name: str = "camera",
    vessel_label: str = "vessel",
    gall_label: str = "gall",
    vessel_color: tuple = (25, 82, 255),
    gall_color: tuple = (255, 105, 180),
    color_tolerance: int = 10,
    prefer_semantic_info: bool = True,
    gall_dilation_radius: int = 8,
    compute_every: int = 1,
    stable_frames: int = 5,
    disconnect_decay: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Update-and-return shaping counter ``n_t`` and previous ``n_{t-1}``.

    Update rule:
    - connected: ``n_t = min(n_{t-1} + 1, stable_frames)``
    - disconnected: ``n_t = max(n_{t-1} - disconnect_decay, 0)``
    """
    connected_mask, _ = _get_cd_only_connected_masks(
        env,
        camera_cfg_name,
        vessel_label,
        gall_label,
        vessel_color,
        gall_color,
        color_tolerance,
        prefer_semantic_info,
        gall_dilation_radius,
        compute_every,
    )

    counter = getattr(env, "_vessel_cd_gall_reward_counter", None)
    prev_counter = getattr(env, "_vessel_cd_gall_reward_counter_prev", None)

    needs_reinit = (not torch.is_tensor(counter)) or (not torch.is_tensor(prev_counter))
    if torch.is_tensor(counter) and torch.is_tensor(prev_counter):
        if counter.shape[0] != env.num_envs or prev_counter.shape[0] != env.num_envs:
            needs_reinit = True
        elif hasattr(counter, "is_inference") and counter.is_inference():
            needs_reinit = True
        elif hasattr(prev_counter, "is_inference") and prev_counter.is_inference():
            needs_reinit = True

    if needs_reinit:
        counter = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        prev_counter = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        setattr(env, "_vessel_cd_gall_reward_counter", counter)
        setattr(env, "_vessel_cd_gall_reward_counter_prev", prev_counter)
        setattr(env, "_vessel_cd_gall_reward_counter_fp", None)

    current_fp = getattr(env, "_trisect_fp", None)
    last_fp = getattr(env, "_vessel_cd_gall_reward_counter_fp", None)
    if current_fp == last_fp:
        return counter, prev_counter

    prev_counter.copy_(counter)
    cap = max(1, int(stable_frames))
    decay = max(0, int(disconnect_decay))

    if connected_mask.any():
        counter[connected_mask] = torch.clamp(counter[connected_mask] + 1, max=cap)

    disconnected_mask = ~connected_mask
    if disconnected_mask.any() and decay > 0:
        counter[disconnected_mask] = torch.clamp(counter[disconnected_mask] - decay, min=0)

    setattr(env, "_vessel_cd_gall_reward_counter_fp", current_fp)
    return counter, prev_counter


def _trisect_fingerprint(sem_data) -> int:
    """Quick fingerprint of semantic segmentation tensor to detect frame changes."""
    if sem_data is None or sem_data.numel() == 0:
        return 0
    try:
        h, w = sem_data.shape[1], sem_data.shape[2]
        ys = [0, h // 4, h // 2, 3 * h // 4, h - 1]
        xs = [0, w // 4, w // 2, 3 * w // 4, w - 1]
        sample = sem_data[0, ys, :, :][:, xs, :].reshape(-1)
        return hash(sample.cpu().to(torch.int32).numpy().tobytes())
    except Exception:
        return id(sem_data)


def _trisect_step_key(env, sem_data):
    """Return a cache key that advances every environment step.

    We prefer ``common_step_counter`` so repeated reward/termination calls
    within one step share the same trisection result, while consecutive steps
    always advance connectivity counters even if semantic pixels look similar.
    """
    step_counter = getattr(env, "common_step_counter", None)
    try:
        if step_counter is not None:
            return ("common_step_counter", int(step_counter))
    except Exception:
        pass

    # Fallback for environments without a global step counter.
    episode_buf = getattr(env, "episode_length_buf", None)
    if torch.is_tensor(episode_buf) and episode_buf.numel() > 0:
        try:
            return ("episode_length", int(episode_buf.reshape(-1)[0].item()))
        except Exception:
            pass

    # Last-resort fallback keeps previous behavior.
    return ("semantic_fingerprint", _trisect_fingerprint(sem_data))


def _get_label_mask_batch(
    sem_data: torch.Tensor,
    sem_info: Any,
    label: str,
    fallback_color: tuple | None,
    tolerance: int,
    prefer_info: bool,
) -> torch.Tensor:
    """Extract boolean mask ``[N, H, W]`` for a semantic label across all envs."""
    mask = None
    if prefer_info and label:
        mask = _semantic_label_mask_from_info(sem_data, sem_info, label)
    if mask is None and fallback_color is not None:
        rgb = sem_data[..., :3] if sem_data.shape[-1] >= 3 else sem_data
        r, g, b = fallback_color[:3]
        mask = (
            (torch.abs(rgb[..., 0].float() - r) < tolerance)
            & (torch.abs(rgb[..., 1].float() - g) < tolerance)
            & (torch.abs(rgb[..., 2].float() - b) < tolerance)
        )
    if mask is None:
        mask = torch.zeros(
            sem_data.shape[0], sem_data.shape[1], sem_data.shape[2],
            dtype=torch.bool, device=sem_data.device,
        )
    return mask


def _compute_trisection_all(
    env,
    camera_cfg_name: str,
    vessel_label: str,
    gall_label: str,
    prefer_info: bool,
    vessel_color: tuple,
    gall_color: tuple,
    color_tol: int,
    gall_dil: int,
) -> list:
    """Compute trisection analysis for every environment."""
    camera = env.scene[camera_cfg_name]
    sem_data = camera.data.output.get("semantic_segmentation")
    sem_info = camera.data.info.get("semantic_segmentation", {})
    n = env.num_envs

    def _empty_one() -> dict:
        return {
            "success": False,
            "segments": [[], [], []],
            "straightness": [0.0, 0.0, 0.0],
            "straightness_length_ratio": [0.0, 0.0, 0.0],
            "straightness_pca": [0.0, 0.0, 0.0],
            "cd_chd_angle_deg": -1.0,
            "junction": None,
            "gall_connections": [False, False, False],
            "gall_connection_count": 0,
            "cd_only_connected": False,
            "cd_to_gall_min_distance_px": -1.0,
            "hard_connectivity": CONNECTIVITY_NOT_CONNECTED,
            "hard_connected": False,
            "cd_endpoint": None,
            "gall_centroid": None,
        }

    if sem_data is None:
        return [_empty_one() for _ in range(n)]

    v_masks = _get_label_mask_batch(sem_data, sem_info, vessel_label, vessel_color, color_tol, prefer_info)
    g_masks = _get_label_mask_batch(sem_data, sem_info, gall_label, gall_color, color_tol, prefer_info)

    results: list = []
    for i in range(n):
        v_np = v_masks[i].cpu().numpy().astype(bool)
        g_np = g_masks[i].cpu().numpy().astype(bool)
        analysis = _vessel_trisect(v_np)
        analysis.setdefault("straightness_length_ratio", analysis.get("straightness", [0.0, 0.0, 0.0]))
        analysis.setdefault(
            "straightness_pca",
            [_skel_straightness_pca(seg) for seg in analysis.get("segments", [[], [], []])[:3]],
        )
        analysis.setdefault("cd_chd_angle_deg", -1.0)
        analysis["gall_connections"] = [False, False, False]
        analysis["gall_connection_count"] = 0
        analysis["cd_only_connected"] = False
        analysis["cd_to_gall_min_distance_px"] = -1.0
        analysis["hard_connectivity"] = CONNECTIVITY_NOT_CONNECTED
        analysis["hard_connected"] = False
        analysis["cd_endpoint"] = None
        analysis["gall_centroid"] = None

        if analysis["success"] and len(analysis["segments"]) == 3 and len(analysis["segments"][0]) > 0 and _HAS_CV2:
            segs = analysis["segments"][:3]
            gc: list[bool] = []
            anchors: list[tuple[int, int] | None] = []
            gall_centroid_ref: tuple[int, int] | None = None
            for seg in segs:
                connected, seg_anchor, gall_centroid = _segment_gall_connected_by_merge(seg, g_np, gall_dil)
                gc.append(connected)
                anchors.append(seg_anchor)
                if gall_centroid_ref is None and gall_centroid is not None:
                    gall_centroid_ref = gall_centroid

            while len(gc) < 3:
                gc.append(False)
            while len(anchors) < 3:
                anchors.append(None)

            analysis["gall_connections"] = gc
            analysis["gall_connection_count"] = int(sum(gc))
            analysis["cd_only_connected"] = bool(gc[0] and (not gc[1]) and (not gc[2]))
            analysis["hard_connectivity"] = (
                CONNECTIVITY_CONNECTED if analysis["cd_only_connected"] else CONNECTIVITY_NOT_CONNECTED
            )
            analysis["hard_connected"] = analysis["cd_only_connected"]
            analysis["cd_endpoint"] = anchors[0]
            analysis["gall_centroid"] = gall_centroid_ref

            cd_arm = segs[0]
            cd_mask = _draw_arm_mask(v_np.shape, cd_arm, thickness=2)
            gall_boundary = _extract_boundary(g_np)
            analysis["cd_to_gall_min_distance_px"] = _compute_min_distance_between_masks(cd_mask, gall_boundary)
        results.append(analysis)
    return results


def _get_trisection_cached(
    env,
    camera_cfg_name: str,
    vessel_label: str,
    gall_label: str,
    prefer_info: bool,
    vessel_color: tuple,
    gall_color: tuple,
    color_tol: int,
    gall_dil: int,
    compute_every: int,
) -> list:
    """Return cached trisection results, recomputing once per environment step.

    Within the same environment step the cached results are returned immediately
    so that multiple reward/termination terms share one computation.
    *compute_every* controls how often the heavy skeleton analysis runs,
    measured in environment action steps.
    """
    try:
        camera = env.scene[camera_cfg_name]
        sem_data = camera.data.output.get("semantic_segmentation")
    except Exception:
        sem_data = None

    step_key = _trisect_step_key(env, sem_data)
    cache = getattr(env, "_trisect_cache", None)
    cached_fp = getattr(env, "_trisect_fp", None)

    # Same env-step -> reuse (handles multiple reward/termination calls per step)
    if cache is not None and cached_fp == step_key:
        return cache

    # New env-step detected
    step = getattr(env, "_trisect_step", 0) + 1
    setattr(env, "_trisect_step", step)
    setattr(env, "_trisect_fp", step_key)

    if cache is not None and (step % max(1, compute_every)) != 0:
        return cache  # reuse stale cache until compute_every boundary

    # Heavy computation
    try:
        results = _compute_trisection_all(
            env, camera_cfg_name, vessel_label, gall_label,
            prefer_info, vessel_color, gall_color, color_tol, gall_dil,
        )
    except Exception:
        n = getattr(env, "num_envs", 1)
        results = [{
            "success": False,
            "segments": [[], [], []],
            "straightness": [0.0, 0.0, 0.0],
            "straightness_length_ratio": [0.0, 0.0, 0.0],
            "straightness_pca": [0.0, 0.0, 0.0],
            "cd_chd_angle_deg": -1.0,
            "junction": None,
            "gall_connections": [False, False, False],
            "gall_connection_count": 0,
            "cd_only_connected": False,
        } for _ in range(n)]

    setattr(env, "_trisect_cache", results)
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Three new reward functions for vessel trisection analysis
# ──────────────────────────────────────────────────────────────────────────────


def vessel_trisection_reward(
    env: ManagerBasedRLEnv,
    camera_cfg_name: str = "camera",
    vessel_label: str = "vessel",
    gall_label: str = "gall",
    vessel_color: tuple = (25, 82, 255),
    gall_color: tuple = (255, 105, 180),
    color_tolerance: int = 10,
    prefer_semantic_info: bool = True,
    gall_dilation_radius: int = 5,
    compute_every: int = 1,
) -> torch.Tensor:
    """Reward for successfully splitting the vessel skeleton into 3 segments.

    The vessel mask is skeletonized and a Y-junction is detected.  If the
    junction exists and exactly 3 valid arms (each >= 2 skeleton points) are
    found, the reward is 1.0; otherwise it is 0.0.

    When no Y-junction is found the longest skeleton path is split into 3
    equal-length segments as a fallback.

    Args:
        env: The environment instance.
        camera_cfg_name: Name of the camera sensor in the scene.
        vessel_label: Semantic label for the vessel class.
        gall_label: Semantic label for the gall (gallbladder) class.
        vessel_color: Fallback RGB color for vessel when label lookup fails.
        gall_color: Fallback RGB color for gall when label lookup fails.
        color_tolerance: Pixel-value tolerance for color matching.
        prefer_semantic_info: Prefer label-based matching over color.
        gall_dilation_radius: Dilation radius (px) for gall adjacency check.
        compute_every: Recompute skeleton every N frames (1 = every frame).

    Returns:
        Tensor of shape ``(num_envs,)`` with value 1.0 or 0.0 per env.
    """
    results = _get_trisection_cached(
        env, camera_cfg_name, vessel_label, gall_label,
        prefer_semantic_info, vessel_color, gall_color,
        color_tolerance, gall_dilation_radius, compute_every,
    )
    reward = torch.zeros(env.num_envs, device=env.device)
    for i, r in enumerate(results):
        if r["success"]:
            reward[i] = 1.0
    return reward


def vessel_segment_straightness_reward(
    env: ManagerBasedRLEnv,
    camera_cfg_name: str = "camera",
    vessel_label: str = "vessel",
    gall_label: str = "gall",
    vessel_color: tuple = (25, 82, 255),
    gall_color: tuple = (255, 105, 180),
    color_tolerance: int = 10,
    prefer_semantic_info: bool = True,
    gall_dilation_radius: int = 5,
    compute_every: int = 1,
    straightness_mode: str = "length_ratio",
) -> torch.Tensor:
    """Reward based on the average straightness of 3 vessel skeleton segments.

    Straightness metric is controlled by ``straightness_mode``:
    - ``"length_ratio"``: ``endpoint_distance / path_length``
    - ``"pca"``: first principal-component explained-variance ratio

    A perfectly straight segment scores close to 1.0. The returned reward is
    the arithmetic mean of the three arm values.

    If trisection fails the reward is 0.0.

    Args:
        (same as :func:`vessel_trisection_reward`)

    Returns:
        Tensor of shape ``(num_envs,)`` in ``[0, 1]``.
    """
    mode = str(straightness_mode).strip().lower()
    results = _get_trisection_cached(
        env, camera_cfg_name, vessel_label, gall_label,
        prefer_semantic_info, vessel_color, gall_color,
        color_tolerance, gall_dilation_radius, compute_every,
    )
    reward = torch.zeros(env.num_envs, device=env.device)
    for i, r in enumerate(results):
        if not r["success"]:
            continue
        if mode == "pca":
            straightness_values = r.get("straightness_pca", [0.0, 0.0, 0.0])
        else:
            straightness_values = r.get("straightness_length_ratio", r.get("straightness", [0.0, 0.0, 0.0]))
        if len(straightness_values) == 3:
            reward[i] = sum(straightness_values) / 3.0
    return reward


def vessel_branch_angle_range_penalty(
    env: ManagerBasedRLEnv,
    camera_cfg_name: str = "camera",
    vessel_label: str = "vessel",
    gall_label: str = "gall",
    vessel_color: tuple = (25, 82, 255),
    gall_color: tuple = (255, 105, 180),
    color_tolerance: int = 10,
    prefer_semantic_info: bool = True,
    gall_dilation_radius: int = 5,
    compute_every: int = 1,
    min_angle_deg: float = 30.0,
    max_angle_deg: float = 120.0,
) -> torch.Tensor:
    """Penalty when CD-CHD branch angle is outside [min_angle_deg, max_angle_deg].

    Penalty is linear in the amount of violation and normalized by the valid range.
    Invalid/failed trisection returns zero penalty.
    """
    valid_span = max(float(max_angle_deg - min_angle_deg), 1e-6)
    results = _get_trisection_cached(
        env,
        camera_cfg_name,
        vessel_label,
        gall_label,
        prefer_semantic_info,
        vessel_color,
        gall_color,
        color_tolerance,
        gall_dilation_radius,
        compute_every,
    )
    penalty = torch.zeros(env.num_envs, device=env.device)
    for i, r in enumerate(results):
        if not r.get("success", False):
            continue
        angle = float(r.get("cd_chd_angle_deg", -1.0))
        if angle < 0.0:
            continue
        below = max(float(min_angle_deg) - angle, 0.0)
        above = max(angle - float(max_angle_deg), 0.0)
        penalty[i] = (below + above) / valid_span
    return penalty


def _vessel_connectivity_reward_components(
    env: ManagerBasedRLEnv,
    camera_cfg_name: str = "camera",
    vessel_label: str = "vessel",
    gall_label: str = "gall",
    vessel_color: tuple = (25, 82, 255),
    gall_color: tuple = (255, 105, 180),
    color_tolerance: int = 10,
    prefer_semantic_info: bool = True,
    gall_dilation_radius: int = 8,
    compute_every: int = 1,
    stable_frames: int = 5,
    progress_target_frames: int | None = None,
    disconnect_decay: int = 1,
    hold_reward: float = 0.01,
    shaping_weight: float = 0.3,
    shaping_power: float = 2.0,
    break_penalty: float = 0.2,
    done_bonus: float = 1.5,
    reward_clip: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the 4 connectivity reward components (hold/progress/break/done).

    Let ``c_t`` be whether CD-only hard connectivity is satisfied at step ``t``.
    The shaping uses two counters:
    - strict counter ``m_t`` for completion, reset to 0 on disconnect
    - shaping counter ``n_t`` for progress, with soft decay on disconnect
    Progress normalization is intentionally fixed to ``progress_target_frames`` so
    curriculum stage changes only affect the done threshold.
    """
    strict_counter = _get_cd_only_connectivity_counter(
        env,
        camera_cfg_name,
        vessel_label,
        gall_label,
        vessel_color,
        gall_color,
        color_tolerance,
        prefer_semantic_info,
        gall_dilation_radius,
        compute_every,
        disconnect_decay,
    )

    shaping_counter, shaping_prev = _get_cd_only_connectivity_shaping_counter(
        env,
        camera_cfg_name,
        vessel_label,
        gall_label,
        vessel_color,
        gall_color,
        color_tolerance,
        prefer_semantic_info,
        gall_dilation_radius,
        compute_every,
        stable_frames,
        disconnect_decay,
    )

    connected_mask, prev_connected_mask = _get_cd_only_connected_masks(
        env,
        camera_cfg_name,
        vessel_label,
        gall_label,
        vessel_color,
        gall_color,
        color_tolerance,
        prefer_semantic_info,
        gall_dilation_radius,
        compute_every,
    )

    target_frames = max(1, int(stable_frames))
    progress_frames = max(1, int(target_frames if progress_target_frames is None else progress_target_frames))
    norm_curr = torch.clamp(shaping_counter.float() / float(progress_frames), min=0.0, max=1.0)
    norm_prev = torch.clamp(shaping_prev.float() / float(progress_frames), min=0.0, max=1.0)
    power = max(float(shaping_power), 1e-6)
    phi_curr = torch.pow(norm_curr, power)
    phi_prev = torch.pow(norm_prev, power)

    hold_component = float(hold_reward) * connected_mask.float()
    progress_component = float(shaping_weight) * (phi_curr - phi_prev)

    break_mask = prev_connected_mask & (~connected_mask)
    break_component = -float(break_penalty) * break_mask.float()

    done_mask = strict_counter == target_frames
    done_component = float(done_bonus) * done_mask.float()

    clip_value = float(reward_clip)
    if clip_value > 0.0:
        total = hold_component + progress_component + break_component + done_component
        abs_total = torch.abs(total)
        clip_mask = abs_total > clip_value
        if clip_mask.any():
            scale = torch.ones_like(total)
            scale[clip_mask] = clip_value / (abs_total[clip_mask] + 1e-8)
            hold_component = hold_component * scale
            progress_component = progress_component * scale
            break_component = break_component * scale
            done_component = done_component * scale

    return hold_component, progress_component, break_component, done_component


def vessel_gall_hard_connectivity_hold_reward(
    env: ManagerBasedRLEnv,
    camera_cfg_name: str = "camera",
    vessel_label: str = "vessel",
    gall_label: str = "gall",
    vessel_color: tuple = (25, 82, 255),
    gall_color: tuple = (255, 105, 180),
    color_tolerance: int = 10,
    prefer_semantic_info: bool = True,
    gall_dilation_radius: int = 8,
    compute_every: int = 1,
    stable_frames: int = 5,
    progress_target_frames: int | None = None,
    disconnect_decay: int = 1,
    hold_reward: float = 0.01,
    shaping_weight: float = 0.3,
    shaping_power: float = 2.0,
    break_penalty: float = 0.2,
    done_bonus: float = 1.5,
    reward_clip: float = 1.0,
) -> torch.Tensor:
    """Connectivity component 1: per-step hold reward ``r_hold * c_t``."""
    hold, _, _, _ = _vessel_connectivity_reward_components(
        env,
        camera_cfg_name,
        vessel_label,
        gall_label,
        vessel_color,
        gall_color,
        color_tolerance,
        prefer_semantic_info,
        gall_dilation_radius,
        compute_every,
        stable_frames,
        progress_target_frames,
        disconnect_decay,
        hold_reward,
        shaping_weight,
        shaping_power,
        break_penalty,
        done_bonus,
        reward_clip,
    )
    return hold


def vessel_gall_hard_connectivity_progress_reward(
    env: ManagerBasedRLEnv,
    camera_cfg_name: str = "camera",
    vessel_label: str = "vessel",
    gall_label: str = "gall",
    vessel_color: tuple = (25, 82, 255),
    gall_color: tuple = (255, 105, 180),
    color_tolerance: int = 10,
    prefer_semantic_info: bool = True,
    gall_dilation_radius: int = 8,
    compute_every: int = 1,
    stable_frames: int = 5,
    progress_target_frames: int | None = None,
    disconnect_decay: int = 1,
    hold_reward: float = 0.01,
    shaping_weight: float = 0.3,
    shaping_power: float = 2.0,
    break_penalty: float = 0.2,
    done_bonus: float = 1.5,
    reward_clip: float = 1.0,
) -> torch.Tensor:
    """Connectivity component 2: potential-based progress term."""
    _, progress, _, _ = _vessel_connectivity_reward_components(
        env,
        camera_cfg_name,
        vessel_label,
        gall_label,
        vessel_color,
        gall_color,
        color_tolerance,
        prefer_semantic_info,
        gall_dilation_radius,
        compute_every,
        stable_frames,
        progress_target_frames,
        disconnect_decay,
        hold_reward,
        shaping_weight,
        shaping_power,
        break_penalty,
        done_bonus,
        reward_clip,
    )
    return progress


def vessel_gall_hard_connectivity_break_reward(
    env: ManagerBasedRLEnv,
    camera_cfg_name: str = "camera",
    vessel_label: str = "vessel",
    gall_label: str = "gall",
    vessel_color: tuple = (25, 82, 255),
    gall_color: tuple = (255, 105, 180),
    color_tolerance: int = 10,
    prefer_semantic_info: bool = True,
    gall_dilation_radius: int = 8,
    compute_every: int = 1,
    stable_frames: int = 5,
    progress_target_frames: int | None = None,
    disconnect_decay: int = 1,
    hold_reward: float = 0.01,
    shaping_weight: float = 0.3,
    shaping_power: float = 2.0,
    break_penalty: float = 0.2,
    done_bonus: float = 1.5,
    reward_clip: float = 1.0,
) -> torch.Tensor:
    """Connectivity component 3: one-step break penalty."""
    _, _, break_comp, _ = _vessel_connectivity_reward_components(
        env,
        camera_cfg_name,
        vessel_label,
        gall_label,
        vessel_color,
        gall_color,
        color_tolerance,
        prefer_semantic_info,
        gall_dilation_radius,
        compute_every,
        stable_frames,
        progress_target_frames,
        disconnect_decay,
        hold_reward,
        shaping_weight,
        shaping_power,
        break_penalty,
        done_bonus,
        reward_clip,
    )
    return break_comp


def vessel_gall_hard_connectivity_done_bonus_reward(
    env: ManagerBasedRLEnv,
    camera_cfg_name: str = "camera",
    vessel_label: str = "vessel",
    gall_label: str = "gall",
    vessel_color: tuple = (25, 82, 255),
    gall_color: tuple = (255, 105, 180),
    color_tolerance: int = 10,
    prefer_semantic_info: bool = True,
    gall_dilation_radius: int = 8,
    compute_every: int = 1,
    stable_frames: int = 5,
    progress_target_frames: int | None = None,
    disconnect_decay: int = 1,
    hold_reward: float = 0.01,
    shaping_weight: float = 0.3,
    shaping_power: float = 2.0,
    break_penalty: float = 0.2,
    done_bonus: float = 1.5,
    reward_clip: float = 1.0,
) -> torch.Tensor:
    """Connectivity component 4: terminal-style bonus at target streak."""
    _, _, _, done = _vessel_connectivity_reward_components(
        env,
        camera_cfg_name,
        vessel_label,
        gall_label,
        vessel_color,
        gall_color,
        color_tolerance,
        prefer_semantic_info,
        gall_dilation_radius,
        compute_every,
        stable_frames,
        progress_target_frames,
        disconnect_decay,
        hold_reward,
        shaping_weight,
        shaping_power,
        break_penalty,
        done_bonus,
        reward_clip,
    )
    return done


def vessel_gall_hard_connectivity_reward(
    env: ManagerBasedRLEnv,
    camera_cfg_name: str = "camera",
    vessel_label: str = "vessel",
    gall_label: str = "gall",
    vessel_color: tuple = (25, 82, 255),
    gall_color: tuple = (255, 105, 180),
    color_tolerance: int = 10,
    prefer_semantic_info: bool = True,
    gall_dilation_radius: int = 8,
    compute_every: int = 1,
    stable_frames: int = 5,
    progress_target_frames: int | None = None,
    disconnect_decay: int = 1,
    hold_reward: float = 0.01,
    shaping_weight: float = 0.3,
    shaping_power: float = 2.0,
    break_penalty: float = 0.2,
    done_bonus: float = 1.5,
    reward_clip: float = 1.0,
) -> torch.Tensor:
    """Legacy summed connectivity reward (sum of 4 exposed components)."""
    hold, progress, break_comp, done = _vessel_connectivity_reward_components(
        env,
        camera_cfg_name,
        vessel_label,
        gall_label,
        vessel_color,
        gall_color,
        color_tolerance,
        prefer_semantic_info,
        gall_dilation_radius,
        compute_every,
        stable_frames,
        progress_target_frames,
        disconnect_decay,
        hold_reward,
        shaping_weight,
        shaping_power,
        break_penalty,
        done_bonus,
        reward_clip,
    )
    return hold + progress + break_comp + done


def vessel_cd_gall_boundary_distance(
    env: ManagerBasedRLEnv,
    camera_cfg_name: str = "camera",
    vessel_label: str = "vessel",
    gall_label: str = "gall",
    vessel_color: tuple = (25, 82, 255),
    gall_color: tuple = (255, 105, 180),
    color_tolerance: int = 10,
    prefer_semantic_info: bool = True,
    gall_dilation_radius: int = 8,
    compute_every: int = 1,
    max_distance: float = 100.0,
) -> torch.Tensor:
    """Distance metric used by reward3 in the offline analysis pipeline.

    Returns the minimum Euclidean distance in pixels from the rendered CD arm
    mask to the gall boundary. Lower is better. Failed trisection falls back to
    ``max_distance``.
    """
    results = _get_trisection_cached(
        env, camera_cfg_name, vessel_label, gall_label,
        prefer_semantic_info, vessel_color, gall_color,
        color_tolerance, gall_dilation_radius, compute_every,
    )
    reward = torch.full((env.num_envs,), float(max_distance), device=env.device)
    for i, r in enumerate(results):
        dist = float(r.get("cd_to_gall_min_distance_px", -1.0))
        if r["success"] and dist >= 0.0:
            reward[i] = min(dist, max_distance)
    return reward


def vessel_gall_single_connection_reward(
    env: ManagerBasedRLEnv,
    camera_cfg_name: str = "camera",
    vessel_label: str = "vessel",
    gall_label: str = "gall",
    vessel_color: tuple = (25, 82, 255),
    gall_color: tuple = (255, 105, 180),
    color_tolerance: int = 10,
    prefer_semantic_info: bool = True,
    gall_dilation_radius: int = 5,
    compute_every: int = 1,
) -> torch.Tensor:
    """Task-completion reward: exactly 1 of 3 vessel segments touches gall.

    After the vessel skeleton is split into 3 arms, each arm's endpoint
    region is checked for adjacency to the dilated gall (gallbladder) mask.
    If *exactly one* segment is adjacent the reward is 1.0 (task complete);
    otherwise it is 0.0.

    Args:
        (same as :func:`vessel_trisection_reward`)

    Returns:
        Tensor of shape ``(num_envs,)`` with value 1.0 or 0.0 per env.
    """
    results = _get_trisection_cached(
        env, camera_cfg_name, vessel_label, gall_label,
        prefer_semantic_info, vessel_color, gall_color,
        color_tolerance, gall_dilation_radius, compute_every,
    )
    reward = torch.zeros(env.num_envs, device=env.device)
    for i, r in enumerate(results):
        if r["success"] and r.get("gall_connection_count") == 1:
            reward[i] = 1.0
    return reward
