# Copyright (c) 2026, The ORBIT-Surgical Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pinocchio-backed task-space actions."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
import xml.etree.ElementTree as ET

import numpy as np
import pinocchio as pin
import torch
from pinocchio.robot_wrapper import RobotWrapper

import isaaclab.utils.math as math_utils
from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass


def _read_urdf_joint_names(urdf_path: str) -> list[str]:
    root = ET.parse(urdf_path).getroot()
    return [joint.attrib["name"] for joint in root.findall("joint") if joint.attrib.get("type") != "fixed"]


@configclass
class PinocchioInverseKinematicsActionCfg(ActionTermCfg):
    """Absolute pose IK action solved with Pinocchio DLS."""

    class_type: type[ActionTerm] = MISSING

    joint_names: list[str] = MISSING
    all_joint_names: list[str] = MISSING
    urdf_path: str = MISSING
    mesh_path: str | None = None
    base_body_name: str = MISSING
    ee_frame_name: str = MISSING
    position_gain: float = 0.5
    orientation_gain: float = 0.5
    damping: float = 0.05
    max_delta: float = 0.08


class PinocchioInverseKinematicsAction(ActionTerm):
    """Use Pinocchio forward kinematics/Jacobians to track an absolute end-effector pose."""

    cfg: PinocchioInverseKinematicsActionCfg
    _asset: Articulation

    def __init__(self, cfg: PinocchioInverseKinematicsActionCfg, env):
        super().__init__(cfg, env)

        self._joint_ids, self._joint_names = self._asset.find_joints(self.cfg.joint_names, preserve_order=True)
        self._all_joint_ids, self._all_joint_names = self._asset.find_joints(
            self.cfg.all_joint_names, preserve_order=True
        )
        base_body_ids, base_body_names = self._asset.find_bodies(self.cfg.base_body_name)
        if len(base_body_ids) != 1:
            raise ValueError(
                f"Expected one base body named {self.cfg.base_body_name!r}, found {len(base_body_ids)}: {base_body_names}"
            )
        self._base_body_id = base_body_ids[0]

        self._robot_wrapper = (
            RobotWrapper.BuildFromURDF(self.cfg.urdf_path, self.cfg.mesh_path)
            if self.cfg.mesh_path
            else RobotWrapper.BuildFromURDF(self.cfg.urdf_path)
        )
        self._model = self._robot_wrapper.model
        self._data = self._model.createData()
        self._pin_joint_names = _read_urdf_joint_names(self.cfg.urdf_path)
        self._frame_id = self._model.getFrameId(self.cfg.ee_frame_name)
        if self._frame_id >= len(self._model.frames):
            raise ValueError(f"Frame {self.cfg.ee_frame_name!r} not found in {self.cfg.urdf_path}")

        self._pin_to_asset_all = [self._all_joint_names.index(joint_name) for joint_name in self._pin_joint_names]
        self._controlled_pin_indices = [self._pin_joint_names.index(joint_name) for joint_name in self._joint_names]

        lower = []
        upper = []
        for joint_name in self._joint_names:
            pin_index = self._pin_joint_names.index(joint_name)
            lower.append(float(self._model.lowerPositionLimit[pin_index]))
            upper.append(float(self._model.upperPositionLimit[pin_index]))
        self._joint_lower = torch.tensor(lower, device=self.device)
        self._joint_upper = torch.tensor(upper, device=self.device)

        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, len(self._joint_ids), device=self.device)
        self._identity = np.eye(6)

    @property
    def action_dim(self) -> int:
        return 7

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions

    def apply_actions(self) -> None:
        all_joint_pos = self._asset.data.joint_pos[:, self._all_joint_ids]
        base_state = self._asset.data.body_link_state_w[:, self._base_body_id, :7]
        base_pos_w = base_state[:, :3]
        base_quat_w = base_state[:, 3:7]

        target_pos_w = self._raw_actions[:, :3]
        target_quat_w = self._raw_actions[:, 3:7]
        target_pos_b, target_quat_b = math_utils.subtract_frame_transforms(
            base_pos_w, base_quat_w, target_pos_w, target_quat_w
        )
        target_rot_b = math_utils.matrix_from_quat(target_quat_b).detach().cpu().numpy()
        target_pos_b_np = target_pos_b.detach().cpu().numpy()
        all_joint_pos_np = all_joint_pos.detach().cpu().numpy()

        joint_targets = []
        for env_id in range(self.num_envs):
            q_asset = all_joint_pos_np[env_id]
            q_pin = q_asset[self._pin_to_asset_all].astype(np.float64)

            pin.forwardKinematics(self._model, self._data, q_pin)
            pin.updateFramePlacements(self._model, self._data)

            current_pose = self._data.oMf[self._frame_id]
            desired_pose = pin.SE3(target_rot_b[env_id], target_pos_b_np[env_id])
            err = pin.log6(current_pose.inverse() * desired_pose).vector
            err[:3] *= self.cfg.position_gain
            err[3:] *= self.cfg.orientation_gain

            jacobian = pin.computeFrameJacobian(self._model, self._data, q_pin, self._frame_id, pin.ReferenceFrame.LOCAL)
            lambda_sq = self.cfg.damping * self.cfg.damping
            dq = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + lambda_sq * self._identity, err)
            dq = np.clip(dq, -self.cfg.max_delta, self.cfg.max_delta)

            q_next = q_pin.copy()
            q_next[self._controlled_pin_indices] += dq[self._controlled_pin_indices]
            controlled_next = torch.tensor(
                q_next[self._controlled_pin_indices], device=self.device, dtype=torch.float32
            )
            controlled_next = torch.clamp(controlled_next, self._joint_lower, self._joint_upper)
            joint_targets.append(controlled_next)

        self._processed_actions = torch.stack(joint_targets)
        self._asset.set_joint_position_target(self._processed_actions, self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._raw_actions[env_ids] = torch.zeros(self.action_dim, device=self.device)


PinocchioInverseKinematicsActionCfg.class_type = PinocchioInverseKinematicsAction
