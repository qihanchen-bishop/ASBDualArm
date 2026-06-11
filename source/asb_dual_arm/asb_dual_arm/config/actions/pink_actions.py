# Copyright (c) 2026, The ORBIT-Surgical Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pink IK action variants used by the MSR extension."""

from __future__ import annotations

from dataclasses import MISSING
import xml.etree.ElementTree as ET

import numpy as np
import pinocchio as pin
import torch
from pink import solve_ik
from pink.configuration import Configuration
from pink.exceptions import FrameNotFound
from pinocchio.robot_wrapper import RobotWrapper

from isaaclab.controllers.pink_ik import PinkIKController, NullSpacePostureTask
from isaaclab.envs.mdp.actions.pink_actions_cfg import PinkInverseKinematicsActionCfg
from isaaclab.envs.mdp.actions.pink_task_space_actions import PinkInverseKinematicsAction
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils.string import resolve_matching_names_values
from isaaclab.utils import configclass


def _read_urdf_joint_names(urdf_path: str) -> list[str]:
    root = ET.parse(urdf_path).getroot()
    return [joint.attrib["name"] for joint in root.findall("joint") if joint.attrib.get("type") != "fixed"]


class CompatiblePinkKinematicsConfiguration(Configuration):
    """Pink kinematics configuration that avoids Pinocchio std::vector string conversion."""

    def __init__(
        self,
        controlled_joint_names: list[str],
        urdf_path: str,
        mesh_path: str | None = None,
        copy_data: bool = True,
        forward_kinematics: bool = True,
    ):
        self._controlled_joint_names = controlled_joint_names
        self._all_joint_names = _read_urdf_joint_names(urdf_path)

        if mesh_path:
            self.robot_wrapper = RobotWrapper.BuildFromURDF(urdf_path, mesh_path)
        else:
            self.robot_wrapper = RobotWrapper.BuildFromURDF(urdf_path)
        self.full_model = self.robot_wrapper.model
        self.full_data = self.robot_wrapper.data
        self.full_q = self.robot_wrapper.q0

        self._controlled_joint_indices = [
            idx for idx, joint_name in enumerate(self._all_joint_names) if joint_name in self._controlled_joint_names
        ]

        joints_to_lock = []
        for joint_name in self._all_joint_names:
            if joint_name not in self._controlled_joint_names:
                joints_to_lock.append(self.full_model.getJointId(joint_name))

        if len(joints_to_lock) == 0:
            self.controlled_model = self.full_model
            self.controlled_data = self.full_data
            self.controlled_q = self.full_q
        else:
            self.controlled_model = pin.buildReducedModel(self.full_model, joints_to_lock, self.full_q)
            self.controlled_data = self.controlled_model.createData()
            self.controlled_q = self.full_q[self._controlled_joint_indices]

        super().__init__(self.controlled_model, self.controlled_data, self.controlled_q, copy_data, forward_kinematics)

    def update(self, q: np.ndarray | None = None) -> None:
        if q is not None and len(q) != len(self._all_joint_names):
            raise ValueError("q must have the same length as the number of joints in the model")
        if q is not None:
            super().update(q[self._controlled_joint_indices])
            q_readonly = q.copy()
            q_readonly.setflags(write=False)
            self.full_q = q_readonly
            pin.computeJointJacobians(self.full_model, self.full_data, q)
            pin.updateFramePlacements(self.full_model, self.full_data)
        else:
            super().update()
            pin.computeJointJacobians(self.full_model, self.full_data, self.full_q)
            pin.updateFramePlacements(self.full_model, self.full_data)

    def get_frame_jacobian(self, frame: str) -> np.ndarray:
        if not self.full_model.existFrame(frame):
            raise FrameNotFound(frame, self.full_model.frames)
        frame_id = self.full_model.getFrameId(frame)
        jacobian = pin.getFrameJacobian(self.full_model, self.full_data, frame_id, pin.ReferenceFrame.LOCAL)
        return jacobian[:, self._controlled_joint_indices]

    def get_transform_frame_to_world(self, frame: str) -> pin.SE3:
        frame_id = self.full_model.getFrameId(frame)
        try:
            return self.full_data.oMf[frame_id].copy()
        except IndexError as index_error:
            raise FrameNotFound(frame, self.full_model.frames) from index_error

    def check_limits(self, tol: float = 1e-6, safety_break: bool = True) -> None:
        if safety_break:
            super().check_limits(tol, safety_break)

    @property
    def controlled_joint_names_pinocchio_order(self) -> list[str]:
        return [self._all_joint_names[i] for i in self._controlled_joint_indices]

    @property
    def all_joint_names_pinocchio_order(self) -> list[str]:
        return self._all_joint_names


class CompatiblePinkIKController(PinkIKController):
    """Pink IK controller that uses :class:`CompatiblePinkKinematicsConfiguration`."""

    def __init__(self, cfg, robot_cfg, device: str, controlled_joint_indices: list[int]):
        if cfg.joint_names is None:
            raise ValueError("joint_names must be provided in the configuration")
        if cfg.all_joint_names is None:
            raise ValueError("all_joint_names must be provided in the configuration")

        self.cfg = cfg
        self.device = device
        self.controlled_joint_indices = controlled_joint_indices
        self._validate_consistency(cfg, controlled_joint_indices)

        self.pink_configuration = CompatiblePinkKinematicsConfiguration(
            urdf_path=cfg.urdf_path,
            mesh_path=cfg.mesh_path,
            controlled_joint_names=cfg.joint_names,
        )

        pink_joint_names = self.pink_configuration.all_joint_names_pinocchio_order
        joint_pos_dict = robot_cfg.init_state.joint_pos
        indices, _, values = resolve_matching_names_values(
            joint_pos_dict, pink_joint_names, preserve_order=False, strict=False
        )
        self.init_joint_positions = np.zeros(len(pink_joint_names))
        self.init_joint_positions[indices] = np.array(values)

        for task in cfg.variable_input_tasks:
            if isinstance(task, NullSpacePostureTask):
                task.set_target(self.init_joint_positions)
                continue
            task.set_target_from_configuration(self.pink_configuration)
        for task in cfg.fixed_input_tasks:
            task.set_target_from_configuration(self.pink_configuration)

        self._setup_joint_ordering_mappings()


class MultiAssetPinkInverseKinematicsAction(PinkInverseKinematicsAction):
    """Pink IK action that uses the configured articulation asset instead of ``scene.cfg.robot``.

    Isaac Lab's upstream Pink action is written for scene configs with a single ``robot`` field. The
    dual-arm SO101 scene exposes two articulation fields, ``robot_1`` and ``robot_2``. This subclass keeps
    the upstream runtime behavior, but passes ``scene.cfg.<articulation_name>`` into the Pink controller.
    """

    cfg: "MultiAssetPinkInverseKinematicsActionCfg"

    def _initialize_ik_controllers(self) -> None:
        assert self._env.num_envs > 0, "Number of environments specified are less than 1."

        robot_cfg = getattr(self._env.scene.cfg, self.cfg.controller.articulation_name)
        self._ik_controllers = []
        for _ in range(self._env.num_envs):
            self._ik_controllers.append(
                CompatiblePinkIKController(
                    cfg=self.cfg.controller.copy(),
                    robot_cfg=robot_cfg,
                    device=self.device,
                    controlled_joint_indices=self._isaaclab_controlled_joint_ids,
                )
            )

    def _apply_gravity_compensation(self) -> None:
        # SO101 is loaded from an existing scene USD with spawn=None. In that setup there is no spawn
        # rigid_props object to inspect, so only use the upstream gravity path when such props exist.
        spawn_cfg = getattr(self._asset.cfg, "spawn", None)
        rigid_props = getattr(spawn_cfg, "rigid_props", None)
        if rigid_props is None:
            return
        super()._apply_gravity_compensation()


@configclass
class MultiAssetPinkInverseKinematicsActionCfg(PinkInverseKinematicsActionCfg):
    """Configuration for :class:`MultiAssetPinkInverseKinematicsAction`."""

    class_type: type[ActionTerm] = MultiAssetPinkInverseKinematicsAction

    pink_controlled_joint_names: list[str] = MISSING
    hand_joint_names: list[str] = MISSING
