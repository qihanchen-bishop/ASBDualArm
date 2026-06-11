from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.log

import isaaclab.utils.math as math_utils
import isaaclab.utils.string as string_utils
from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from . import actions_cfg

from msr.config.controllers import DifferentialIKWithSoftRCMController


def _clamp_ee_translation_to_workspace_world(
    asset: Articulation,
    cfg,
    ee_pos_curr_b: torch.Tensor,
    processed_actions: torch.Tensor,
) -> torch.Tensor:
    """Clamp relative translation action so desired EE position stays inside world-space bounds."""
    if not getattr(cfg, "enforce_workspace_bounds", False):
        return processed_actions
    if cfg.x_bounds_world is None or cfg.y_bounds_world is None or cfg.z_bounds_world is None:
        return processed_actions

    root_pos_w = asset.data.root_pos_w
    root_quat_w = asset.data.root_quat_w

    # Relative IK uses translational delta in the robot root frame.
    delta_pos_b = processed_actions[:, :3]
    ee_pos_des_b = ee_pos_curr_b + delta_pos_b
    ee_pos_des_w, _ = math_utils.combine_frame_transforms(root_pos_w, root_quat_w, ee_pos_des_b)

    clamped_w = ee_pos_des_w.clone()
    clamped_w[:, 0] = torch.clamp(clamped_w[:, 0], min=cfg.x_bounds_world[0], max=cfg.x_bounds_world[1])
    clamped_w[:, 1] = torch.clamp(clamped_w[:, 1], min=cfg.y_bounds_world[0], max=cfg.y_bounds_world[1])
    clamped_w[:, 2] = torch.clamp(clamped_w[:, 2], min=cfg.z_bounds_world[0], max=cfg.z_bounds_world[1])

    ee_pos_clamped_b, _ = math_utils.subtract_frame_transforms(root_pos_w, root_quat_w, clamped_w)
    processed_actions[:, :3] = ee_pos_clamped_b - ee_pos_curr_b
    return processed_actions


class DifferentialInverseKinematicsWorkspaceClampedAction(DifferentialInverseKinematicsAction):
    """Differential IK action with hard world-space workspace clamp on EE target position."""

    def process_actions(self, actions: torch.Tensor):
        # store the raw actions
        self._raw_actions[:] = actions
        self._processed_actions[:] = self.raw_actions * self._scale
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )

        # obtain current EE pose in root frame
        ee_pos_curr, ee_quat_curr = self._compute_frame_pose()
        self._processed_actions = _clamp_ee_translation_to_workspace_world(
            self._asset, self.cfg, ee_pos_curr, self._processed_actions
        )

        # set command into controller
        self._ik_controller.set_command(self._processed_actions, ee_pos_curr, ee_quat_curr)


class DifferentialInverseKinematicsWithSoftRCMAction(ActionTerm):
    r"""Inverse Kinematics action term under RCM constraint.

    This action term performs pre-processing of the raw actions using scaling transformation.

    .. math::
        \text{action} = \text{scaling} \times \text{input action}
        \text{joint position} = J^{-} \times \text{action}

    where :math:`\text{scaling}` is the scaling applied to the input action, and :math:`\text{input action}`
    is the input action from the user, :math:`J` is the Jacobian over the articulation's actuated joints,
    and \text{joint position} is the desired joint position command for the articulation's joints.
    """

    cfg: actions_cfg.DifferentialInverseKinematicsWithSoftRCMActionCfg
    """The configuration of the action term."""
    _asset: Articulation
    """The articulation asset on which the action term is applied."""
    _scale: torch.Tensor
    """The scaling factor applied to the input action. Shape is (1, action_dim)."""
    _clip: torch.Tensor
    """The clip applied to the input action."""

    def __init__(self, cfg: actions_cfg.DifferentialInverseKinematicsWithSoftRCMActionCfg, env: ManagerBasedEnv):
        # initialize the action term
        super().__init__(cfg, env)

        # resolve the joints over which the action term is applied
        self._joint_ids, self._joint_names = self._asset.find_joints(self.cfg.joint_names)
        self._num_joints = len(self._joint_ids)
        # parse the body index
        body_ids, body_names = self._asset.find_bodies(self.cfg.body_name)
        f1_ids, f1_names = self._asset.find_bodies(self.cfg.f1_name)
        f2_ids, f2_names = self._asset.find_bodies(self.cfg.f2_name)
        if len(body_ids) != 1:
            raise ValueError(
                f"Expected one match for the body name: {self.cfg.body_name}. Found {len(body_ids)}: {body_names}."
            )
        # save only the first body index
        self._body_idx = body_ids[0]
        self._body_name = body_names[0]
        self._f1_idx = f1_ids[0]
        self._f1_name = f1_names[0]
        self._f2_idx = f2_ids[0]
        self._f2_name = f2_names[0]
        # check if articulation is fixed-base
        # if fixed-base then the jacobian for the base is not computed
        # this means that number of bodies is one less than the articulation's number of bodies
        if self._asset.is_fixed_base:
            self._jacobi_body_idx = self._body_idx - 1
            self._jacobi_f1_idx = self._f1_idx - 1
            self._jacobi_f2_idx = self._f2_idx - 1
            self._jacobi_joint_ids = self._joint_ids
        else:
            self._jacobi_body_idx = self._body_idx
            self._jacobi_f1_idx = self._f1_idx
            self._jacobi_f2_idx = self._f2_idx
            self._jacobi_joint_ids = [i + 6 for i in self._joint_ids]

        # log info for debugging
        omni.log.info(
            f"Resolved joint names for the action term {self.__class__.__name__}:"
            f" {self._joint_names} [{self._joint_ids}]"
        )
        omni.log.info(
            f"Resolved body name for the action term {self.__class__.__name__}: {self._body_name} [{self._body_idx}]"
        )
        # Avoid indexing across all joints for efficiency
        if self._num_joints == self._asset.num_joints:
            self._joint_ids = slice(None)

        # create the differential IK controller
        self._ik_controller = DifferentialIKWithSoftRCMController(
            cfg=self.cfg.controller, num_envs=self.num_envs, device=self.device
        )

        # create tensors for raw and processed actions
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self.raw_actions)

        # save the scale as tensors
        self._scale = torch.zeros((self.num_envs, self.action_dim), device=self.device)
        self._scale[:] = torch.tensor(self.cfg.scale, device=self.device)

        # convert the fixed offsets to torch tensors of batched shape
        if self.cfg.body_offset is not None:
            self._offset_pos = torch.tensor(self.cfg.body_offset.pos, device=self.device).repeat(self.num_envs, 1)
            self._offset_rot = torch.tensor(self.cfg.body_offset.rot, device=self.device).repeat(self.num_envs, 1)
        else:
            self._offset_pos, self._offset_rot = None, None

        # parse clip
        if self.cfg.clip is not None:
            if isinstance(cfg.clip, dict):
                self._clip = torch.tensor([[-float("inf"), float("inf")]], device=self.device).repeat(
                    self.num_envs, self.action_dim, 1
                )
                index_list, _, value_list = string_utils.resolve_matching_names_values(self.cfg.clip, self._joint_names)
                self._clip[:, index_list] = torch.tensor(value_list, device=self.device)
            else:
                raise ValueError(f"Unsupported clip type: {type(cfg.clip)}. Supported types are dict.")
        
        # set the RCM position
        if self.cfg.rcm_pos is not None:
            # Bin: To be checked.
            self.rcm_pos = torch.tensor(self.cfg.rcm_pos, device=self.device).repeat(self.num_envs, 1)
            self.rcm_beta = None  # need to re-calculate
        else:
            self.rcm_pos = None
            self.rcm_beta = self.cfg.rcm_beta * torch.ones(self.num_envs, 1, device=self.device)

        # set the refer orientation to default None
        self.ee_quat_ref = None
        
    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return self._ik_controller.action_dim

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def jacobian_w(self) -> torch.Tensor:
        return self._asset.root_physx_view.get_jacobians()[:, self._jacobi_body_idx, :, self._jacobi_joint_ids]

    @property
    def jacobian_b(self) -> torch.Tensor:
        jacobian = self.jacobian_w
        base_rot = self._asset.data.root_quat_w
        base_rot_matrix = math_utils.matrix_from_quat(math_utils.quat_inv(base_rot))
        jacobian[:, :3, :] = torch.bmm(base_rot_matrix, jacobian[:, :3, :])
        jacobian[:, 3:, :] = torch.bmm(base_rot_matrix, jacobian[:, 3:, :])
        return jacobian

    """
    Operations.
    """

    def process_actions(self, actions: torch.Tensor):
        # store the raw actions
        self._raw_actions[:] = actions
        self._processed_actions[:] = self.raw_actions * self._scale
        # print('cfg_clip: ', self.cfg.clip)
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )
        # obtain quantities from simulation
        ee_pos_curr, ee_quat_curr = self._compute_frame_pose()
        # initialize the RCM position. Only need to calculate in the begining.
        if self.rcm_pos is None:
            print("RCM pos to be initialized ...")
            self.rcm_pos = self._compute_rcm_pos()  # torch.tensor(self.cfg.rcm_pos, device=self.device).repeat(self.num_envs, 1)
            self._ik_controller.set_rcm_target(self.rcm_pos, self.rcm_beta)
            print("RCM pos initialized.")
        if self.ee_quat_ref is None and self._body_name.find("ecm") != -1: 
            print("EE ori to be initialized ...")
            self.ee_quat_ref = ee_quat_curr.clone()
            self._ik_controller.set_ori_target(self.ee_quat_ref)
            print("EE ori initialized.")

        # set command into controller
        self._ik_controller.set_command(self._processed_actions, ee_pos_curr, ee_quat_curr)


class DifferentialInverseKinematicsWithSoftRCMWorkspaceClampedAction(DifferentialInverseKinematicsWithSoftRCMAction):
    """Soft-RCM differential IK action with hard world-space workspace clamp."""

    def process_actions(self, actions: torch.Tensor):
        # store the raw actions
        self._raw_actions[:] = actions
        self._processed_actions[:] = self.raw_actions * self._scale
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )
        # obtain quantities from simulation
        ee_pos_curr, ee_quat_curr = self._compute_frame_pose()
        # initialize the RCM position. Only need to calculate in the begining.
        if self.rcm_pos is None:
            print("RCM pos to be initialized ...")
            self.rcm_pos = self._compute_rcm_pos()
            self._ik_controller.set_rcm_target(self.rcm_pos, self.rcm_beta)
            print("RCM pos initialized.")
        if self.ee_quat_ref is None and self._body_name.find("ecm") != -1:
            print("EE ori to be initialized ...")
            self.ee_quat_ref = ee_quat_curr.clone()
            self._ik_controller.set_ori_target(self.ee_quat_ref)
            print("EE ori initialized.")

        self._processed_actions = _clamp_ee_translation_to_workspace_world(
            self._asset, self.cfg, ee_pos_curr, self._processed_actions
        )

        # set command into controller
        self._ik_controller.set_command(self._processed_actions, ee_pos_curr, ee_quat_curr)

    def apply_actions(self):
        # obtain quantities from simulation
        ee_pos_curr, ee_quat_curr = self._compute_frame_pose()
        rcm_pos_b = self._compute_rcm_pos()
        joint_pos = self._asset.data.joint_pos[:, self._joint_ids]
        # compute the delta in joint-space
        if ee_quat_curr.norm() != 0:
            jacobian = self._compute_full_jacobian()
            joint_pos_des, rcm_beta_des = self._ik_controller.compute(ee_pos_curr, ee_quat_curr, rcm_pos_b, jacobian, joint_pos, self.rcm_beta)
        else:
            print('debug: attention ...')
            joint_pos_des = joint_pos.clone()
        # set the joint position command
        self._asset.set_joint_position_target(joint_pos_des, self._joint_ids)
        self.rcm_beta = rcm_beta_des

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._raw_actions[env_ids] = 0.0

    """
    Helper functions.
    """

    def _compute_rcm_pos(self) -> torch.Tensor:
        """Computes the RCM position in the robot base frame.

        Returns:
            The RCM position in the world frame.
        """
        # obtain quantities from simulation
        f1_pos_w = self._asset.data.body_pos_w[:, self._f1_idx]
        f1_quat_w = self._asset.data.body_quat_w[:, self._f1_idx]
        f2_pos_w = self._asset.data.body_pos_w[:, self._f2_idx]
        f2_quat_w = self._asset.data.body_quat_w[:, self._f2_idx]
        root_pos_w = self._asset.data.root_pos_w
        root_quat_w = self._asset.data.root_quat_w
        # compute the pose of the body in the root frame
        f1_pos_b, f1_quat_b = math_utils.subtract_frame_transforms(root_pos_w, root_quat_w, f1_pos_w, f1_quat_w)
        f2_pos_b, f2_quat_b = math_utils.subtract_frame_transforms(root_pos_w, root_quat_w, f2_pos_w, f2_quat_w)
        # compute the RCM position in the world frame
        rcm_pos_b = f1_pos_b + self.rcm_beta * (f2_pos_b - f1_pos_b)

        return rcm_pos_b

    def _compute_frame_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes the pose of the target frame in the root frame.

        Returns:
            A tuple of the body's position and orientation in the root frame.
        """
        # obtain quantities from simulation
        ee_pos_w = self._asset.data.body_pos_w[:, self._body_idx]
        ee_quat_w = self._asset.data.body_quat_w[:, self._body_idx]
        root_pos_w = self._asset.data.root_pos_w
        root_quat_w = self._asset.data.root_quat_w
        # compute the pose of the body in the root frame
        ee_pose_b, ee_quat_b = math_utils.subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)
        # account for the offset
        if self.cfg.body_offset is not None:
            ee_pose_b, ee_quat_b = math_utils.combine_frame_transforms(
                ee_pose_b, ee_quat_b, self._offset_pos, self._offset_rot
            )

        return ee_pose_b, ee_quat_b

    def _compute_frame_jacobian(self):
        """Computes the geometric Jacobian of the target frame in the root frame.

        This function accounts for the target frame offset and applies the necessary transformations to obtain
        the right Jacobian from the parent body Jacobian.
        """
        # read the parent jacobian
        jacobian = self.jacobian_b
        # account for the offset
        if self.cfg.body_offset is not None:
            # Modify the jacobian to account for the offset
            # -- translational part
            # v_link = v_ee + w_ee x r_link_ee = v_J_ee * q + w_J_ee * q x r_link_ee
            #        = (v_J_ee + w_J_ee x r_link_ee ) * q
            #        = (v_J_ee - r_link_ee_[x] @ w_J_ee) * q
            jacobian[:, 0:3, :] += torch.bmm(-math_utils.skew_symmetric_matrix(self._offset_pos), jacobian[:, 3:, :])
            # -- rotational part
            # w_link = R_link_ee @ w_ee
            # jacobian[:, 3:, :] = torch.bmm(math_utils.matrix_from_quat(self._offset_rot), jacobian[:, 3:, :])

        return jacobian
    
    def _compute_full_jacobian(self):
        """Compute the full jacobian including the robots end-effector and the RCM."""
        ee_jacobian_b = self._compute_frame_jacobian()
        # compute the jacobian of the RCM
        root_rot_matrix = math_utils.matrix_from_quat(math_utils.quat_inv(self._asset.data.root_quat_w))
        f1_jacobian_trans_w = self._asset.root_physx_view.get_jacobians()[:, self._jacobi_f1_idx, :3, self._jacobi_joint_ids]
        f1_jacobian_trans_b = torch.bmm(root_rot_matrix, f1_jacobian_trans_w)
        f2_jacobian_trans_w = self._asset.root_physx_view.get_jacobians()[:, self._jacobi_f2_idx, :3, self._jacobi_joint_ids]
        f2_jacobian_trans_b = torch.bmm(root_rot_matrix, f2_jacobian_trans_w)
        rcm_jacobian_b = f1_jacobian_trans_b + self.rcm_beta.unsqueeze(-1) * (f2_jacobian_trans_b - f1_jacobian_trans_b)
        
        # compute current pose of the end-effector, f1 frame, and f2 frame
        root_pos_w = self._asset.data.root_pos_w
        root_quat_w = self._asset.data.root_quat_w
        f1_pos_w = self._asset.data.body_pos_w[:, self._f1_idx]
        f1_quat_w = self._asset.data.body_quat_w[:, self._f1_idx]
        f2_pos_w = self._asset.data.body_pos_w[:, self._f2_idx]
        f2_quat_w = self._asset.data.body_quat_w[:, self._f2_idx]

        f1_pos_b, f1_quat_b = math_utils.subtract_frame_transforms(root_pos_w, root_quat_w, f1_pos_w, f1_quat_w)
        f2_pos_b, f2_quat_b = math_utils.subtract_frame_transforms(root_pos_w, root_quat_w, f2_pos_w, f2_quat_w)
        # calculate delta_p in J_rcm
        # rcm_pos_b = f1_pos_b + self.rcm_beta * (f2_pos_b - f1_pos_b)
        delta_p = f2_pos_b - f1_pos_b
        delta_p = delta_p.unsqueeze(-1)

        # merge the jacobian
        Z1 = torch.zeros((self.num_envs, 6, 1), device=self.device)
        Jm1 = torch.concatenate((ee_jacobian_b, Z1), dim=-1)
        Jm2 = torch.concatenate((rcm_jacobian_b, delta_p), dim=-1)
        full_jacobian_b = torch.concatenate((Jm1, Jm2), dim=1)  # shape: (num_envs, 6+3, num_joints(12) + 1)
        
        # # debug:
        # full_jacobian_b[:, 3] *= -1.0
        # full_jacobian_b[:, 5] *= -1.0

        return full_jacobian_b

