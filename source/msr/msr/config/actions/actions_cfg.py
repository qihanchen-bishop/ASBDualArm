from dataclasses import MISSING

from isaaclab.utils import configclass
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg

from .task_space_actions import (
    DifferentialInverseKinematicsWithSoftRCMAction,
    DifferentialInverseKinematicsWorkspaceClampedAction,
    DifferentialInverseKinematicsWithSoftRCMWorkspaceClampedAction,
)
from msr.config.controllers import DifferentialIKWithSoftRCMControllerCfg


@configclass
class DifferentialInverseKinematicsWithSoftRCMActionCfg(ActionTermCfg):
    """Configuration for inverse differential kinematics action term under RCM constraint.

    See :class:`DifferentialInverseKinematicsAction` for more details.
    """

    @configclass
    class OffsetCfg:
        """The offset pose from parent frame to child frame.

        On many robots, end-effector frames are fictitious frames that do not have a corresponding
        rigid body. In such cases, it is easier to define this transform w.r.t. their parent rigid body.
        For instance, for the Franka Emika arm, the end-effector is defined at an offset to the the
        "panda_hand" frame.
        """

        pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
        """Translation w.r.t. the parent frame. Defaults to (0.0, 0.0, 0.0)."""
        rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
        """Quaternion rotation ``(w, x, y, z)`` w.r.t. the parent frame. Defaults to (1.0, 0.0, 0.0, 0.0)."""

    class_type: type[ActionTerm] = DifferentialInverseKinematicsWithSoftRCMAction

    joint_names: list[str] = MISSING
    """List of joint names or regex expressions that the action will be mapped to."""
    body_name: str = MISSING
    """Name of the body or frame for which IK is performed."""
    body_offset: OffsetCfg | None = None
    """Offset of target frame w.r.t. to the body frame. Defaults to None, in which case no offset is applied."""
    f1_name: str = MISSING
    """Name of the first frame for the RCM constraint."""
    f2_name: str = MISSING
    """Name of the second frame for the RCM constraint."""
    scale: float | tuple[float, ...] = 1.0
    """Scale factor for the action. Defaults to 1.0."""
    controller: DifferentialIKWithSoftRCMControllerCfg = MISSING
    """The configuration for the differential IK controller."""
    rcm_pos: tuple[float, float, float] | None = None
    """The position of the RCM point in the world frame. Defaults to None, in which case no RCM is applied."""
    rcm_beta: float = 0.5
    """The parameter for setting the RCM position. Defaults to 0.5."""


@configclass
class DifferentialInverseKinematicsWorkspaceClampedActionCfg(ActionTermCfg):
    """Differential IK action cfg with hard workspace clamp in world coordinates."""

    @configclass
    class OffsetCfg:
        pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
        rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

    class_type: type[ActionTerm] = DifferentialInverseKinematicsWorkspaceClampedAction

    joint_names: list[str] = MISSING
    body_name: str = MISSING
    body_offset: OffsetCfg | None = None
    scale: float | tuple[float, ...] = 1.0
    controller: DifferentialIKControllerCfg = MISSING

    enforce_workspace_bounds: bool = True
    x_bounds_world: tuple[float, float] | None = None
    y_bounds_world: tuple[float, float] | None = None
    z_bounds_world: tuple[float, float] | None = None


@configclass
class DifferentialInverseKinematicsWithSoftRCMWorkspaceClampedActionCfg(ActionTermCfg):
    """Soft-RCM differential IK action cfg with hard workspace clamp in world coordinates."""

    @configclass
    class OffsetCfg:
        pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
        rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

    class_type: type[ActionTerm] = DifferentialInverseKinematicsWithSoftRCMWorkspaceClampedAction

    joint_names: list[str] = MISSING
    body_name: str = MISSING
    body_offset: OffsetCfg | None = None
    f1_name: str = MISSING
    f2_name: str = MISSING
    scale: float | tuple[float, ...] = 1.0
    controller: DifferentialIKWithSoftRCMControllerCfg = MISSING
    rcm_pos: tuple[float, float, float] | None = None
    rcm_beta: float = 0.5

    enforce_workspace_bounds: bool = True
    x_bounds_world: tuple[float, float] | None = None
    y_bounds_world: tuple[float, float] | None = None
    z_bounds_world: tuple[float, float] | None = None
