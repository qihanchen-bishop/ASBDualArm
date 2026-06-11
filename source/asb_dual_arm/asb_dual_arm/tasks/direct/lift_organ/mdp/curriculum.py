from __future__ import annotations
from collections.abc import Sequence
from typing import TYPE_CHECKING, Optional

from isaaclab.managers import CurriculumTermCfg, ManagerTermBase

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class anneal_reward_weight(ManagerTermBase):
    """
    在线性区间 [start_step, end_step] 内将奖励项的 weight 从 start -> end。
    支持任意实数（可升可降，可为负数），区间外保持端点值。
    用法示例：
        CurrTerm(func=mdp.anneal_reward_weight,
                 params={"term_name": "action_rate",
                         "start": -1e-5, "end": -1e-4,
                         "start_step": 0, "end_step": 60000})
    """

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        term_name: str,
        start: float,
        end: float,
        start_step: int,
        end_step: int,
        clip_min: Optional[float] = None,
        clip_max: Optional[float] = None,
    ) -> float:
        t = env.common_step_counter
        if end_step <= start_step:
            value = end
        elif t <= start_step:
            value = start
        elif t >= end_step:
            value = end
        else:
            alpha = (t - start_step) / float(end_step - start_step)
            value = start + (end - start) * alpha

        # 可选裁剪
        if clip_min is not None:
            value = max(value, clip_min)
        if clip_max is not None:
            value = min(value, clip_max)

        term_cfg = env.reward_manager.get_term_cfg(term_name)
        term_cfg.weight = float(value)
        env.reward_manager.set_term_cfg(term_name, term_cfg)
        return term_cfg.weight


class anneal_reward_param(ManagerTermBase):
    """
    在线性区间 [start_step, end_step] 内将某个奖励参数（如 "std"）从 start -> end。
    支持任意实数（可升可降，可为负数），区间外保持端点值。
    用法示例：
        CurrTerm(func=mdp.anneal_reward_param,
                 params={"term_name": "position_tracking",
                         "param_name": "std",
                         "start": 0.08, "end": 0.03,
                         "start_step": 0, "end_step": 40000,
                         "clip_min": 1e-4})  # 防止 std 变成非正
    """

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        term_name: str,
        param_name: str,
        start: float,
        end: float,
        start_step: int,
        end_step: int,
        clip_min: Optional[float] = None,
        clip_max: Optional[float] = None,
    ):
        t = env.common_step_counter
        if end_step <= start_step:
            value = end
        elif t <= start_step:
            value = start
        elif t >= end_step:
            value = end
        else:
            alpha = (t - start_step) / float(end_step - start_step)
            value = start + (end - start) * alpha

        # 可选裁剪（例如 std 需要 >0）
        if clip_min is not None:
            value = max(value, clip_min)
        if clip_max is not None:
            value = min(value, clip_max)

        term_cfg = env.reward_manager.get_term_cfg(term_name)
        term_cfg.params[param_name] = float(value)
        env.reward_manager.set_term_cfg(term_name, term_cfg)
        return term_cfg.params[param_name]
