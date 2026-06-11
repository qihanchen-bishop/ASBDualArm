from __future__ import annotations
from collections import deque
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


class schedule_connectivity_stable_frames(ManagerTermBase):
    """Piecewise schedule for connectivity ``stable_frames``.

    This updates the shared ``stable_frames`` parameter for all connectivity
    reward components and the success termination term so that completion
    criteria and shaping remain synchronized across curriculum stages.
    """

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        phase_frames: Sequence[int],
        phase_steps: Sequence[int] | None = None,
        promotion_mode: str = "global_step",
        promotion_success_rate_threshold: float = 0.6,
        promotion_min_successes: int = 5,
        promotion_window_episodes: int = 20,
        promotion_min_episodes: int = 5,
        promotion_min_consecutive_successes: int = 0,
        promotion_required_stable_rounds: int = 1,
        demotion_enabled: bool = False,
        demotion_success_rate_threshold: float = 0.2,
        demotion_min_episodes: int = 20,
        demotion_required_stable_rounds: int = 10,
        adjustment_cooldown_rounds: int = 0,
        reward_term_names: Sequence[str] = (
            "progress_connectivity",
            "hold_connectivity",
            "break_connectivity",
            "done_bonus_connectivity",
        ),
        termination_term_name: str = "success",
    ) -> dict[str, float]:
        _ = env_ids

        if len(phase_frames) == 0:
            raise ValueError("phase_frames must be non-empty.")

        schedule_frames = [max(1, int(frame)) for frame in phase_frames]
        global_step = int(env.common_step_counter)
        stage_index = int(getattr(env, "_connectivity_curriculum_stage_idx", 0))
        stage_index = max(0, min(stage_index, len(schedule_frames) - 1))

        promotion_success_rate = 0.0
        promotion_success_count = 0
        promotion_window_count = 0
        promotion_consecutive_successes = 0
        promotion_stable_rounds = int(getattr(env, "_connectivity_curriculum_promotion_stable_rounds", 0))
        demotion_stable_rounds = int(getattr(env, "_connectivity_curriculum_demotion_stable_rounds", 0))
        round_index = int(getattr(env, "_connectivity_curriculum_round_index", 0))
        last_adjust_round = int(getattr(env, "_connectivity_curriculum_last_adjust_round", -10**9))
        promoted = 0
        demoted = 0

        required_promotion_rounds = max(1, int(promotion_required_stable_rounds))
        required_demotion_rounds = max(1, int(demotion_required_stable_rounds))
        cooldown_rounds = max(0, int(adjustment_cooldown_rounds))

        mode = str(promotion_mode).lower().strip()
        if mode == "global_step":
            if phase_steps is None:
                raise ValueError("phase_steps must be provided when promotion_mode='global_step'.")
            if len(phase_steps) != len(schedule_frames):
                raise ValueError("phase_steps and phase_frames must have equal length.")

            schedule = sorted([(int(s), int(f)) for s, f in zip(phase_steps, schedule_frames)], key=lambda x: x[0])
            resolved_stage = 0
            for idx, (start_step, _) in enumerate(schedule):
                if global_step >= start_step:
                    resolved_stage = idx
                else:
                    break
            stage_index = resolved_stage
            target_frames = int(schedule[stage_index][1])
        elif mode in ("success_rate", "metric"):
            history = getattr(env, "_vcd_recent_successes", None)
            if isinstance(history, deque):
                if promotion_window_episodes > 0 and len(history) > int(promotion_window_episodes):
                    values = [int(v) for v in list(history)[-int(promotion_window_episodes) :]]
                else:
                    values = [int(v) for v in list(history)]
            else:
                values = []

            promotion_window_count = len(values)
            if promotion_window_count > 0:
                promotion_success_count = int(sum(values))
                promotion_success_rate = float(promotion_success_count) / float(promotion_window_count)
                for value in reversed(values):
                    if value > 0:
                        promotion_consecutive_successes += 1
                    else:
                        break

            min_eps = max(int(promotion_min_episodes), int(promotion_min_successes), 1)
            promotion_window_ok = (
                promotion_window_count >= min_eps
                and promotion_success_count >= int(promotion_min_successes)
                and promotion_success_rate > float(promotion_success_rate_threshold)
                and promotion_consecutive_successes >= int(promotion_min_consecutive_successes)
            )

            demotion_window_ok = (
                bool(demotion_enabled)
                and stage_index > 0
                and promotion_window_count >= max(1, int(demotion_min_episodes))
                and promotion_success_rate < float(demotion_success_rate_threshold)
            )

            if promotion_window_ok:
                promotion_stable_rounds += 1
            else:
                promotion_stable_rounds = 0

            if demotion_window_ok:
                demotion_stable_rounds += 1
            else:
                demotion_stable_rounds = 0

            round_index += 1

            cooldown_satisfied = (round_index - last_adjust_round) >= cooldown_rounds

            can_demote = (
                bool(demotion_enabled)
                and stage_index > 0
                and demotion_stable_rounds > required_demotion_rounds
                and cooldown_satisfied
            )
            can_promote = (
                stage_index < (len(schedule_frames) - 1)
                and promotion_stable_rounds > required_promotion_rounds
                and cooldown_satisfied
            )

            if can_demote:
                stage_index -= 1
                demoted = 1
                promotion_stable_rounds = 0
                demotion_stable_rounds = 0
                last_adjust_round = round_index
            elif can_promote:
                stage_index += 1
                promoted = 1
                promotion_stable_rounds = 0
                demotion_stable_rounds = 0
                last_adjust_round = round_index

            target_frames = int(schedule_frames[stage_index])
        else:
            raise ValueError(
                f"Unsupported promotion_mode '{promotion_mode}'. Expected 'global_step' or 'success_rate'."
            )

        setattr(env, "_connectivity_curriculum_promotion_stable_rounds", int(promotion_stable_rounds))
        setattr(env, "_connectivity_curriculum_demotion_stable_rounds", int(demotion_stable_rounds))
        setattr(env, "_connectivity_curriculum_round_index", int(round_index))
        setattr(env, "_connectivity_curriculum_last_adjust_round", int(last_adjust_round))

        setattr(env, "_connectivity_curriculum_stage_idx", int(stage_index))

        # Keep all connectivity reward components on the same stable-frames target.
        for term_name in reward_term_names:
            try:
                term_cfg = env.reward_manager.get_term_cfg(term_name)
            except ValueError:
                continue
            if term_cfg is None or term_cfg.params is None:
                continue
            current_value = int(term_cfg.params.get("stable_frames", target_frames))
            if current_value != target_frames:
                term_cfg.params["stable_frames"] = target_frames
                env.reward_manager.set_term_cfg(term_name, term_cfg)

        # Keep success termination threshold aligned with reward shaping.
        if termination_term_name:
            try:
                term_cfg = env.termination_manager.get_term_cfg(termination_term_name)
            except ValueError:
                term_cfg = None
            if term_cfg is not None and term_cfg.params is not None:
                current_value = int(term_cfg.params.get("stable_frames", target_frames))
                if current_value != target_frames:
                    term_cfg.params["stable_frames"] = target_frames
                    env.termination_manager.set_term_cfg(termination_term_name, term_cfg)

        setattr(env, "_connectivity_curriculum_stable_frames", int(target_frames))

        sim_dt = float(getattr(getattr(getattr(env, "cfg", None), "sim", None), "dt", 0.0))
        decimation = int(getattr(getattr(env, "cfg", None), "decimation", 1))
        stable_seconds = float(target_frames) * sim_dt * float(max(1, decimation))

        return {
            "stage_index": float(stage_index),
            "stable_frames": float(target_frames),
            "stable_seconds": stable_seconds,
            "global_step": float(global_step),
            "promotion_mode": 1.0 if mode in ("success_rate", "metric") else 0.0,
            "promotion_success_rate": float(promotion_success_rate),
            "promotion_success_count": float(promotion_success_count),
            "promotion_window_episodes": float(promotion_window_count),
            "promotion_consecutive_successes": float(promotion_consecutive_successes),
            "promotion_required_stable_rounds": float(required_promotion_rounds),
            "promotion_stable_rounds": float(promotion_stable_rounds),
            "promotion_triggered": float(promoted),
            "demotion_enabled": 1.0 if demotion_enabled else 0.0,
            "demotion_success_rate_threshold": float(demotion_success_rate_threshold),
            "demotion_required_stable_rounds": float(required_demotion_rounds),
            "demotion_stable_rounds": float(demotion_stable_rounds),
            "demotion_triggered": float(demoted),
            "curriculum_round_index": float(round_index),
        }
