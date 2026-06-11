from __future__ import annotations

import torch
from collections.abc import Sequence

from isaaclab.envs import ManagerBasedRLEnv


class VesselSemEnv(ManagerBasedRLEnv):
    """Vessel env with a 1-second post-reset action hold.

    During the first ``WARMUP_STEPS`` RL steps of each episode, actions are
    replaced with zeros. This applies both right after startup and after every
    reset.
    """

    # step_dt = sim.dt * decimation = 0.005 * 2 = 0.01 s -> 100 steps ~= 1 s
    WARMUP_STEPS: int = 100
    STAGGER_TIMEOUT_PHASE: bool = True

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)

        # Use a dedicated per-env warmup counter so action hold is independent
        # from episode_length_buf (which we may randomize for timeout staggering).
        self._warmup_step_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        # Initial timeout phase staggering: randomize episode counters once at startup.
        # This makes different envs time out at different steps, reducing reset bursts.
        if self.STAGGER_TIMEOUT_PHASE and self.max_episode_length > 1:
            self.episode_length_buf.copy_(
                torch.randint(
                    low=0,
                    high=self.max_episode_length,
                    size=(self.num_envs,),
                    device=self.device,
                    dtype=torch.long,
                )
            )

    def _reset_idx(self, env_ids: Sequence[int]):
        super()._reset_idx(env_ids)
        self._warmup_step_buf[env_ids] = 0
        for attr_name, reset_value in (
            ("_vessel_area_prev", 0.0),
            ("_vessel_area_mean", 0.0),
            ("_vessel_area_var", 1.0),
            ("_vessel_hard_conn_streak", 0),
        ):
            if hasattr(self, attr_name):
                getattr(self, attr_name)[env_ids] = reset_value

    def step(self, action: torch.Tensor):  # type: ignore[override]
        warming_up = self._warmup_step_buf < self.WARMUP_STEPS
        if warming_up.any():
            action = torch.where(warming_up.unsqueeze(-1), torch.zeros_like(action), action)
        out = super().step(action)
        self._warmup_step_buf += 1
        return out
