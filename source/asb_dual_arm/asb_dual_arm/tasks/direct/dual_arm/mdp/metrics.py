from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _resolve_env_ids(env: ManagerBasedRLEnv, env_ids: Sequence[int] | torch.Tensor | slice | None) -> torch.Tensor:
    if env_ids is None:
        return torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    if isinstance(env_ids, slice):
        all_env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
        return all_env_ids[env_ids]
    if torch.is_tensor(env_ids):
        return env_ids.to(device=env.device, dtype=torch.long)
    return torch.as_tensor(list(env_ids), device=env.device, dtype=torch.long)


def _ensure_connectivity_diag_state(env: ManagerBasedRLEnv, success_frames: int) -> None:
    success_frames = int(max(1, success_frames))
    thresholds = (20, 50, 100)
    shape = (env.num_envs,)

    def _init_or_reinit(name: str, default: int = 0) -> torch.Tensor:
        tensor = getattr(env, name, None)
        needs_reinit = not torch.is_tensor(tensor)
        if torch.is_tensor(tensor):
            if tensor.shape != shape:
                needs_reinit = True
            elif hasattr(tensor, "is_inference") and tensor.is_inference():
                needs_reinit = True
        if needs_reinit:
            if default == -1:
                tensor = torch.full(shape, -1, device=env.device, dtype=torch.long)
            else:
                tensor = torch.zeros(shape, device=env.device, dtype=torch.long)
            setattr(env, name, tensor)
        return tensor

    _init_or_reinit("_vcd_episode_max_streak", default=0)
    _init_or_reinit("_vcd_episode_disconnect_count", default=0)
    _init_or_reinit("_vcd_episode_first_connected_step", default=-1)

    first_reach = getattr(env, "_vcd_episode_first_reach_step", None)
    if not isinstance(first_reach, dict):
        first_reach = {}
    for threshold in thresholds:
        key = int(threshold)
        tensor = first_reach.get(key, None)
        needs_reinit = not torch.is_tensor(tensor)
        if torch.is_tensor(tensor):
            if tensor.shape != shape:
                needs_reinit = True
            elif hasattr(tensor, "is_inference") and tensor.is_inference():
                needs_reinit = True
        if needs_reinit:
            tensor = torch.full(shape, -1, device=env.device, dtype=torch.long)
            first_reach[key] = tensor
    setattr(env, "_vcd_episode_first_reach_step", first_reach)

    recent_successes = getattr(env, "_vcd_recent_successes", None)
    if not isinstance(recent_successes, deque):
        setattr(env, "_vcd_recent_successes", deque())
    if not isinstance(getattr(env, "_vcd_recent_success_sum", None), int):
        setattr(env, "_vcd_recent_success_sum", 0)

    setattr(env, "_vcd_success_frames", success_frames)
    setattr(env, "_vcd_first_reach_thresholds", thresholds)
    setattr(env, "_vcd_success_window_size", 100)


def _reset_connectivity_diag_state(env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> None:
    if env_ids.numel() == 0:
        return
    if not hasattr(env, "_vcd_episode_max_streak"):
        return

    env._vcd_episode_max_streak[env_ids] = 0
    env._vcd_episode_disconnect_count[env_ids] = 0
    env._vcd_episode_first_connected_step[env_ids] = -1

    thresholds = getattr(env, "_vcd_first_reach_thresholds", ())
    first_reach = getattr(env, "_vcd_episode_first_reach_step", {})
    for threshold in thresholds:
        tensor = first_reach.get(int(threshold), None)
        if torch.is_tensor(tensor):
            tensor[env_ids] = -1


def update_vessel_connectivity_diagnostics(
    env: ManagerBasedRLEnv,
    strict_counter: torch.Tensor,
    connected_mask: torch.Tensor,
    prev_connected_mask: torch.Tensor,
    success_frames: int,
) -> None:
    """Update per-step diagnostics state for vessel connectivity."""
    _ensure_connectivity_diag_state(env, success_frames)

    active_mask = env.episode_length_buf > 0
    if not active_mask.any():
        return

    step_ids = active_mask.nonzero(as_tuple=False).squeeze(-1)
    step_count = env.episode_length_buf[step_ids]

    env._vcd_episode_max_streak[step_ids] = torch.maximum(
        env._vcd_episode_max_streak[step_ids], strict_counter[step_ids]
    )

    break_mask = prev_connected_mask[step_ids] & (~connected_mask[step_ids])
    env._vcd_episode_disconnect_count[step_ids] += break_mask.long()

    first_connected_mask = connected_mask[step_ids] & (env._vcd_episode_first_connected_step[step_ids] < 0)
    if first_connected_mask.any():
        env._vcd_episode_first_connected_step[step_ids[first_connected_mask]] = step_count[first_connected_mask]

    thresholds = getattr(env, "_vcd_first_reach_thresholds", ())
    first_reach = getattr(env, "_vcd_episode_first_reach_step", {})
    for threshold in thresholds:
        threshold = int(threshold)
        threshold_tensor = first_reach.get(threshold, None)
        if not torch.is_tensor(threshold_tensor):
            continue
        threshold_mask = (strict_counter[step_ids] >= threshold) & (threshold_tensor[step_ids] < 0)
        if threshold_mask.any():
            threshold_tensor[step_ids[threshold_mask]] = step_count[threshold_mask]


def collect_and_reset_vessel_connectivity_diagnostics(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int] | torch.Tensor | slice | None,
) -> None:
    """Collect episode summary into pending Info logs and reset diagnostics state."""
    env_ids = _resolve_env_ids(env, env_ids)
    if env_ids.numel() == 0:
        return

    success_frames = int(getattr(env, "_vcd_success_frames", 100))
    _ensure_connectivity_diag_state(env, success_frames)

    sim_dt = float(getattr(getattr(getattr(env, "cfg", None), "sim", None), "dt", 0.0))
    decimation = int(getattr(getattr(env, "cfg", None), "decimation", 1))
    stable_seconds = float(success_frames) * sim_dt * float(max(1, decimation))

    if getattr(env, "common_step_counter", 0) > 0:
        sentinel = float(getattr(env, "max_episode_length", 0) + 1)

        success = (env._vcd_episode_max_streak[env_ids] >= success_frames).float()
        max_streak = env._vcd_episode_max_streak[env_ids].float()
        disconnect_count = env._vcd_episode_disconnect_count[env_ids].float()

        first_connected = env._vcd_episode_first_connected_step[env_ids].float()
        first_connected = torch.where(
            first_connected >= 0,
            first_connected,
            torch.full_like(first_connected, sentinel),
        )

        pending_log = {
            "Episode_Connectivity/SuccessRate100": torch.tensor(0.0, device=env.device, dtype=torch.float32),
            "Episode_Connectivity/MaxConsecutiveConnectedFrames": max_streak.mean(),
            "Episode_Connectivity/FirstReachStepConnected": first_connected.mean(),
            "Episode_Connectivity/DisconnectCount": disconnect_count.mean(),
            "Episode_Connectivity/StableFramesTarget": torch.tensor(
                float(success_frames), device=env.device, dtype=torch.float32
            ),
            "Episode_Connectivity/StableTargetSeconds": torch.tensor(
                stable_seconds, device=env.device, dtype=torch.float32
            ),
        }

        recent_successes: deque[int] = getattr(env, "_vcd_recent_successes")
        recent_success_sum: int = int(getattr(env, "_vcd_recent_success_sum"))
        window_size = int(getattr(env, "_vcd_success_window_size", 100))
        for value in success.to(dtype=torch.int).tolist():
            if len(recent_successes) == window_size:
                recent_success_sum -= int(recent_successes.popleft())
            recent_successes.append(int(value))
            recent_success_sum += int(value)
        setattr(env, "_vcd_recent_success_sum", int(recent_success_sum))

        window_count = max(1, len(recent_successes))
        pending_log["Episode_Connectivity/SuccessRate100"] = torch.tensor(
            float(recent_success_sum) / float(window_count),
            device=env.device,
            dtype=torch.float32,
        )

        existing_pending = getattr(env, "_pending_episode_log", None)
        if not isinstance(existing_pending, dict):
            existing_pending = {}
        existing_pending.update(pending_log)
        setattr(env, "_pending_episode_log", existing_pending)

    _reset_connectivity_diag_state(env, env_ids)