# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause

"""Train RSSM world model from Isaac Sim online interaction.

Input: RGB image + arm end-effector action.
Target: next RGB frame prediction.
"""

from __future__ import annotations

import argparse
import os
import random
from collections import deque
from dataclasses import dataclass
from datetime import datetime

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="RSSM world model training with Isaac Sim online rollouts.")
parser.add_argument("--task", type=str, default="Isaac-VesselSemFixed-SingleRobot-IK-Play-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--seed", type=int, default=42)

# data collection
parser.add_argument("--collect_steps", type=int, default=40000)
parser.add_argument("--warmup_steps", type=int, default=2000)
parser.add_argument("--train_every", type=int, default=20)
parser.add_argument("--updates_per_train", type=int, default=2)
parser.add_argument("--action_scale", type=float, default=0.5)
parser.add_argument("--action_dim", type=int, default=-1, help="Use first N dims from env action.")

# model and optimization
parser.add_argument("--seq_len", type=int, default=12)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--buffer_capacity", type=int, default=200000)
parser.add_argument("--image_size", type=int, default=64)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--kl_scale", type=float, default=1.0)
parser.add_argument("--free_nats", type=float, default=1.0)
parser.add_argument("--grad_clip_norm", type=float, default=100.0)

# logging and checkpoints
parser.add_argument("--exp_name", type=str, default="rssm_vessel")
parser.add_argument("--log_every", type=int, default=100)
parser.add_argument("--save_every", type=int, default=2000)
parser.add_argument("--tb_log_subdir", type=str, default="tensorboard")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(headless=args_cli.headless, enable_cameras=True)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

import isaaclab_tasks  # noqa: F401
import msr.tasks  # noqa: F401

from rssm import RSSM, kl_normal


@dataclass
class Transition:
    obs: torch.Tensor
    action: torch.Tensor
    next_obs: torch.Tensor
    done: torch.Tensor


class SequenceReplayBuffer:
    """Simple sequence replay buffer for world model training."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.data: deque[Transition] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.data)

    def add(self, obs: torch.Tensor, action: torch.Tensor, next_obs: torch.Tensor, done: torch.Tensor) -> None:
        self.data.append(Transition(obs=obs, action=action, next_obs=next_obs, done=done))

    def sample_sequences(self, batch_size: int, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        max_start = len(self.data) - seq_len
        if max_start <= 1:
            raise RuntimeError("Replay buffer does not contain enough transitions for sequence sampling.")

        obs_batch = []
        act_batch = []

        sampled = 0
        trials = 0
        while sampled < batch_size and trials < batch_size * 50:
            trials += 1
            start_idx = random.randint(0, max_start - 1)
            window = list(self.data)[start_idx : start_idx + seq_len]

            # Avoid crossing terminal states inside one sequence.
            done_stack = torch.stack([tr.done for tr in window], dim=0)
            if done_stack[:-1].any():
                continue

            obs_seq = torch.stack([tr.next_obs for tr in window], dim=0)
            act_seq = torch.stack([tr.action for tr in window], dim=0)
            obs_batch.append(obs_seq)
            act_batch.append(act_seq)
            sampled += 1

        if sampled < batch_size:
            raise RuntimeError("Failed to sample enough non-terminal sequences. Increase data collection.")

        obs_tensor = torch.stack(obs_batch, dim=0).to(device)
        act_tensor = torch.stack(act_batch, dim=0).to(device)
        return obs_tensor, act_tensor


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_chw_float01(images: torch.Tensor, image_size: int) -> torch.Tensor:
    """Convert NHWC uint8/float image to NCHW float32 in [0, 1]."""
    x = images.to(torch.float32)
    if x.max() > 1.0:
        x = x / 255.0
    if x.shape[-1] == 4:
        x = x[..., :3]
    x = x.permute(0, 3, 1, 2).contiguous()
    if x.shape[-1] != image_size or x.shape[-2] != image_size:
        x = F.interpolate(x, size=(image_size, image_size), mode="bilinear", align_corners=False)
    return torch.clamp(x, 0.0, 1.0)


def extract_rgb_obs(obs: dict, env, image_size: int) -> torch.Tensor:
    """Read RGB observation from policy dict or scene camera fallback."""
    policy_obs = obs.get("policy", None)

    if isinstance(policy_obs, dict) and "rgb_image" in policy_obs:
        return to_chw_float01(policy_obs["rgb_image"], image_size=image_size)

    if isinstance(policy_obs, torch.Tensor) and policy_obs.ndim == 4 and policy_obs.shape[-1] in (3, 4):
        return to_chw_float01(policy_obs, image_size=image_size)

    camera = env.unwrapped.scene["camera"]
    rgb = camera.data.output["rgb"]
    return to_chw_float01(rgb, image_size=image_size)


def infer_action_batch_shape(action_space_shape: tuple[int, ...], num_envs: int) -> tuple[int, int]:
    """Infer batched action tensor shape as (num_envs, action_dim_total)."""
    if len(action_space_shape) == 1:
        return num_envs, int(action_space_shape[0])
    return int(action_space_shape[0]), int(np.prod(action_space_shape[1:]))


def make_random_arm_action(batch_size: int, total_action_dim: int, action_dim: int, scale: float, device: torch.device) -> torch.Tensor:
    action = torch.zeros((batch_size, total_action_dim), device=device)
    arm_part = torch.empty((batch_size, action_dim), device=device).uniform_(-scale, scale)
    action[:, :action_dim] = arm_part
    return action


def save_reconstruction_grid(
    model: RSSM,
    obs_seq: torch.Tensor,
    act_seq: torch.Tensor,
    save_path: str,
) -> None:
    with torch.no_grad():
        out = model.rollout_observe(actions=act_seq[:1], images=obs_seq[:1])
        pred = out["recon"][0]
        gt = obs_seq[0]

        # Concatenate gt and pred side-by-side for each time step.
        rows = torch.cat([gt, torch.sigmoid(pred)], dim=-1)
        grid = torch.cat([rows[t] for t in range(rows.shape[0])], dim=-2)
        grid = (grid.clamp(0.0, 1.0) * 255.0).to(torch.uint8).permute(1, 2, 0).cpu().numpy()

    try:
        from PIL import Image

        Image.fromarray(grid).save(save_path)
    except Exception:
        # Pillow is optional; skip preview if unavailable.
        pass


def main() -> None:
    set_seed(args_cli.seed)

    device = torch.device(args_cli.device)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    log_root_path = os.path.join(project_root, "logs", "worldmodel")
    log_root_path = os.path.abspath(log_root_path)

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )

    env = gym.make(args_cli.task, cfg=env_cfg)
    obs, _ = env.reset(seed=args_cli.seed)

    action_batch_size, action_total_dim = infer_action_batch_shape(
        action_space_shape=env.unwrapped.action_space.shape,
        num_envs=args_cli.num_envs,
    )
    action_dim = action_total_dim if args_cli.action_dim <= 0 else min(args_cli.action_dim, action_total_dim)

    model = RSSM(action_dim=action_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args_cli.lr)

    buffer = SequenceReplayBuffer(capacity=args_cli.buffer_capacity)

    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{args_cli.exp_name}_{run_tag}"
    out_dir = os.path.join(log_root_path, "runs", run_name)
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    vis_dir = os.path.join(out_dir, "recon")
    tb_dir = os.path.join(out_dir, args_cli.tb_log_subdir)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(tb_dir, exist_ok=True)

    tb_writer = SummaryWriter(log_dir=tb_dir)
    tb_writer.add_scalar("train/boot", 1.0, 0)
    tb_writer.add_text("meta/task", str(args_cli.task), 0)
    tb_writer.flush()

    print("=" * 80)
    print("RSSM World Model Training")
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    print(f"Exact experiment name requested from command line: {run_name}")
    print(f"Task: {args_cli.task}")
    print(f"Num envs: {args_cli.num_envs}")
    print(f"Action tensor shape: ({action_batch_size}, {action_total_dim})")
    print(f"Using first action dims as arm action: {action_dim}/{action_total_dim}")
    print(f"Output dir: {out_dir}")
    print(f"TensorBoard dir: {tb_dir}")
    print(f"Event files: {os.path.join(tb_dir, 'events.out.tfevents*')}")
    print("=" * 80)

    episode_steps = torch.zeros(args_cli.num_envs, dtype=torch.int64)

    for step in range(1, args_cli.collect_steps + 1):
        curr_rgb = extract_rgb_obs(obs, env, image_size=args_cli.image_size)
        action = make_random_arm_action(
            batch_size=action_batch_size,
            total_action_dim=action_total_dim,
            action_dim=action_dim,
            scale=args_cli.action_scale,
            device=device,
        )

        next_obs, _, terminated, truncated, _ = env.step(action)
        next_rgb = extract_rgb_obs(next_obs, env, image_size=args_cli.image_size)

        terminated_t = torch.as_tensor(terminated, device=device).to(torch.bool)
        truncated_t = torch.as_tensor(truncated, device=device).to(torch.bool)
        done = torch.logical_or(terminated_t, truncated_t)

        for env_id in range(args_cli.num_envs):
            buffer.add(
                obs=curr_rgb[env_id].detach().cpu(),
                action=action[env_id, :action_dim].detach().cpu(),
                next_obs=next_rgb[env_id].detach().cpu(),
                done=done[env_id].detach().cpu(),
            )

        obs = next_obs
        episode_steps += 1

        # Train periodically after warmup.
        if step >= args_cli.warmup_steps and step % args_cli.train_every == 0:
            metrics = {}
            for _ in range(args_cli.updates_per_train):
                obs_seq, act_seq = buffer.sample_sequences(
                    batch_size=args_cli.batch_size,
                    seq_len=args_cli.seq_len,
                    device=device,
                )

                outputs = model.rollout_observe(actions=act_seq, images=obs_seq)
                recon_logits = outputs["recon"]

                recon_loss = F.mse_loss(torch.sigmoid(recon_logits), obs_seq)
                kl = kl_normal(
                    post_mean=outputs["post_mean"],
                    post_std=outputs["post_std"],
                    prior_mean=outputs["prior_mean"],
                    prior_std=outputs["prior_std"],
                )
                kl = torch.clamp(kl, min=args_cli.free_nats).mean()

                loss = recon_loss + args_cli.kl_scale * kl

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args_cli.grad_clip_norm)
                optimizer.step()

                metrics = {
                    "loss": float(loss.item()),
                    "recon": float(recon_loss.item()),
                    "kl": float(kl.item()),
                    "grad_norm": float(grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm),
                }

            if metrics:
                tb_writer.add_scalar("train/loss", metrics["loss"], step)
                tb_writer.add_scalar("train/recon", metrics["recon"], step)
                tb_writer.add_scalar("train/kl", metrics["kl"], step)
                tb_writer.add_scalar("train/grad_norm", metrics["grad_norm"], step)
                tb_writer.add_scalar("train/buffer_size", float(len(buffer)), step)
                tb_writer.add_scalar("train/lr", float(optimizer.param_groups[0]["lr"]), step)

            if step % args_cli.log_every == 0:
                print(
                    f"[step {step:06d}] "
                    f"loss={metrics['loss']:.4f} "
                    f"recon={metrics['recon']:.4f} "
                    f"kl={metrics['kl']:.4f} "
                    f"buffer={len(buffer)}"
                )

            if step % args_cli.save_every == 0:
                ckpt_path = os.path.join(ckpt_dir, f"rssm_step_{step:06d}.pt")
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "step": step,
                        "args": vars(args_cli),
                    },
                    ckpt_path,
                )

                # Save a quick qualitative preview.
                try:
                    sample_obs, sample_act = buffer.sample_sequences(batch_size=1, seq_len=args_cli.seq_len, device=device)
                    preview_path = os.path.join(vis_dir, f"recon_step_{step:06d}.png")
                    save_reconstruction_grid(model, sample_obs, sample_act, preview_path)
                except RuntimeError:
                    pass

        if done.any():
            reset_ids = torch.nonzero(done, as_tuple=False).squeeze(-1)
            if reset_ids.numel() > 0:
                episode_steps[reset_ids] = 0

    final_ckpt = os.path.join(ckpt_dir, "rssm_final.pt")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": args_cli.collect_steps,
            "args": vars(args_cli),
        },
        final_ckpt,
    )

    print("Training complete.")
    print(f"Final checkpoint: {final_ckpt}")

    tb_writer.flush()
    tb_writer.close()

    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
    finally:
        simulation_app.close()
