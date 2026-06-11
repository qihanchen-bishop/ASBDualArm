# RSSM World Model Training

This folder provides online unsupervised training code for a recurrent state-space world model (RSSM).

## Goal

- Input: RGB image + robot arm action (first N action dimensions).
- Target: next RGB frame prediction.
- Data source: Isaac Sim environment rollouts collected online.

## Files

- `rssm.py`: RSSM model, encoder/decoder, KL utility.
- `train_rssm_world_model.py`: environment rollout + replay buffer + training loop.

## Run

```bash
${IsaacLab_PATH}/isaaclab.sh -p /workspace/isaaclab/source/ASBDualArm/scripts/worldmodel/train_rssm_world_model.py \
  --task Isaac-VesselSemFixed-SingleRobot-IK-Play-v0 \
  --num_envs 1 \
  --headless \
  --collect_steps 40000 \
  --warmup_steps 2000 \
  --seq_len 12 \
  --batch_size 16 \
  --action_dim 6
```

## Notes

- If your selected task has different action layout, tune `--action_dim`.
- Checkpoints, recon previews, and TensorBoard event files are saved under `logs/worldmodel/runs/<exp_name>_<timestamp>/`.
