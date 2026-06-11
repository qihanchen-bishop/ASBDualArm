# 关键修改的位置
1 在joint里面修改asset

# 常见指令
运行键盘控制数据采集（用的哪个环境要验证一下）./isaaclab.sh -p /workspace/isaaclab/source/ASBDualArm/scripts/env/state_machine/dual_arm.py --num_envs 1 --enable_cameras --save-data --usd_path /workspace/isaaclab/source/ASBDualArm/source/msr/msr/assets/others/msr_organ.usd

./isaaclab.sh -p source/ASBDualArm/scripts/skrl/test.py --task Isaac-DualArm-VesselSem-IK-ConnectivityOnly-Play-v0 --checkpoint /workspace/isaaclab/logs/skrl/vessel_semantic/2026-04-22_08-35-00_ppo_torch/checkpoints/best_agent.pt --enable_cameras --stable_frames 50 --max_action_steps 1000 --debug
-
-save_success_video：开启成功回合视频保存
--save_all_success_videos：保存所有成功回合（不加这个时，只保存第一个成功回合）
--success_video_fps：视频帧率，默认 30

./isaaclab.sh -p /workspace/isaaclab/source/ASBDualArm/scripts/skrl/train.py --task Isaac-DualArm-VesselSem-IK-ConnectivityOnly-v0 --num_envs 32 --seed 42 --headless --enable_cameras --max_iterations 10000

./isaaclab.sh -p /workspace/isaaclab/source/ASBDualArm/scripts/skrl/env_test.py --task Isaac-DualArm-VesselSem-IK-ConnectivityOnly-Play-v0 --random/zero --enable_cameras --disable_ee_workspace_constraint

./isaaclab.sh -p /workspace/isaaclab/source/ASBDualArm/scripts/skrl/env_test.py --task Isaac-DualArm-VesselSem-DualArm-IK-ConnectivityOnly-Play-v0  --zero --enable_cameras --disable_ee_workspace_constraint

./isaaclab.sh -p source/ASBDualArm/scripts/skrl/env_test.py --task Isaac-DualArm-VesselSem-IK-ConnectivityOnly-v0 --zero --enable_cameras --disable_ee_workspace_constraint --disable_physics_replication --num_envs 1 --max_steps 1

 ./isaaclab.sh -p /workspace/isaaclab/source/mytask/scripts/env/state_machine/dual_arm.py --num_envs 1 --enable_cameras --usd_path /workspace/isaaclab/source/ASBDualArm/source/msr/msr/assets/others/msr_organ.usd

./isaaclab.sh -p source/ASBDualArm/scripts/env/state_machine/dual-so-arm-tel.py --num_envs 1 --enable_cameras

cd /workspace/isaaclab
./isaaclab.sh -p source/ASBDualArm/scripts/env/state_machine/so-arm-policy-test.py --num_envs 1 --enable_cameras --policy-path /workspace/isaaclab/source/ASBDualArm/policy/act_sim_cube1/checkpoints/100000/pretrained_model --test-times 10 --reset-settle-seconds 0.75 --target-random-x 0.10  --target-random-y 0.02

./isaaclab.sh -p source/ASBDualArm/scripts/env/state_machine/so-arm-policy-test.py --policy-type diffusion  --policy-path /workspace/isaaclab/source/ASBDualArm/policy/diffusion_policy/100000/pretrained_model --test-times 100 --reset-settle-seconds 0.75   --target-random-x 0.10  --target-random-y 0.02 --num_envs 1 --enable_cameras  --headless


./isaaclab.sh -p source/ASBDualArm/scripts/env/state_machine/so-arm-policy-test.py \
  --policy-class my_pkg.my_policy:MyPolicy \
  --policy-path /path/to/custom/pretrained_model

./isaaclab.sh -p source/ASBDualArm/scripts/env/state_machine/so-arm-policy-test.py \
  --num_envs 1 --enable_cameras --interactive

./isaaclab.sh -p source/ASBDualArm/scripts/env/state_machine/dual-so-arm-tel.py --record-dataset-path /workspace/isaaclab/source/ASBDualArm/saved_data/cube1 --num_envs 1 --enable_cameras

cd /workspace/isaaclab

./isaaclab.sh -p source/lerobot/mycode/gui_view_lerobot_dataset.py \
  --web \
  --host 0.0.0.0 \
  --port 8765 \
  --root /workspace/isaaclab/source/ASBDualArm/saved_data/cube1

cd /workspace/isaaclab

./isaaclab.sh -p source/ASBDualArm/scripts/env/state_machine/so-arm-policy-test.py \
  --headless \
  --enable_cameras \
  --num_envs 1 \
  --policy-path /workspace/isaaclab/source/ASBDualArm/policy/1A \
  --policy-type mask_act \
  --test-times 5 \
  --max-frames-per-test 600

./isaaclab.sh -p source/ASBDualArm/scripts/env/state_machine/so-arm-policy-test.py \
  --headless \
  --enable_cameras \
  --num_envs 1 \
  --policy-path /workspace/isaaclab/source/ASBDualArm/policy/1B \
  --policy-type mask_act \
  --none_randon \
  --grid-repeats-per-point 10 \
  --max-frames-per-test 600

cd /workspace/isaaclab

./isaaclab.sh -p source/ASBDualArm/scripts/env/state_machine/so-arm-policy-test.py \
  --policy-type act \
  --policy-path /workspace/isaaclab/source/ASBDualArm/policy/act_sim_cube1/checkpoints/100000/pretrained_model \
  --none_randon \
  --grid-repeats-per-point 10 \
  --num_envs 1 \
  --enable_cameras \
  --headless\
  --max-frames-per-test 600

./isaaclab.sh -p source/ASBDualArm/scripts/env/state_machine/so-arm-policy-test.py \
  --policy-type act \
  --policy-path /workspace/isaaclab/source/ASBDualArm/policy/act_sim_cube1/checkpoints/100000/pretrained_model \
  --test-times 100 \
  --num_envs 1 \
  --enable_cameras \
  --headless\
  --max-frames-per-test 600

  cd /workspace/isaaclab

./isaaclab.sh -p source/ASBDualArm/scripts/env/state_machine/so-arm-policy-test.py \
  --policy-type act \
  --policy-path /workspace/isaaclab/source/ASBDualArm/policy/act_sim_cube1/checkpoints/100000/pretrained_model \
  --grid_random \
  --grid-size 10 \
  --grid-repeats-per-point 10 \
  --num_envs 1 \
  --enable_cameras \
  --headless

  GRID_SIZE=10 REPEATS=10 source/ASBDualArm/policy/run_grid_random_tests.sh