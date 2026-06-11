# 关键修改的位置
1 在joint里面修改asset

# 常见指令
运行键盘控制数据采集（用的哪个环境要验证一下）./isaaclab.sh -p /workspace/isaaclab/source/ASBDualArm/scripts/env/state_machine/lift_organ_fixed.py --num_envs 1 --enable_cameras --save-data --usd_path /workspace/isaaclab/source/ASBDualArm/source/msr/msr/assets/others/msr_organ.usd

./isaaclab.sh -p source/ASBDualArm/scripts/skrl/test.py --task Isaac-VesselSemFixed-SingleRobot-IK-ConnectivityOnly-Play-v0 --checkpoint /workspace/isaaclab/logs/skrl/vessel_semantic/2026-04-22_08-35-00_ppo_torch/checkpoints/best_agent.pt --enable_cameras --stable_frames 50 --max_action_steps 1000 --debug
-
-save_success_video：开启成功回合视频保存
--save_all_success_videos：保存所有成功回合（不加这个时，只保存第一个成功回合）
--success_video_fps：视频帧率，默认 30

./isaaclab.sh -p /workspace/isaaclab/source/ASBDualArm/scripts/skrl/train.py --task Isaac-VesselSemFixed-SingleRobot-IK-ConnectivityOnly-v0 --num_envs 32 --seed 42 --headless --enable_cameras --max_iterations 10000

./isaaclab.sh -p /workspace/isaaclab/source/ASBDualArm/scripts/skrl/env_test.py --task Isaac-VesselSemFixed-SingleRobot-IK-ConnectivityOnly-Play-v0 --random/zero --enable_cameras --disable_ee_workspace_constraint

./isaaclab.sh -p /workspace/isaaclab/source/ASBDualArm/scripts/skrl/env_test.py --task Isaac-LiftOrganFixed-Upe6-SingleRobot-JointPos-Play-v0  --zero --enable_cameras --disable_ee_workspace_constraint

./isaaclab.sh -p source/ASBDualArm/scripts/skrl/env_test.py --task Isaac-VesselSemFixed-SingleRobot-IK-ConnectivityOnly-v0 --zero --enable_cameras --disable_ee_workspace_constraint --disable_physics_replication --num_envs 1 --max_steps 1

 ./isaaclab.sh -p /workspace/isaaclab/source/ASBDualArm/scripts/env/state_machine/lift_organ_fixed.py --num_envs 1 --enable_cameras --usd_path /workspace/isaaclab/source/ASBDualArm/source/msr/msr/assets/others/msr_organ.usd