# MSR Surgical Robotics - Isaac Lab Project

## Overview

This project implements reinforcement/imitation learning environments and policies for the **Modular Surgical Robot (MSR)** using NVIDIA Isaac Lab. It provides a comprehensive suite of surgical manipulation tasks with both **PSM (Patient Side Manipulator)** and **ECM (Endoscopic Camera Manipulator)** arms, spanning **rigid object manipulation**, **deformable tissue interaction**, and **fluid dynamics simulation** for realistic surgical training scenarios.

**Key Features:**

- **Multi-Robot Operations**: PSM (Patient Side Manipulator) and ECM (Endoscopic Camera) control
- **Multi-stage RL Tasks**: Reach, pick, lift, place, handover, and camera tracking operations
- **Diverse Simulation Types**: 
  - Rigid object manipulation (blocks, needles, surgical tools)
  - Deformable tissue interaction (soft body dynamics)
  - Fluid dynamics simulation (bleeding, irrigation)
- **State Machine Validation**: Pre-trained scripted behaviors for testing environments
- **Flexible Action Spaces**: Joint position and IK-based relative control
- **Comprehensive Reward Shaping**: Engineered rewards for approach, grasping, lifting, and orientation control
- **Multi-Robot Coordination**: Bi-manual manipulation and camera-tool coordination

**Keywords:** surgical robotics, reinforcement learning, isaac lab, psm, pick and place

---

## Installation

### Prerequisites

1. **Install Isaac Lab** by following the [official installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)
   - Recommended: Use conda installation for easier Python environment management

2. **Clone this repository** outside the `IsaacLab` directory:
   ```bash
   git clone <repository-url> ASBDualArm
   cd ASBDualArm
   ```

3. **Install the extension** in editable mode:
   ```bash
   # If Isaac Lab is installed in conda/venv:
   python -m pip install -e source/asb_dual_arm
   
   # If using isaaclab.sh wrapper:
   # ./isaaclab.sh -p -m pip install -e source/asb_dual_arm
   ```

### Verification

List available tasks:
```bash
python scripts/list_envs.py
```

Expected output should include tasks like:
- `Isaac-Reach-MSRPSM-*`
- `Isaac-Lift-Needle-MSRPSM-*`  
- `Isaac-Lift-Block-MSRPSM-*`

---

## Quick Start

### 1. Test with State Machines (Scripted Policies)

State machines provide scripted demonstrations for validating environments:

```bash
# Reach target pose (baseline task)
python scripts/env/state_machine/reach_msrpsm_sm.py --num_envs 4

# Full end-to-end lift task
python scripts/env/state_machine/lift_needle_msrpsm_rel_sm.py --num_envs 4

# Block manipulation
python scripts/env/state_machine/lift_block_msrpsm_sm.py --num_envs 4
```

### 2. Train RL Policies

Train policies using SKRL (PPO algorithm):

```bash
# Reach task (baseline)
python scripts/skrl/train.py --task Isaac-Reach-MSRPSM-IK-Rel-v0 --headless

# Train end-to-end lift policy
python scripts/skrl/train.py --task Isaac-Lift-Needle-MSRPSM-IK-Rel-v1 --headless
```

**Training Tips:**
- Remove `--headless` to visualize training
- Logs saved to `./logs/skrl/<experiment_name>/`
- Training typically takes several hours on modern GPUs

### 3. Monitor Training with TensorBoard

```bash
# Monitor specific experiment
tensorboard --logdir ./logs/skrl/lift_needle_grasp/

# Monitor all experiments
tensorboard --logdir ./logs/skrl/
```

Access at `http://localhost:6006` to view:
- Episode rewards (reaching, aligning, grasping, lifting, etc.)
- Success rates
- Policy loss and entropy

### 4. Play Trained Policies

Load and visualize trained policies:

```bash
# Play trained policy
python scripts/skrl/play.py \
    --task Isaac-Lift-Needle-MSRPSM-IK-Rel-v1 \
    --num_envs 4 \
    --checkpoint ./logs/skrl/<experiment_dir>/<timestamp>/best_agent.pt
```

**Checkpoint Tips:**
- Use `best_agent.pt` for best-performing policy
- Use `last_agent.pt` for most recent checkpoint
- Find checkpoints in experiment log directories with timestamps

---

## Project Structure

```
ASBDualArm/
├── scripts/
│   ├── skrl/                    # RL training & evaluation
│   │   ├── train.py             # Train policies
│   │   ├── play.py              # Play trained policies
│   │   └── infer.py             # Run inference
│   └── env/
│       └── state_machine/       # Scripted demonstrations
│           ├── lift_needle_*    # Needle manipulation SMs
│           └── lift_block_*     # Block manipulation SMs
├── source/asb_dual_arm/asb_dual_arm/
│   ├── tasks/direct/            # Task definitions
│   │   ├── lift/                # Lift task environments
│   │   │   ├── lift_env_cfg.py  # Base config (full task)
│   │   │   ├── msr/             # MSR-specific configs
│   │   │   │   ├── ik_rel_env_cfg.py  # IK-based control
│   │   │   │   └── agents/      # RL hyperparameters
│   │   │   └── mdp/             # MDP components
│   │   │       ├── rewards.py   # Reward functions
│   │   │       ├── terminations.py
│   │   │       └── events.py
│   │   └── reach/               # Reach task (baseline)
│   ├── assets/                  # Robot URDF/USD files
│   └── config/robot/msr.py      # Robot configuration
└── logs/skrl/                   # Training logs & checkpoints
```

---

## Available Tasks

| Task Name | Description | Action Space |
|-----------|-------------|--------------|
| `Isaac-Reach-MSRPSM-IK-Rel-v0` | Reach target pose (no object) | 6-DoF EE delta + gripper |
| `Isaac-Lift-Needle-MSRPSM-IK-Rel-v1` | **Full lift task** (approach → grasp → lift → place) | 6-DoF EE delta + gripper |

| `Isaac-Lift-Block-MSRPSM-IK-Rel-v0` | Block lift task | 6-DoF EE delta + gripper |

---

## Reward Structure (Full Lift Task)

The full end-to-end lift task uses the following rewards:

| Reward Term | Weight | Purpose |
|-------------|--------|---------|
| `reaching_object` | 5.0 | Encourage EE to approach object |
| `aligning_object` | 15.0 | Encourage correct gripper orientation |
| `grasp_success` | 50.0 | Conditional gripper control (open when far, close when near) |
| `lifting_object` | 15.0 | Reward lifting object above threshold |
| `object_goal_tracking` | 16.0 | Encourage moving object to target position |
| `object_goal_tracking_fine_grained` | 5.0 | Fine-grained positioning reward |
| `object_goal_orientation_tracking` | 10.0 | Encourage matching target orientation |
| `action_rate` | -0.0001 | Penalize large actions (smoothness) |

**Note**: The `grasp_success` reward handles the full gripper behavior - no additional gripper rewards needed!

---

## Key Configuration Files

### Robot Configuration
- `source/asb_dual_arm/asb_dual_arm/config/robot/msr.py`: MSR PSM robot articulation, actuators, PD gains

### Environment Configs
- `source/asb_dual_arm/asb_dual_arm/tasks/direct/lift/lift_env_cfg.py`: Full lift task (base)
- `source/asb_dual_arm/asb_dual_arm/tasks/direct/lift/msr/ik_rel_env_cfg.py`: IK control variants and stage configs

### RL Hyperparameters
- `source/asb_dual_arm/asb_dual_arm/tasks/direct/lift/msr/agents/skrl_ppo_cfg_pick.yaml`: Pick stage PPO config
- `source/asb_dual_arm/asb_dual_arm/tasks/direct/lift/msr/agents/skrl_ppo_cfg_lift_pregrasp.yaml`: Lift stage PPO config

---


## Development Workflow

### Adding New Rewards

1. Define reward function in `source/asb_dual_arm/asb_dual_arm/tasks/direct/lift/mdp/rewards.py`
2. Add `RewTerm` to environment config's `RewardsCfg`
3. Test with state machine before RL training

### Creating New Tasks

1. Create environment config inheriting from `LiftEnvCfg` or similar
2. Register in `source/asb_dual_arm/asb_dual_arm/tasks/direct/lift/msr/__init__.py`
3. Add YAML config in `agents/` directory
4. Create state machine for validation (optional but recommended)

### Tips for Stable Training

- Start with state machine validation

- Monitor all reward components in TensorBoard
- Tune PD gains if robot is unstable
- Disable `gripper_closed` reward - `grasp_success` is sufficient

---

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{msr-isaaclab-2025,
  title={MSR Surgical Robotics with Isaac Lab},
  author={Your Name},
  year={2025},
  publisher={GitHub},
  url={https://github.com/your-repo/ASBDualArm}
}
```

---

## License

This project is licensed under the BSD-3-Clause License - see LICENSE file for details.

## Acknowledgments

- Built on [NVIDIA Isaac Lab](https://github.com/isaac-sim/IsaacLab)
- Uses [SKRL](https://github.com/Toni-SM/skrl) for RL training
- MSR robot design from [MSR Project](https://github.com/your-msr-repo)