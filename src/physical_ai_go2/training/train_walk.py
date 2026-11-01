"""
Walking policy: follows forward / sideways / turning velocity commands (no jump flag).

uv run go2-train-walk    # logs/go2-walking/
"""

from physical_ai_go2.training.training_config import create_observation_config, create_robot_config
from physical_ai_go2.training.training_runner import run_training


def create_reward_config():
    return {
        "tracking_sigma": 0.25,
        "base_height_target": 0.3,
        "feet_height_target": 0.075,
        "reward_scales": {
            "tracking_lin_vel": 1.0,
            "tracking_ang_vel": 0.2,
            "lin_vel_z": -1.0,
            "base_height": -50.0,
            "action_rate": -0.005,
            "similar_to_default": -0.1,
        },
    }


def create_command_config():
    return {
        "num_commands": 3,  # vx, vy, yaw rate
        # wide command ranges so that the policy can be steered by keyboard (see teleop.py)
        "lin_vel_x_range": [-0.5, 1.0],
        "lin_vel_y_range": [-0.5, 0.5],
        "ang_vel_range": [-1.0, 1.0],
    }


def create_configs():
    return create_robot_config(), create_observation_config(), create_reward_config(), create_command_config()


def main():
    run_training(create_configs, default_experiment_name="go2-walking")


if __name__ == "__main__":
    main()
