"""
Walking policy: follows forward / sideways / turning velocity commands (no jump flag).

uv run go2-train-walk    # logs/go2-walking/
"""

from physical_ai_go2.train_common import get_base_cfgs, run


def get_cfgs():
    env_cfg, obs_cfg = get_base_cfgs()
    reward_cfg = {
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
    command_cfg = {
        "num_commands": 3,  # vx, vy, yaw rate
        # wide command ranges so that the policy can be steered by keyboard (see teleop.py)
        "lin_vel_x_range": [-0.5, 1.0],
        "lin_vel_y_range": [-0.5, 0.5],
        "ang_vel_range": [-1.0, 1.0],
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg


def main():
    run(get_cfgs, default_exp_name="go2-walking")


if __name__ == "__main__":
    main()
