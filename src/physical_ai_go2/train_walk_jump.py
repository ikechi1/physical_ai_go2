"""
Walking + jumping policy: follows velocity commands and jumps while the jump flag is raised.

uv run go2-train-walk-jump    # logs/go2-walk-jump/
"""

from physical_ai_go2.train_common import get_base_cfgs, run


def get_cfgs():
    env_cfg, obs_cfg = get_base_cfgs()
    # TODO: jump rewards. These are the walking rewards as a starting point; lin_vel_z and base_height
    # penalize leaving the ground, so they need to be revised for jumping.
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
        # [解説] 指令を 3 つから 4 つに増やした。4 つ目が跳べフラグ。go2_env.py はこの数を見てジャンプの仕組みを有効にする。
        "num_commands": 4,  # vx, vy, yaw rate, jump flag
        # wide command ranges so that the policy can be steered by keyboard (see teleop.py)
        "lin_vel_x_range": [-0.5, 1.0],
        "lin_vel_y_range": [-0.5, 0.5],
        "ang_vel_range": [-1.0, 1.0],
        # jump flag: raised with this probability at each command resample, then held for jump_duration_s
        # [解説] 4秒ごとの指令の引き直しで、ジャンプを始める確率。大きくするとジャンプの練習量が増え、歩行の練習量が減る。
        "jump_prob": 0.3,
        # [解説] 跳べフラグを立てておく時間（秒）。この間がジャンプ1回ぶん。報酬で使う jump_time もこの範囲で動く。
        "jump_duration_s": 1.0,
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg


def main():
    run(get_cfgs, default_exp_name="go2-walk-jump")


if __name__ == "__main__":
    main()
