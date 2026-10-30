"""
Jumping policy: stands still and jumps while the jump flag is raised (no velocity commands).

uv run go2-train-jump    # logs/go2-jump/
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
        # [解説] 跳べフラグを使うため指令は 4 つ。速度指令の範囲を 0 にして、その場で立つ・跳ぶだけを学習させる。
        # [解説] 歩行＆ジャンプ用と観測の形が同じなので、学習結果を比べやすい。
        "num_commands": 4,  # vx, vy, yaw rate, jump flag
        "lin_vel_x_range": [0.0, 0.0],
        "lin_vel_y_range": [0.0, 0.0],
        "ang_vel_range": [0.0, 0.0],
        # jump flag: raised with this probability at each command resample, then held for jump_duration_s
        # [解説] 1.0 なら4秒ごとに必ず跳ぶ（1.0 秒跳んで、残り 3 秒は立っている）。
        "jump_prob": 1.0,
        # [解説] 跳べフラグを立てておく時間（秒）。この間がジャンプ1回ぶん。報酬で使う jump_time もこの範囲で動く。
        "jump_duration_s": 1.0,
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg


def main():
    run(get_cfgs, default_exp_name="go2-jump")


if __name__ == "__main__":
    main()
