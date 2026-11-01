"""
Walking + jumping policy: follows velocity commands and jumps while the jump flag is raised.

uv run go2-train-walk-jump    # logs/go2-walk-jump/
"""

from training import train_jump
from training.training_config import create_observation_config, create_robot_config
from training.training_runner import run_training


def create_reward_config():
    # jump rewards found for the in-place jump (train_jump.py), on top of the walking command ranges below
    # [解説] 報酬はその場ジャンプで見つけた設定（train_jump.py、仮説20）をそのまま使う。
    # [解説] 歩行の報酬（tracking_*、action_rate、similar_to_default など）もその中に含まれている。
    # [解説] 違いは指令の範囲（歩く速度の指令を出す）と、跳べフラグを立てる確率だけ。
    reward_config = train_jump.create_reward_config()
    # [解説] 仮説W2: 体高の罰を歩行用の -50 に戻す（-500 はその場ジャンプで、しゃがんで待つのを防ぐために強めた値）。
    reward_config["reward_scales"]["base_height"] = -50.0
    # [解説] 仮説W2: 途中で終了（転倒・傾き）したら 1 回だけ罰。1 ステップぶん（0.02 秒）なので、重み -500 で 1 回 -10。
    # [解説] わざと傾けてエピソードを終わらせる方が得にならないようにする。
    reward_config["reward_scales"]["termination"] = -500.0
    return reward_config


def create_command_config():
    return {
        # [解説] 指令を 3 つから 4 つに増やした。4 つ目が跳べフラグ。go2_environment.py はこの数を見てジャンプの仕組みを有効にする。
        "num_commands": 4,  # vx, vy, yaw rate, jump flag
        # wide command ranges so that the policy can be steered by keyboard (see teleop.py)
        "lin_vel_x_range": [-0.5, 1.0],
        "lin_vel_y_range": [-0.5, 0.5],
        "ang_vel_range": [-1.0, 1.0],
        # jump flag: raised with this probability at each command resample, then held for jump_duration_s
        # [解説] 4秒ごとの指令の引き直しで、ジャンプを始める確率。大きくするとジャンプの練習量が増え、歩行の練習量が減る。
        # [解説] 仮説W4: 0.3 だとフラグが立つ時間は約 7.5% で、ジャンプの練習量が足りない。1.0 にして約 25%（その場ジャンプと同じ）にする。
        "jump_prob": 1.0,
        # [解説] 跳べフラグを立てておく時間（秒）。この間がジャンプ1回ぶん。報酬で使うジャンプの経過時間もこの範囲で動く。
        "jump_duration_s": 1.0,
    }


def create_configs():
    return create_robot_config(), create_observation_config(), create_reward_config(), create_command_config()


def main():
    run_training(create_configs, default_experiment_name="go2-walk-jump")


if __name__ == "__main__":
    main()
