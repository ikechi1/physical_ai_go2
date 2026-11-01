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
        # [解説] 仮説1（+15cm）は外した。0.15 に戻すと仮説1の設定になる。
        "jump_height_offset": 0.0,  # base_height_target is raised by this while the jump flag is up
        "jump_air_time_target": 0.2,  # air time per flight that earns the jump_air_time reward [s]
        # [解説] 仮説14B: 最低 0.25 秒を超えた滞空だけに払う。落下が 1 回目の飛行になっても本当のジャンプに払えるよう、1 回目だけの制限は外す。
        # [解説] 仮説17: 今のジャンプ（滞空 約 0.26 秒）でも報酬の傾きが出るよう、下限を 0.25 秒から 0.2 秒に下げる。
        "jump_air_time_min": 0.2,  # air time below this earns no jump_air_time reward [s]
        # [解説] 仮説18: 2 回跳びで滞空報酬を 2 回分もらえないよう、1 回目の飛行だけに戻す。
        "jump_air_time_first_flight_only": True,  # only the first flight after the jump flag rises earns air time
        "jump_up_vel_max": 2.0,  # upward velocity above this earns no more jump_up_vel reward [m/s]
        # [解説] 仮説8: しゃがむ時間（約 0.2 秒）を足して、踏み切りの窓を 0.3 秒から 0.5 秒に延ばす。
        "jump_up_vel_window_s": 0.5,  # jump_up_vel is only paid this long after the jump flag rises [s]
        "jump_up_vel_height_limit": 0.45,  # jump_up_vel is only paid while the base is below this height [m]
        "lin_vel_z_off_while_jumping": True,  # drop the lin_vel_z penalty while the jump flag is up
        "base_height_off_while_jumping": True,  # drop the base_height penalty while the jump flag is up
        "feet_height_target": 0.075,
        "reward_scales": {
            "tracking_lin_vel": 1.0,
            "tracking_ang_vel": 0.2,
            "lin_vel_z": -1.0,
            # [解説] 仮説7: フラグが 0 のときの体高の罰を -50 から -500 に強くし、しゃがんで構えずに 0.3m に戻るようにする。
            # [解説] フラグが 1 の間は base_height_off_while_jumping で罰をかけていない。
            "base_height": -500.0,
            "action_rate": -0.005,
            "similar_to_default": -0.1,
            # [解説] 仮説2: 滞空時間の報酬。重み 10 で、1回のジャンプで最大 10 × 0.2 秒 = 2.0。
            # [解説] 仮説16: 高く（長く）跳ぶ動機を強めるため、滞空報酬の重みを 10 から 20 に倍にする。
            "jump_air_time": 20.0,
            # [解説] 仮説3: 上向きの速度の報酬。重み 20 で、上向き 1 m/s のとき 1 秒あたり 20。
            # [解説] 仮説15: 落下の反動で上向き速度を稼ぐ得を減らすため、重みを 20 から 10 に半分にする。
            "jump_up_vel": 10.0,
            # [解説] 仮説10: 胴体が床に触れたら罰。腹打ちは約 0.04 秒（2 ステップ）なので、1 回あたり -200 × 0.04 = -8。
            # [解説] 落下で得られる滞空報酬（最大 10 × 0.2 = 2）より大きくなるようにしている。
            "base_contact": -200.0,
            # [解説] 仮説12: フラグの終わりに、跳び上がった回数がちょうど 1 回なら 1 回だけ払う。
            # [解説] 1 ステップぶん（0.02 秒）しか払わないので、重み 200 で 1 回のジャンプあたり +4。
            "jump_one_flight": 200.0,
            # [解説] 仮説20: 1 回目の飛行のあと、4 本とも接地している時間に報酬（仮説19の 2 回目以降の罰は削除）。
            # [解説] 着地はフラグから約 0.45 秒なので、残り約 0.55 秒ずっと接地していれば 8 × 0.55 ≈ +4.4。
            "landed_stance": 8.0,
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
