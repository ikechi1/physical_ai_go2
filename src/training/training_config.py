# [解説] 学習スクリプト（train_walk.py など）が共通で使う設定。
# [解説] 報酬と指令の設定は学習内容ごとに違うので、各 train_*.py の create_configs に書かれている。
# [解説] 辞書のキー名は保存済みの設定（logs/*/cfgs.pkl）と rsl_rl が読むので、元の名前のまま。


# [解説] rsl_rl の OnPolicyRunner に渡す学習設定を作る。
def create_ppo_config(experiment_name):
    return {
        # [解説] 学習アルゴリズム PPO の設定。
        "algorithm": {
            "class_name": "PPO",
            # [解説] 1 回の更新で方策を変えてよい幅。行動の確率の比を 1 ± 0.2 に制限する。
            "clip_param": 0.2,
            # [解説] 更新前後の方策の差（KL ダイバージェンス）の目標。schedule が adaptive のときに使う。
            "desired_kl": 0.01,
            # [解説] 探索を促す項の重み。大きいほど行動のばらつきを保とうとする。
            "entropy_coef": 0.01,
            # [解説] 割引率。将来の報酬をどれだけ重視するか。1 ステップ先の報酬を 0.99 倍して数える。
            "gamma": 0.99,
            # [解説] GAE（行動の良さの推定方法）の係数。
            "lam": 0.95,
            # [解説] 学習率の初期値。adaptive なので学習中に変わる。
            "learning_rate": 0.001,
            # [解説] 勾配の大きさの上限。更新が急に大きくなるのを防ぐ。
            "max_grad_norm": 1.0,
            # [解説] 集めたデータを何周使って学習するか。
            "num_learning_epochs": 5,
            # [解説] 1 周のデータをいくつに分けて更新するか。1 イテレーションの更新回数は 5 * 4 = 20 回。
            "num_mini_batches": 4,
            # [解説] KL が desired_kl より大きくずれたら学習率を下げ、小さすぎたら上げる。
            "schedule": "adaptive",
            # [解説] 価値関数の更新にも clip_param と同様の制限をかける。
            "use_clipped_value_loss": True,
            # [解説] 損失全体の中での価値関数の重み。
            "value_loss_coef": 1.0,
        },
        # [解説] 方策のネットワーク。観測を受け取り、行動を出す。
        "actor": {
            "class_name": "MLPModel",
            # [解説] 隠れ層のユニット数。全結合の 3 層。
            "hidden_dims": [512, 256, 128],
            # [解説] 活性化関数。
            "activation": "elu",
            # [解説] 行動は正規分布からサンプリングする。ネットワークの出力が平均で、標準偏差は別に学習する。
            "distribution_cfg": {
                "class_name": "GaussianDistribution",
                # [解説] 標準偏差の初期値。最初は行動のばらつきが大きく、学習が進むと小さくなっていく。
                "init_std": 1.0,
                "std_type": "scalar",
            },
        },
        # [解説] 価値関数のネットワーク。今の状態の良さを推定する。学習中にだけ使い、ロボットを動かすときは使わない。
        "critic": {
            "class_name": "MLPModel",
            "hidden_dims": [512, 256, 128],
            "activation": "elu",
        },
        # [解説] 各ネットワークに渡す観測の指定。"policy" は Go2Environment.get_observations が返す観測の名前。
        # [解説] actor と critic は同じ観測を見ている。
        "obs_groups": {
            "actor": ["policy"],
            "critic": ["policy"],
        },
        # [解説] 1 イテレーションで環境ごとに集めるステップ数。24 * 4096 環境 = 98304 ステップ分のデータになる。
        "num_steps_per_env": 24,
        # [解説] モデルを保存する間隔（イテレーション）。
        "save_interval": 50,
        "run_name": experiment_name,
        # [解説] 学習の経過を TensorBoard 形式で記録する。
        "logger": "tensorboard",
    }


# robot and PD settings shared by every policy; each train_*.py adds its rewards and commands
def create_robot_config():
    return {
        # [解説] ポリシーが出力する行動の数。Go2 は 4 本の脚に 3 関節ずつで 12。
        "num_actions": 12,
        # joint/link names
        # [解説] 基準の姿勢（立ち姿）の関節角度。リセット時はこの姿勢に戻り、行動が 0 のときの目標角度にもなる。
        # [解説] hip は脚の開閉、thigh は太もも、calf はすねの関節。後ろ脚だけ thigh が 1.0 と少し大きい。
        "default_joint_angles": {  # [rad]
            "FL_hip_joint": 0.0,
            "FR_hip_joint": 0.0,
            "RL_hip_joint": 0.0,
            "RR_hip_joint": 0.0,
            "FL_thigh_joint": 0.8,
            "FR_thigh_joint": 0.8,
            "RL_thigh_joint": 1.0,
            "RR_thigh_joint": 1.0,
            "FL_calf_joint": -1.5,
            "FR_calf_joint": -1.5,
            "RL_calf_joint": -1.5,
            "RR_calf_joint": -1.5,
        },
        # [解説] 行動と観測の関節の並び順。行動の 0 番目が FR_hip、1 番目が FR_thigh、という対応になる。
        # [解説] FR/FL/RR/RL は 右前/左前/右後/左後 の脚。
        "joint_names": [
            "FR_hip_joint",
            "FR_thigh_joint",
            "FR_calf_joint",
            "FL_hip_joint",
            "FL_thigh_joint",
            "FL_calf_joint",
            "RR_hip_joint",
            "RR_thigh_joint",
            "RR_calf_joint",
            "RL_hip_joint",
            "RL_thigh_joint",
            "RL_calf_joint",
        ],
        # PD
        # [解説] 関節を目標角度へ動かす PD 制御のゲイン。kp は角度のずれを戻す強さ（ばね）、kd は速度を抑える強さ（ダンパ）。
        "kp": 20.0,
        "kd": 0.5,
        # termination
        # [解説] 胴体の傾きがこの角度を超えたら転倒とみなしてエピソードを打ち切り、リセットする。roll は左右、pitch は前後の傾き。
        "termination_if_roll_greater_than": 10,  # degree
        "termination_if_pitch_greater_than": 10,
        # base pose
        # [解説] リセット時の胴体の位置 x, y, z [m]。z = 0.42 は目標の体高 0.3m より高く、少し浮いた状態から始まる。
        "base_init_pos": [0.0, 0.0, 0.42],
        # [解説] リセット時の胴体の向き（クォータニオン w, x, y, z）。この値は回転なし。
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        # [解説] 1 エピソードの長さの上限（秒）。1 ステップ 0.02 秒なので 1000 ステップ。
        "episode_length_s": 20.0,
        # [解説] 速度指令を引き直す間隔（秒）。ジャンプ指令もこのタイミングで出る。
        "resampling_time_s": 4.0,
        # [解説] 行動を関節の目標角度に直す倍率。目標角度 [rad] = 行動 * 0.25 + default_joint_angles。
        "action_scale": 0.25,
        # [解説] 実機の 1 ステップ遅れを再現するかどうか。ただし今は go2_environment.py の SIMULATE_ACTION_LATENCY で True に固定していて、この値は読まれない。
        "simulate_action_latency": True,
        # [解説] 行動をこの範囲（±100）に切り詰める。極端な値を防ぐための安全策。
        "clip_actions": 100.0,
    }


# observation settings shared by every policy
def create_observation_config():
    return {
        # [解説] 観測をポリシーに渡すときの倍率。値の大きさをそろえて学習しやすくするためのもの。
        "obs_scales": {
            # [解説] 胴体の並進速度は観測に入っていないので、vx, vy の速度指令にだけ掛かる。
            "lin_vel": 2.0,
            # [解説] 胴体の角速度と、yaw の速度指令に掛かる。
            "ang_vel": 0.25,
            # [解説] 関節角度（基準の姿勢からのずれ）に掛かる。
            "dof_pos": 1.0,
            # [解説] 関節速度に掛かる。
            "dof_vel": 0.05,
        },
    }
