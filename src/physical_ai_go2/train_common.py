# [解説] このファイルは train_walk.py などの学習スクリプトが共通で使う部品。
# [解説] get_train_cfg は PPO とネットワークの設定、get_base_cfgs はロボットと観測の設定、run は学習の実行を担当する。
# [解説] 報酬と指令の設定は学習内容ごとに違うので、各 train_*.py の get_cfgs に書かれている。
import argparse
import os
import pickle
import shutil
from importlib import metadata

# [解説] rsl_rl のバージョン確認。5 未満ならインストールを促すエラーで止める。
try:
    if int(metadata.version("rsl-rl-lib").split(".")[0]) < 5:
        raise ImportError
except (metadata.PackageNotFoundError, ImportError) as e:
    raise ImportError("Please install 'rsl-rl-lib>=5.0.0'.") from e
import torch
from rsl_rl.runners import OnPolicyRunner

import genesis as gs

from physical_ai_go2.go2_env import Go2Env


# [解説] rsl_rl の OnPolicyRunner に渡す学習設定を作る。
def get_train_cfg(exp_name):
    train_cfg_dict = {
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
        # [解説] 各ネットワークに渡す観測の指定。"policy" は Go2Env.get_observations が返す観測の名前。
        # [解説] actor と critic は同じ観測を見ている。
        "obs_groups": {
            "actor": ["policy"],
            "critic": ["policy"],
        },
        # [解説] 1 イテレーションで環境ごとに集めるステップ数。24 * 4096 環境 = 98304 ステップ分のデータになる。
        "num_steps_per_env": 24,
        # [解説] モデルを保存する間隔（イテレーション）。
        "save_interval": 50,
        "run_name": exp_name,
        # [解説] 学習の経過を TensorBoard 形式で記録する。
        "logger": "tensorboard",
    }

    return train_cfg_dict


def get_base_cfgs():
    # robot, PD and observation settings shared by every policy; each train_*.py adds its rewards and commands
    env_cfg = {
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
        # [解説] 実機の 1 ステップ遅れを再現するかどうか。ただし今は Go2Env が True に固定していて、この値は読まれない。
        "simulate_action_latency": True,
        # [解説] 行動をこの範囲（±100）に切り詰める。極端な値を防ぐための安全策。
        "clip_actions": 100.0,
    }
    obs_cfg = {
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
    return env_cfg, obs_cfg


# [解説] 歩行のポリシー（観測 45 次元）の重みを、歩行＆ジャンプ（観測 47 次元）の初期値として読み込む。
# [解説] 47 次元の観測は、45 次元の 9 番目（速度指令の直後）に跳べフラグ、最後にジャンプの進み具合が増えた形。
# [解説] 最初の層の重みのその 2 列を 0 にして足すので、読み込んだ直後は 2 つの入力を無視して、元の歩行と同じ動きをする。
def load_walking_weights(runner, path, device):
    loaded = torch.load(path, weights_only=False, map_location=device)
    for key in ("actor_state_dict", "critic_state_dict"):
        w = loaded[key]["mlp.0.weight"]
        if w.shape[1] == 45:
            zero = torch.zeros(w.shape[0], 1, device=w.device)
            loaded[key]["mlp.0.weight"] = torch.cat([w[:, :9], zero, w[:, 9:], zero], dim=1)
    # [解説] 最適化の状態（Adam の統計）と学習回数は引き継がず、ネットワークの重みと行動のばらつき（std）だけ使う。
    runner.alg.load(loaded, {"actor": True, "critic": True}, strict=True)


# [解説] 学習の本体。get_cfgs は各 train_*.py の設定関数、default_exp_name は -e を省略したときの実験名。
# [解説] 流れは 引数の解析 → 設定の用意 → ログフォルダの準備 → 設定の保存 → Genesis と環境の作成 → 学習。
def run(get_cfgs, default_exp_name):
    # [解説] コマンドライン引数の定義。
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e", "--exp-name", type=str, default=default_exp_name, help="Experiment name; also the log directory under logs/"
    )
    parser.add_argument("-b", "--num-envs", type=int, default=4096, help="Number of parallel environments")
    parser.add_argument("--max-iterations", type=int, default=500, help="Number of learning iterations")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--overwrite", action="store_true", help="Delete an existing logs/<exp-name> and retrain")
    parser.add_argument(
        "--reward-scale",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Override a reward scale, e.g. --reward-scale action_rate=0 (repeatable)",
    )
    parser.add_argument(
        "--init-from", type=str, default=None, metavar="MODEL_PT", help="Start from the weights of a saved model"
    )
    args = parser.parse_args()

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    # [解説] --reward-scale action_rate=0 のような指定で報酬の重みを上書きする。存在しない報酬名ならエラーで止める。
    for item in args.reward_scale:
        name, _, value = item.partition("=")
        if name not in reward_cfg["reward_scales"] or not value:
            parser.error(f"--reward-scale {item}: expected NAME=VALUE with NAME in {list(reward_cfg['reward_scales'])}")
        reward_cfg["reward_scales"][name] = float(value)
    train_cfg = get_train_cfg(args.exp_name)

    # [解説] 過去の学習結果を誤って消さないよう、同じ名前のフォルダがあれば止める。--overwrite のときだけ削除する。
    if os.path.exists(log_dir):
        if not args.overwrite:
            parser.error(f"{log_dir} already exists. Use another --exp-name, or --overwrite to delete it and retrain.")
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    # [解説] 設定をファイルに保存する。teleop.py はこれを読んで環境を作る（報酬は空にし、エピソードの上限は伸ばす）。
    # [解説] Go2Env は reward_scales を dt 倍に書き換えるが、ここは環境を作る前なので元の値が保存される。
    with open(f"{log_dir}/cfgs.pkl", "wb") as f:
        pickle.dump([env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg], f)

    # Training throughput comes from the 4096 parallel environments, which need a GPU (Metal on Apple Silicon).
    gs.init(backend=gs.gpu, precision="32", logging_level="warning", seed=args.seed, performance_mode=True)

    # [解説] 並列環境を作る。show_viewer は省略しているので画面は出ない。
    env = Go2Env(
        num_envs=args.num_envs, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg, command_cfg=command_cfg
    )

    # [解説] 環境、方策、PPO をまとめて学習を回す rsl_rl のクラス。モデルとログは log_dir に保存される。
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    if args.init_from:
        load_walking_weights(runner, args.init_from, gs.device)

    # [解説] init_at_random_ep_len は各環境の経過ステップをランダムにする設定。全環境のリセットが同時に起きるのを防ぐ。
    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)

