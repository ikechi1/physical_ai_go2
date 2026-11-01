import argparse
import os
import pickle
import shutil
from importlib import metadata

# [解説] rsl_rl のバージョン確認。5 未満ならインストールを促すエラーで止める。
try:
    if int(metadata.version("rsl-rl-lib").split(".")[0]) < 5:
        raise ImportError
except (metadata.PackageNotFoundError, ImportError) as error:
    raise ImportError("Please install 'rsl-rl-lib>=5.0.0'.") from error
from rsl_rl.runners import OnPolicyRunner

import genesis

from physical_ai_go2.go2_environment import Go2Environment
from physical_ai_go2.training_config import create_ppo_config
from physical_ai_go2.walking_weights import load_walking_weights


def create_argument_parser(default_experiment_name):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e",
        "--exp-name",
        type=str,
        default=default_experiment_name,
        help="Experiment name; also the log directory under logs/",
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
    return parser


# [解説] --reward-scale action_rate=0 のような指定で報酬の重みを上書きする。存在しない報酬名ならエラーで止める。
def apply_reward_scale_overrides(parser, reward_config, overrides):
    reward_scales = reward_config["reward_scales"]
    for override in overrides:
        name, separator, value = override.partition("=")
        if name not in reward_scales or value == "":
            parser.error(
                "--reward-scale " + override + ": expected NAME=VALUE with NAME in " + str(list(reward_scales.keys()))
            )
        reward_scales[name] = float(value)


# [解説] 過去の学習結果を誤って消さないよう、同じ名前のフォルダがあれば止める。--overwrite のときだけ削除する。
def prepare_log_directory(parser, log_directory, overwrite):
    if os.path.exists(log_directory):
        if not overwrite:
            parser.error(
                log_directory + " already exists. Use another --exp-name, or --overwrite to delete it and retrain."
            )
        shutil.rmtree(log_directory)
    os.makedirs(log_directory, exist_ok=True)


# [解説] 学習の本体。create_configs は各 train_*.py の設定関数、default_experiment_name は -e を省略したときの実験名。
# [解説] 流れは 引数の解析 → 設定の用意 → ログフォルダの準備 → 設定の保存 → Genesis と環境の作成 → 学習。
def run_training(create_configs, default_experiment_name):
    parser = create_argument_parser(default_experiment_name)
    arguments = parser.parse_args()

    log_directory = "logs/" + arguments.exp_name
    environment_config, observation_config, reward_config, command_config = create_configs()
    apply_reward_scale_overrides(parser, reward_config, arguments.reward_scale)
    ppo_config = create_ppo_config(arguments.exp_name)
    prepare_log_directory(parser, log_directory, arguments.overwrite)

    # [解説] 設定をファイルに保存する。teleop.py などはこれを読んで環境を作る（報酬は空にし、エピソードの上限は伸ばす）。
    with open(log_directory + "/cfgs.pkl", "wb") as config_file:
        pickle.dump([environment_config, observation_config, reward_config, command_config, ppo_config], config_file)

    # Training throughput comes from the 4096 parallel environments, which need a GPU (Metal on Apple Silicon).
    genesis.init(
        backend=genesis.gpu, precision="32", logging_level="warning", seed=arguments.seed, performance_mode=True
    )

    environment = Go2Environment(
        environment_count=arguments.num_envs,
        environment_config=environment_config,
        observation_config=observation_config,
        reward_config=reward_config,
        command_config=command_config,
    )

    # [解説] 環境、方策、PPO をまとめて学習を回す rsl_rl のクラス。モデルとログは log_directory に保存される。
    runner = OnPolicyRunner(environment, ppo_config, log_directory, device=genesis.device)
    if arguments.init_from is not None:
        load_walking_weights(runner, arguments.init_from, genesis.device)

    # [解説] init_at_random_ep_len は各環境の経過ステップをランダムにする設定。全環境のリセットが同時に起きるのを防ぐ。
    runner.learn(num_learning_iterations=arguments.max_iterations, init_at_random_ep_len=True)
