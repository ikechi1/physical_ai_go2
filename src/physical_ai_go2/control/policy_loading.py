import glob
import os
import pickle
import re

from rsl_rl.runners import OnPolicyRunner

import genesis

from physical_ai_go2.environment.go2_environment import Go2Environment


# [解説] logs/<実験名>/model_<回数>.pt のうち、学習回数がいちばん大きいファイルを返す。
def find_latest_checkpoint(log_directory):
    checkpoint_paths = glob.glob(os.path.join(log_directory, "model_*.pt"))
    if len(checkpoint_paths) == 0:
        raise FileNotFoundError(
            "No checkpoint in " + log_directory + ". Train first with: uv run go2-train-walk "
            "(or go2-train-jump / go2-train-walk-jump)"
        )
    latest_path = None
    latest_iteration = -1
    for checkpoint_path in checkpoint_paths:
        iteration = int(re.search(r"model_(\d+)\.pt", checkpoint_path).group(1))
        if iteration > latest_iteration:
            latest_iteration = iteration
            latest_path = checkpoint_path
    return latest_path


# [解説] 学習済みポリシーを動かすための環境とポリシーを作る（キーボード操作や動画の撮影用）。
# [解説] 報酬は使わないので空にし、エピソードの上限は実質なしにする（転倒・傾きのときだけリセットされる）。
class LoadedPolicy:
    def __init__(
        self, experiment_name, environment_count, checkpoint_iteration=None, show_viewer=False, camera_resolution=None
    ):
        log_directory = "logs/" + experiment_name
        with open(log_directory + "/cfgs.pkl", "rb") as config_file:
            saved_configs = pickle.load(config_file)
        environment_config = saved_configs[0]
        observation_config = saved_configs[1]
        reward_config = saved_configs[2]
        command_config = saved_configs[3]
        ppo_config = saved_configs[4]
        reward_config["reward_scales"] = {}
        environment_config["episode_length_s"] = 1e6

        self.environment = Go2Environment(
            environment_count=environment_count,
            environment_config=environment_config,
            observation_config=observation_config,
            reward_config=reward_config,
            command_config=command_config,
            show_viewer=show_viewer,
            camera_resolution=camera_resolution,
        )
        runner = OnPolicyRunner(self.environment, ppo_config, log_directory, device=genesis.device)
        if checkpoint_iteration is None:
            self.checkpoint_path = find_latest_checkpoint(log_directory)
        else:
            self.checkpoint_path = os.path.join(log_directory, "model_" + str(checkpoint_iteration) + ".pt")
        runner.load(self.checkpoint_path)
        self.policy = runner.get_inference_policy(device=genesis.device)
