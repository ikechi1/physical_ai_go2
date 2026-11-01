import torch

import genesis


# [解説] 速度指令（前後の速度、左右の速度、旋回速度）を環境ごとに持つクラス。
# [解説] 学習中は指定の範囲から一様乱数で引き直し、キーボード操作中は外から決めた値を使う。
class VelocityCommands:
    def __init__(self, environment_count, command_config):
        self.environment_count = environment_count
        # [解説] 列 0 が前後の速度 [m/s]、列 1 が左右の速度 [m/s]、列 2 が旋回速度 [rad/s]。
        self.values = torch.zeros((environment_count, 3), dtype=genesis.tc_float, device=genesis.device)
        self.lower_limits = torch.tensor(
            [
                command_config["lin_vel_x_range"][0],
                command_config["lin_vel_y_range"][0],
                command_config["ang_vel_range"][0],
            ],
            dtype=genesis.tc_float,
            device=genesis.device,
        )
        self.upper_limits = torch.tensor(
            [
                command_config["lin_vel_x_range"][1],
                command_config["lin_vel_y_range"][1],
                command_config["ang_vel_range"][1],
            ],
            dtype=genesis.tc_float,
            device=genesis.device,
        )
        # if True, commands are set externally (teleop) and never resampled
        self.is_manual = False

    # [解説] キーボード操作などで、全環境の速度指令を外から決める。以後は引き直さない。
    def set_manual(self, forward_velocity, sideways_velocity, turning_velocity):
        self.is_manual = True
        self.values[:, :] = torch.tensor(
            [forward_velocity, sideways_velocity, turning_velocity], dtype=genesis.tc_float, device=genesis.device
        )

    # [解説] resample_mask が True の環境だけ、新しい速度指令に置き換える。それ以外は今の値のまま。
    def resample(self, resample_mask):
        random_values = torch.rand(size=(self.environment_count, 3), dtype=genesis.tc_float, device=genesis.device)
        new_values = (self.upper_limits - self.lower_limits) * random_values + self.lower_limits
        self.values[:, :] = torch.where(resample_mask[:, None], new_values, self.values)
