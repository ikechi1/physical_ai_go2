import torch

import genesis


# [解説] 報酬を計算するクラス。報酬の名前（設定の reward_scales のキー）と、計算する関数を対応させて持つ。
# [解説] 報酬の名前は TensorBoard の項目名（rew_<名前>）と保存済みの設定（cfgs.pkl）でも使われているので、変えていない。
class Rewards:
    def __init__(self, environment, reward_config, step_seconds):
        self.environment = environment
        self.reward_config = reward_config
        reward_functions_by_name = {
            "tracking_lin_vel": self.tracking_linear_velocity,
            "tracking_ang_vel": self.tracking_angular_velocity,
            "lin_vel_z": self.vertical_velocity,
            "action_rate": self.action_rate,
            "similar_to_default": self.similar_to_default,
            "base_height": self.base_height,
            "jump_air_time": self.jump_air_time,
            "base_contact": self.base_contact,
            "crouch_down_vel": self.crouch_down_velocity,
            "extra_flight": self.extra_flight,
            "landed_stance": self.landed_stance,
            "termination": self.termination,
            "jump_one_flight": self.jump_one_flight,
            "jump_up_vel": self.jump_up_velocity,
        }
        # multiply reward scales by the step time, so that the sum over an episode does not depend on the control rate
        # [解説] 設定にある報酬だけを、設定の順番どおりに使う。重みは 1 ステップの時間（0.02 秒）を掛けておく。
        self.functions = {}
        self.scales = {}
        self.episode_sums = {}
        for name, scale in reward_config["reward_scales"].items():
            self.functions[name] = reward_functions_by_name[name]
            self.scales[name] = scale * step_seconds
            self.episode_sums[name] = torch.zeros(
                (environment.num_envs,), dtype=genesis.tc_float, device=genesis.device
            )

    # [解説] 全報酬を「関数の値 × 重み」で足し合わせて返す。報酬ごとのエピソード合計も記録する。
    def compute(self):
        total = torch.zeros((self.environment.num_envs,), dtype=genesis.tc_float, device=genesis.device)
        for name, reward_function in self.functions.items():
            reward = reward_function() * self.scales[name]
            total += reward
            self.episode_sums[name] += reward
        return total

    # [解説] リセットする環境の報酬ごとのエピソード合計の平均を、1 秒あたりに直して返し、合計を 0 に戻す。
    # [解説] 返す辞書は rsl_rl が TensorBoard に記録する（キーは rew_<名前>）。
    def report_and_clear_episode_sums(self, reset_mask, episode_length_seconds):
        reset_count = reset_mask.sum()
        report = {}
        for name, episode_sum in self.episode_sums.items():
            mean = torch.where(reset_count > 0, episode_sum[reset_mask].sum() / reset_count, 0.0)
            report["rew_" + name] = mean / episode_length_seconds
            episode_sum.masked_fill_(reset_mask, 0.0)
        return report

    def tracking_linear_velocity(self):
        # Tracking of linear velocity commands (xy axes)
        environment = self.environment
        error = torch.sum(
            torch.square(environment.velocity_commands.values[:, :2] - environment.base_linear_velocity[:, :2]), dim=1
        )
        return torch.exp(-error / self.reward_config["tracking_sigma"])

    def tracking_angular_velocity(self):
        # Tracking of angular velocity commands (yaw)
        environment = self.environment
        error = torch.square(environment.velocity_commands.values[:, 2] - environment.base_angular_velocity[:, 2])
        return torch.exp(-error / self.reward_config["tracking_sigma"])

    def vertical_velocity(self):
        # Penalize z axis base linear velocity (optionally not while the jump flag is up)
        environment = self.environment
        reward = torch.square(environment.base_linear_velocity[:, 2])
        if environment.jump_enabled and self.reward_config.get("lin_vel_z_off_while_jumping", False):
            # [解説] 仮説3: 跳べフラグが 1 の間は、上下方向の速度の罰をかけない（跳ぶ邪魔をしないため）。
            reward = reward * ~environment.jump_command.is_jumping()
        return reward

    def action_rate(self):
        # Penalize changes in actions
        environment = self.environment
        return torch.sum(torch.square(environment.last_actions - environment.actions), dim=1)

    def similar_to_default(self):
        # Penalize joint poses far away from default pose
        environment = self.environment
        return torch.sum(torch.abs(environment.joint_positions - environment.default_joint_positions), dim=1)

    def base_height(self):
        # Penalize base height away from target (raised by jump_height_offset while the jump flag is up)
        environment = self.environment
        target = self.reward_config["base_height_target"]
        if environment.jump_enabled:
            # [解説] 仮説1: 跳べフラグが 1 の間は、目標の体高を jump_height_offset だけ上げる。
            # [解説] フラグが 0 のときや、設定に jump_height_offset がないときは今まで通り 0.3m。
            target = target + self.reward_config.get("jump_height_offset", 0.0) * environment.jump_command.flag
        reward = torch.square(environment.base_position[:, 2] - target)
        if environment.jump_enabled and self.reward_config.get("base_height_off_while_jumping", False):
            # [解説] 仮説4: 跳べフラグが 1 の間は、体高の罰をかけない（高く上がっても、しゃがんでも罰されない）。
            reward = reward * ~environment.jump_command.is_jumping()
        return reward

    def jump_air_time(self):
        # Reward every step spent airborne during a jump, between jump_air_time_min and min + target seconds per flight
        # [解説] 仮説2: 跳べフラグが 1 の間、4 本とも浮いているステップごとに報酬 1。
        # [解説] 仮説14B: 空中に出てから min 秒を超えたところから target 秒ぶんだけ払う。
        # [解説] min 秒より短い滞空（脚を浮かせて落ちるだけの落下など）には払わない。
        environment = self.environment
        air_time_min = self.reward_config.get("jump_air_time_min", 0.0)
        airborne_seconds = environment.foot_contact_tracker.airborne_seconds()
        within_target = (airborne_seconds > air_time_min) & (
            airborne_seconds <= air_time_min + self.reward_config["jump_air_time_target"]
        )
        jump_command = environment.jump_command
        reward = (jump_command.is_jumping() & environment.foot_contact_tracker.airborne & within_target).to(
            genesis.tc_float
        )
        if self.reward_config.get("jump_air_time_first_flight_only", False):
            # [解説] 仮説6: 1 回の跳べフラグにつき、最初の滞空だけに報酬を出す。2 回目以降に跳ねても報酬は増えない。
            reward = reward * (jump_command.takeoff_count == 1)
        return reward

    def base_contact(self):
        # Penalize the trunk (base link) touching the ground
        # [解説] 仮説10: 胴体（base リンク）が床に触れていれば、1 ステップごとに 1 の罰（重みは負）。
        return self.environment.foot_contact_tracker.base_contact.float()

    def crouch_down_velocity(self):
        # Penalize squared downward (world frame) base velocity during the crouch phase of a jump
        # [解説] 仮説11: しゃがみ期（フラグから jump_up_vel_start_s 秒）の間、下向きの速度の 2 乗に比例した罰（重みは負）。
        environment = self.environment
        downward_velocity = environment.robot.get_vel()[:, 2].clamp(max=0.0)
        jump_command = environment.jump_command
        crouch_phase = jump_command.is_jumping() & (
            jump_command.elapsed_seconds() <= self.reward_config["jump_up_vel_start_s"]
        )
        return torch.square(downward_velocity) * crouch_phase

    def extra_flight(self):
        # Penalize every takeoff after the first one while the jump flag is up
        # [解説] 仮説19: フラグ中に 2 回目以降の離陸が起きたその瞬間に 1（重みは負）。
        environment = self.environment
        jump_command = environment.jump_command
        extra_takeoff = environment.foot_contact_tracker.takeoff & jump_command.is_jumping()
        return (extra_takeoff & (jump_command.takeoff_count >= 2)).to(genesis.tc_float)

    def landed_stance(self):
        # Reward every step with all four feet on the ground after the first flight of a jump
        # [解説] 仮説20: フラグ中、1 回目の飛行のあとに 4 本とも接地しているステップごとに 1。跳ばなければ 0。
        environment = self.environment
        jump_command = environment.jump_command
        all_feet_down = environment.foot_contact_tracker.foot_contact.all(dim=1)
        return (jump_command.is_jumping() & (jump_command.takeoff_count >= 1) & all_feet_down).to(genesis.tc_float)

    def termination(self):
        # Penalize episodes ending early (falls / tilt), but not time-outs
        # [解説] 仮説W2: 傾きすぎなどでエピソードが途中で終わったステップに 1（重みは負）。時間切れの終了は罰しない。
        environment = self.environment
        return (environment.reset_buffer & ~environment.time_out_buffer).to(genesis.tc_float)

    def jump_one_flight(self):
        # Paid once on the last step of a jump if exactly one takeoff happened while the jump flag was up
        # [解説] 仮説12: フラグが 0 に戻る直前のステップで、跳び上がった回数がちょうど 1 回なら 1 を払う。
        jump_command = self.environment.jump_command
        return (jump_command.is_last_step() & (jump_command.takeoff_count == 1)).to(genesis.tc_float)

    def jump_up_velocity(self):
        # Reward upward (world frame) base velocity during a jump, capped at jump_up_vel_max
        # [解説] 仮説3: 跳べフラグが 1 の間、上向きの速度（m/s）に比例した報酬。下向きの速度は 0 として扱う（罰はしない）。
        environment = self.environment
        upward_velocity = environment.robot.get_vel()[:, 2].clamp(min=0.0, max=self.reward_config["jump_up_vel_max"])
        jump_command = environment.jump_command
        reward = upward_velocity * jump_command.is_jumping()
        if "jump_up_vel_window_s" in self.reward_config:
            # [解説] 仮説5: 跳べフラグが立ってから jump_up_vel_window_s 秒（踏み切り）の間だけ報酬を出す。
            reward = reward * (jump_command.elapsed_seconds() <= self.reward_config["jump_up_vel_window_s"])
        if "jump_up_vel_start_s" in self.reward_config:
            # [解説] 仮説9: 跳べフラグが立ってから jump_up_vel_start_s 秒（しゃがみ期）は報酬を出さない。
            reward = reward * (jump_command.elapsed_seconds() > self.reward_config["jump_up_vel_start_s"])
        if "jump_up_vel_height_limit" in self.reward_config:
            # [解説] 仮説5: 体高が jump_up_vel_height_limit 未満の間だけ報酬を出す。
            reward = reward * (environment.base_position[:, 2] < self.reward_config["jump_up_vel_height_limit"])
        return reward
