import math

import torch
from tensordict import TensorDict

import genesis
from genesis.utils.geom import inv_quat, quat_to_xyz, transform_by_quat, transform_quat_by_quat

from physical_ai_go2.environment.foot_contact_tracker import FootContactTracker
from physical_ai_go2.environment.jump_command import JumpCommand
from physical_ai_go2.environment.rewards import Rewards
from physical_ai_go2.environment.velocity_commands import VelocityCommands

# control frequency on the real robot is 50 Hz
STEP_SECONDS = 0.02
# [解説] 実機の 1 ステップ遅れを再現する（行動は 1 ステップ遅れて関節に届く）。
SIMULATE_ACTION_LATENCY = True


# [解説] Genesis 上の Go2 を並列に動かす強化学習の環境。Genesis の examples/locomotion/go2_env.py が元。
# [解説] rsl_rl の OnPolicyRunner が使う属性（num_envs、num_actions、device、cfg、episode_length_buf、
# [解説] max_episode_length）とメソッド（get_observations、step、reset）は、ライブラリが決めた名前のまま。
# [解説] cfg は rsl_rl がログに記録するためだけに読む。このクラスの中では environment_config を使う。
class Go2Environment:
    def __init__(
        self,
        environment_count,
        environment_config,
        observation_config,
        reward_config,
        command_config,
        show_viewer=False,
        camera_resolution=None,
    ):
        self.environment_config = environment_config
        self.num_envs = environment_count
        self.num_actions = environment_config["num_actions"]
        self.cfg = environment_config
        self.device = genesis.device
        self.max_episode_length = math.ceil(environment_config["episode_length_s"] / STEP_SECONDS)
        self.observation_scales = observation_config["obs_scales"]

        # a 4th command is the jump flag (0 or 1); old 3-command configs keep working without it
        # [解説] 指令の数が 4 つなら「跳べフラグあり」とみなす。logs/go2-walking などは指令が 3 つなので False になる。
        self.jump_enabled = command_config["num_commands"] == 4
        self.jump_probability = 0.0
        if self.jump_enabled:
            self.jump_probability = command_config["jump_prob"]

        self._build_scene(show_viewer, camera_resolution)
        self._find_joints()
        self._create_buffers()

        self.velocity_commands = VelocityCommands(environment_count, command_config)
        self.jump_command = None
        if self.jump_enabled:
            self.jump_command = JumpCommand(environment_count, command_config["jump_duration_s"], STEP_SECONDS)
        self.foot_contact_tracker = FootContactTracker(self.robot, environment_count, STEP_SECONDS)
        self.rewards = Rewards(self, reward_config, STEP_SECONDS)

        self.reset()

    def _build_scene(self, show_viewer, camera_resolution):
        self.scene = genesis.Scene(
            sim_options=genesis.options.SimOptions(dt=STEP_SECONDS, substeps=2),
            rigid_options=genesis.options.RigidOptions(
                enable_self_collision=False,
                # For this locomotion policy, there are usually no more than 20 collision pairs. Setting a low value
                # can save memory. Violating this condition will raise an exception.
                max_collision_pairs=20,
                tolerance=1e-5,
            ),
            vis_options=genesis.options.VisOptions(rendered_envs_idx=[0]),
            viewer_options=genesis.options.ViewerOptions(
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            show_viewer=show_viewer,
        )
        self.scene.add_entity(genesis.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))
        self.robot = self.scene.add_entity(
            genesis.morphs.URDF(
                file="urdf/go2/urdf/go2.urdf",
                pos=self.environment_config["base_init_pos"],
                quat=self.environment_config["base_init_quat"],
            ),
        )
        # [解説] 動画の撮影用（make_video.py）。画面を出さずに描画するカメラ。学習では使わない。
        self.camera = None
        if camera_resolution is not None:
            self.camera = self.scene.add_camera(
                res=camera_resolution, pos=(0.0, -2.0, 0.6), lookat=(0.0, 0.0, 0.3), fov=40
            )
        self.scene.build(n_envs=self.num_envs)

    def _find_joints(self):
        # [解説] 行動の並び（joint_names の順）と、シミュレータの関節の番号を対応させる。
        motor_indices = []
        for joint_name in self.environment_config["joint_names"]:
            motor_indices.append(self.robot.get_joint(joint_name).dof_start)
        self.motor_joint_indices = torch.tensor(motor_indices, dtype=genesis.tc_int, device=genesis.device)
        self.action_order_for_simulator = torch.argsort(self.motor_joint_indices)

        # PD control parameters
        kp_values = []
        kd_values = []
        for joint_index in range(self.num_actions):
            kp_values.append(self.environment_config["kp"])
            kd_values.append(self.environment_config["kd"])
        self.robot.set_dofs_kp(kp_values, self.motor_joint_indices)
        self.robot.set_dofs_kv(kd_values, self.motor_joint_indices)

        default_angles_in_action_order = []
        for joint_name in self.environment_config["joint_names"]:
            default_angles_in_action_order.append(self.environment_config["default_joint_angles"][joint_name])
        self.default_joint_positions = torch.tensor(
            default_angles_in_action_order, dtype=genesis.tc_float, device=genesis.device
        )

    def _create_buffers(self):
        environment_count = self.num_envs
        float_type = genesis.tc_float
        device = genesis.device
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], dtype=float_type, device=device)

        # initial state
        self.initial_base_position = torch.tensor(
            self.environment_config["base_init_pos"], dtype=float_type, device=device
        )
        self.initial_base_quaternion = torch.tensor(
            self.environment_config["base_init_quat"], dtype=float_type, device=device
        )
        self.inverse_initial_base_quaternion = inv_quat(self.initial_base_quaternion)
        # [解説] シミュレータの関節の並び（robot.joints の 2 番目以降）での基準姿勢。
        initial_angles_in_simulator_order = []
        for joint_number in range(1, len(self.robot.joints)):
            joint_name = self.robot.joints[joint_number].name
            initial_angles_in_simulator_order.append(self.environment_config["default_joint_angles"][joint_name])
        self.initial_joint_positions = torch.tensor(initial_angles_in_simulator_order, dtype=float_type, device=device)
        self.initial_qpos = torch.concatenate(
            (self.initial_base_position, self.initial_base_quaternion, self.initial_joint_positions)
        )
        self.initial_projected_gravity = transform_by_quat(self.global_gravity, self.inverse_initial_base_quaternion)

        # state buffers
        self.base_position = torch.empty((environment_count, 3), dtype=float_type, device=device)
        self.base_quaternion = torch.empty((environment_count, 4), dtype=float_type, device=device)
        self.base_euler_angles = torch.empty((environment_count, 3), dtype=float_type, device=device)
        self.base_linear_velocity = torch.empty((environment_count, 3), dtype=float_type, device=device)
        self.base_angular_velocity = torch.empty((environment_count, 3), dtype=float_type, device=device)
        self.projected_gravity = torch.empty((environment_count, 3), dtype=float_type, device=device)
        self.joint_positions = torch.empty((environment_count, self.num_actions), dtype=float_type, device=device)
        self.joint_velocities = torch.empty((environment_count, self.num_actions), dtype=float_type, device=device)
        self.actions = torch.zeros((environment_count, self.num_actions), dtype=float_type, device=device)
        self.last_actions = torch.zeros((environment_count, self.num_actions), dtype=float_type, device=device)
        self.episode_length_buf = torch.empty((environment_count,), dtype=genesis.tc_int, device=device)
        self.reset_buffer = torch.ones((environment_count,), dtype=torch.bool, device=device)
        self.time_out_buffer = torch.zeros((environment_count,), dtype=torch.bool, device=device)
        # [解説] 速度指令を観測に入れるときの倍率（前後・左右の速度と、旋回速度）。
        self.velocity_command_scales = torch.tensor(
            [
                self.observation_scales["lin_vel"],
                self.observation_scales["lin_vel"],
                self.observation_scales["ang_vel"],
            ],
            dtype=float_type,
            device=device,
        )
        self.extras = {}  # extra information for logging

    # [解説] キーボード操作から、全環境の速度指令を決める。跳べフラグは start_jump_for_all で別に管理する。
    def set_velocity_commands(self, forward_velocity, sideways_velocity, turning_velocity):
        self.velocity_commands.set_manual(forward_velocity, sideways_velocity, turning_velocity)

    # [解説] teleop の j キーから呼ばれる。全環境でジャンプを始める（ジャンプ中の環境は無視される）。
    def start_jump_for_all(self):
        self.jump_command.start(self._all_environments_mask())

    def is_jumping(self):
        if not self.jump_enabled:
            return torch.zeros((self.num_envs,), dtype=torch.bool, device=genesis.device)
        return self.jump_command.is_jumping()

    # explicit torch.bool: genesis.tc_bool is int32 on some backends (Metal), which masks do not accept
    def _all_environments_mask(self):
        return torch.ones((self.num_envs,), dtype=torch.bool, device=genesis.device)

    def _resample_commands(self, resample_mask):
        if self.velocity_commands.is_manual:
            return
        self.velocity_commands.resample(resample_mask)
        if self.jump_enabled:
            # each resampled environment starts a jump with probability jump_prob
            # [解説] 引き直しは 4 秒ごと（とリセット時）なので、ジャンプ指令もそのタイミングでしか出ない。
            random_values = torch.rand((self.num_envs,), device=genesis.device)
            start_mask = random_values < self.jump_probability
            self.jump_command.start(start_mask & resample_mask)

    def step(self, actions):
        self._apply_actions(actions)
        self.scene.step()

        self.episode_length_buf += 1
        if self.jump_enabled:
            # [解説] 報酬計算より前に進めるので、報酬関数から見たジャンプの経過時間は、踏み切り直後のステップで 0.02 秒になる。
            self.jump_command.advance()
        self._read_robot_state()
        self.foot_contact_tracker.update()
        if self.jump_enabled:
            self.jump_command.count_takeoffs(self.foot_contact_tracker.takeoff)

        # check termination before the rewards, so that the termination reward can see it; reset happens below
        self._check_termination()
        reward = self.rewards.compute()

        if self.jump_enabled:
            # [解説] 報酬を計算し終えてから、時間切れ（1.0 秒）のジャンプを終わらせる。
            # [解説] そのあとの引き直しで、同じステップのうちに次のジャンプが始まることもある。
            self.jump_command.end_finished_jumps()
        resampling_interval = int(self.environment_config["resampling_time_s"] / STEP_SECONDS)
        self._resample_commands(self.episode_length_buf % resampling_interval == 0)

        self._reset_environments(self.reset_buffer)
        self._update_observation()
        self.last_actions.copy_(self.actions)
        return self.get_observations(), reward, self.reset_buffer, self.extras

    def _apply_actions(self, actions):
        clip_limit = self.environment_config["clip_actions"]
        self.actions = torch.clip(actions, -clip_limit, clip_limit)
        executed_actions = self.actions
        if SIMULATE_ACTION_LATENCY:
            executed_actions = self.last_actions
        target_joint_positions = (
            executed_actions * self.environment_config["action_scale"] + self.default_joint_positions
        )
        self.robot.control_dofs_position(target_joint_positions[:, self.action_order_for_simulator], slice(6, 18))

    def _read_robot_state(self):
        self.base_position = self.robot.get_pos()
        self.base_quaternion = self.robot.get_quat()
        relative_quaternion = transform_quat_by_quat(self.inverse_initial_base_quaternion, self.base_quaternion)
        self.base_euler_angles = quat_to_xyz(relative_quaternion, rpy=True, degrees=True)
        inverse_base_quaternion = inv_quat(self.base_quaternion)
        self.base_linear_velocity = transform_by_quat(self.robot.get_vel(), inverse_base_quaternion)
        self.base_angular_velocity = transform_by_quat(self.robot.get_ang(), inverse_base_quaternion)
        self.projected_gravity = transform_by_quat(self.global_gravity, inverse_base_quaternion)
        self.joint_positions = self.robot.get_dofs_position(self.motor_joint_indices)
        self.joint_velocities = self.robot.get_dofs_velocity(self.motor_joint_indices)

    def _check_termination(self):
        # [解説] 傾き（roll は左右、pitch は前後）が上限を超えるか、エピソードの時間切れで終了する。
        self.time_out_buffer = self.episode_length_buf > self.max_episode_length
        self.reset_buffer = self.time_out_buffer.clone()
        pitch_too_large = (
            torch.abs(self.base_euler_angles[:, 1]) > self.environment_config["termination_if_pitch_greater_than"]
        )
        roll_too_large = (
            torch.abs(self.base_euler_angles[:, 0]) > self.environment_config["termination_if_roll_greater_than"]
        )
        self.reset_buffer |= pitch_too_large
        self.reset_buffer |= roll_too_large
        self.reset_buffer |= self.scene.rigid_solver.get_error_envs_mask()
        # [解説] rsl_rl は extras["time_outs"] を見て、時間切れの終了を転倒と区別する。
        self.extras["time_outs"] = self.time_out_buffer.to(dtype=genesis.tc_float)

    def get_observations(self):
        return TensorDict({"policy": self.observation_buffer}, batch_size=[self.num_envs])

    def _reset_environments(self, reset_mask):
        self.robot.set_qpos(self.initial_qpos, envs_idx=reset_mask, zero_velocity=True, skip_forward=True)

        column_mask = reset_mask[:, None]
        torch.where(column_mask, self.initial_base_position, self.base_position, out=self.base_position)
        torch.where(column_mask, self.initial_base_quaternion, self.base_quaternion, out=self.base_quaternion)
        torch.where(column_mask, self.initial_projected_gravity, self.projected_gravity, out=self.projected_gravity)
        torch.where(column_mask, self.initial_joint_positions, self.joint_positions, out=self.joint_positions)
        self.base_linear_velocity.masked_fill_(column_mask, 0.0)
        self.base_angular_velocity.masked_fill_(column_mask, 0.0)
        self.joint_velocities.masked_fill_(column_mask, 0.0)
        self.actions.masked_fill_(column_mask, 0.0)
        self.last_actions.masked_fill_(column_mask, 0.0)
        self.episode_length_buf.masked_fill_(reset_mask, 0)
        self.foot_contact_tracker.reset(reset_mask)
        self.reset_buffer.masked_fill_(reset_mask, True)

        self.extras["episode"] = self.rewards.report_and_clear_episode_sums(
            reset_mask, self.environment_config["episode_length_s"]
        )

        if self.jump_enabled:
            # [解説] リセットされた環境はジャンプ途中でも打ち切る。その直後の引き直しで、確率 jump_prob で新しいジャンプが始まる。
            self.jump_command.end(reset_mask)
        self._resample_commands(reset_mask)

    def _update_observation(self):
        observation_parts = [
            self.base_angular_velocity * self.observation_scales["ang_vel"],  # 3
            self.projected_gravity,  # 3
            self.velocity_commands.values * self.velocity_command_scales,  # 3
        ]
        if self.jump_enabled:
            observation_parts.append(self.jump_command.flag[:, None])  # 1, jump flag
        observation_parts.append(
            (self.joint_positions - self.default_joint_positions) * self.observation_scales["dof_pos"]
        )
        observation_parts.append(self.joint_velocities * self.observation_scales["dof_vel"])  # 12
        observation_parts.append(self.actions)  # 12
        if self.jump_enabled:
            # [解説] ポリシーの入力に「ジャンプの進み具合」を 0〜1 の数で足す。これで観測は 45 次元から 47 次元になる。
            observation_parts.append(self.jump_command.phase()[:, None])  # 1, jump phase in [0, 1)
        self.observation_buffer = torch.concatenate(observation_parts, dim=-1)

    def reset(self):
        self._reset_environments(self._all_environments_mask())
        self._update_observation()
        return self.get_observations()
