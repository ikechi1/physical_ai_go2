import math

import torch
from tensordict import TensorDict

import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat


def gs_rand(lower, upper, batch_shape):
    assert lower.shape == upper.shape
    return (upper - lower) * torch.rand(size=(*batch_shape, *lower.shape), dtype=gs.tc_float, device=gs.device) + lower


class Go2Env:
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False):
        self.num_envs: int = num_envs
        self.num_actions = env_cfg["num_actions"]
        self.cfg = env_cfg
        self.num_commands = command_cfg["num_commands"]
        self.device = gs.device

        self.simulate_action_latency = True  # there is a 1 step latency on real robot
        self.dt = 0.02  # control frequency on real robot is 50hz
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg

        self.obs_scales: dict[str, float] = obs_cfg["obs_scales"]
        self.reward_scales: dict[str, float] = reward_cfg["reward_scales"]

        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=self.dt,
                substeps=2,
            ),
            rigid_options=gs.options.RigidOptions(
                enable_self_collision=False,
                # For this locomotion policy, there are usually no more than 20 collision pairs. Setting a low value
                # can save memory. Violating this condition will raise an exception.
                max_collision_pairs=20,
                tolerance=1e-5,
            ),
            vis_options=gs.options.VisOptions(
                rendered_envs_idx=[0],
            ),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            show_viewer=show_viewer,
        )

        # add plain
        self.scene.add_entity(
            gs.morphs.URDF(
                file="urdf/plane/plane.urdf",
                fixed=True,
            )
        )

        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file="urdf/go2/urdf/go2.urdf",
                pos=self.env_cfg["base_init_pos"],
                quat=self.env_cfg["base_init_quat"],
            ),
        )

        self.scene.build(n_envs=num_envs)

        # names to indices
        self.motors_dof_idx = torch.tensor(
            [self.robot.get_joint(name).dof_start for name in self.env_cfg["joint_names"]],
            dtype=gs.tc_int,
            device=gs.device,
        )
        self.actions_dof_idx = torch.argsort(self.motors_dof_idx)

        # PD control parameters
        self.robot.set_dofs_kp([self.env_cfg["kp"]] * self.num_actions, self.motors_dof_idx)
        self.robot.set_dofs_kv([self.env_cfg["kd"]] * self.num_actions, self.motors_dof_idx)

        # Define global gravity direction vector
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], dtype=gs.tc_float, device=gs.device)

        # Initial state
        self.init_base_pos = torch.tensor(self.env_cfg["base_init_pos"], dtype=gs.tc_float, device=gs.device)
        self.init_base_quat = torch.tensor(self.env_cfg["base_init_quat"], dtype=gs.tc_float, device=gs.device)
        self.inv_base_init_quat = inv_quat(self.init_base_quat)
        self.init_dof_pos = torch.tensor(
            [self.env_cfg["default_joint_angles"][joint.name] for joint in self.robot.joints[1:]],
            dtype=gs.tc_float,
            device=gs.device,
        )
        self.init_qpos = torch.concatenate((self.init_base_pos, self.init_base_quat, self.init_dof_pos))
        self.init_projected_gravity = transform_by_quat(self.global_gravity, self.inv_base_init_quat)

        # initialize buffers
        self.base_lin_vel = torch.empty((self.num_envs, 3), dtype=gs.tc_float, device=gs.device)
        self.base_ang_vel = torch.empty((self.num_envs, 3), dtype=gs.tc_float, device=gs.device)
        self.projected_gravity = torch.empty((self.num_envs, 3), dtype=gs.tc_float, device=gs.device)
        self.rew_buf = torch.empty((self.num_envs,), dtype=gs.tc_float, device=gs.device)
        self.reset_buf = torch.ones((self.num_envs,), dtype=gs.tc_bool, device=gs.device)
        self.episode_length_buf = torch.empty((self.num_envs,), dtype=gs.tc_int, device=gs.device)
        self.commands = torch.empty((self.num_envs, self.num_commands), dtype=gs.tc_float, device=gs.device)
        # a 4th command is the jump flag (0 or 1); old 3-command configs keep working without it
        # [解説] 指令の数が4つなら「跳べフラグあり」とみなすスイッチ。
        # [解説] 既存の logs/go2-walking などは指令が3つで保存されているので False になり、今まで通り動く。
        self.jump_enabled = self.num_commands == 4
        self.commands_scale = torch.tensor(
            [self.obs_scales["lin_vel"], self.obs_scales["lin_vel"], self.obs_scales["ang_vel"]]
            # [解説] 観測に入れるときの指令の倍率。跳べフラグ（0 か 1）はそのまま入れたいので 1.0。
            # [解説] True * [1.0] = [1.0]、False * [1.0] = [] なので、フラグなしのときは何も足されない。
            + [1.0] * self.jump_enabled,
            device=gs.device,
            dtype=gs.tc_float,
        )
        self.commands_limits: tuple[torch.Tensor, torch.Tensor] = tuple(
            torch.tensor(values, dtype=gs.tc_float, device=gs.device)
            for values in zip(
                self.command_cfg["lin_vel_x_range"],
                self.command_cfg["lin_vel_y_range"],
                self.command_cfg["ang_vel_range"],
            )
        )
        self.actions = torch.zeros((self.num_envs, self.num_actions), dtype=gs.tc_float, device=gs.device)
        self.last_actions = torch.zeros_like(self.actions)
        self.dof_pos = torch.empty_like(self.actions)
        self.dof_vel = torch.empty_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.actions)
        self.base_pos = torch.empty((self.num_envs, 3), dtype=gs.tc_float, device=gs.device)
        self.base_quat = torch.empty((self.num_envs, 4), dtype=gs.tc_float, device=gs.device)
        self.base_euler = torch.empty((self.num_envs, 3), dtype=gs.tc_float, device=gs.device)
        self.default_dof_pos = torch.tensor(
            [self.env_cfg["default_joint_angles"][name] for name in self.env_cfg["joint_names"]],
            dtype=gs.tc_float,
            device=gs.device,
        )
        # steps since the jump flag was raised (0 while not jumping); the flag drops after jump_duration_s
        # [解説] 環境ごとの「フラグが立ってから何ステップたったか」のカウンタ（1ステップ = 0.02 秒）。
        # [解説] episode_length_buf（エピソードの経過ステップ）と同じ作り。
        self.jump_length_buf = torch.zeros((self.num_envs,), dtype=gs.tc_int, device=gs.device)
        if self.jump_enabled:
            # [解説] ジャンプ1回の長さをステップ数に直したもの。1.0 秒 / 0.02 秒 = 50 ステップ。
            self.max_jump_length = math.ceil(command_cfg["jump_duration_s"] / self.dt)
        self.extras = dict()  # extra information for logging
        self.manual_commands = False  # if True, commands are set externally (teleop) and never resampled

        # prepare reward functions and multiply reward scales by dt
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), dtype=gs.tc_float, device=gs.device)

        self.reset()

    def set_commands(self, lin_vel_x, lin_vel_y, ang_vel):
        self.manual_commands = True
        # [解説] キーボードからの速度指令は 0〜2 列目だけに書く。3 列目（跳べフラグ）は jump() で別に管理するので上書きしない。
        self.commands[:, :3] = torch.tensor([lin_vel_x, lin_vel_y, ang_vel], dtype=gs.tc_float, device=gs.device)

    # [解説] teleop の j キーから呼ばれる。全環境（teleop では1つだけ）でジャンプを開始する。
    def jump(self):
        # Raise the jump flag (teleop); ignored while a jump is already in progress
        self._start_jump(self._all_envs())

    @property
    # [解説] 跳べフラグが立っているかを True / False で返す。報酬関数で「ジャンプ中だけ効かせる」ときに掛け算に使える。
    # [解説] @property なので self.jumping と括弧なしで読める。
    def jumping(self):
        return self.commands[:, 3] > 0

    @property
    # [解説] フラグが立ってからの秒数。報酬関数で「踏み切りは最初の 0.3 秒」のように時間で区切るときに使う。
    def jump_time(self):
        # Seconds since the jump flag was raised (0 while not jumping)
        return self.jump_length_buf * self.dt

    # [解説] 全環境ぶんの True の配列を作るだけの補助関数。
    def _all_envs(self):
        # explicit torch.bool: gs.tc_bool is int32 on some backends (Metal), which masks do not accept
        return torch.ones((self.num_envs,), dtype=torch.bool, device=gs.device)

    # [解説] envs_idx が True の環境でジャンプを始める（フラグを 1 にし、カウンタを 0 に戻す）。
    def _start_jump(self, envs_idx):
        # [解説] すでにジャンプ中の環境は除外する。途中でカウンタが 0 に戻って、同じジャンプが延長されないようにするため。
        envs_idx = envs_idx & ~self.jumping
        self.commands[:, 3].masked_fill_(envs_idx, 1.0)
        self.jump_length_buf.masked_fill_(envs_idx, 0)

    # [解説] envs_idx が True の環境でジャンプを終える（フラグを 0 にし、カウンタを 0 に戻す）。
    def _end_jump(self, envs_idx):
        self.commands[:, 3].masked_fill_(envs_idx, 0.0)
        self.jump_length_buf.masked_fill_(envs_idx, 0)

    def _resample_commands(self, envs_idx):
        if self.manual_commands:
            return
        commands = gs_rand(*self.commands_limits, (self.num_envs,))
        if envs_idx is None:
            # [解説] 速度指令（0〜2 列目）だけをランダムに引き直す。跳べフラグは下で別に決める。
            self.commands[:, :3] = commands
        else:
            # [解説] envs_idx が True の環境だけ新しい速度指令に置き換え、それ以外は今の値のまま。
            self.commands[:, :3] = torch.where(envs_idx[:, None], commands, self.commands[:, :3])
        if self.jump_enabled:
            # each resampled env starts a jump with probability jump_prob
            # [解説] 学習中のジャンプ指令の出し方。0〜1 の乱数が jump_prob 未満の環境でジャンプを始める。
            # [解説] 引き直しは4秒ごと（とリセット時）なので、ジャンプ指令もそのタイミングでしか出ない。
            start = torch.rand((self.num_envs,), device=gs.device) < self.command_cfg["jump_prob"]
            self._start_jump(start if envs_idx is None else start & envs_idx)

    def step(self, actions):
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        exec_actions = self.last_actions if self.simulate_action_latency else self.actions
        target_dof_pos = exec_actions * self.env_cfg["action_scale"] + self.default_dof_pos
        self.robot.control_dofs_position(target_dof_pos[:, self.actions_dof_idx], slice(6, 18))
        self.scene.step()

        # update buffers
        self.episode_length_buf += 1
        if self.jump_enabled:
            # [解説] ジャンプ中の環境だけカウンタを 1 進める（True は 1、False は 0 として足される）。
            # [解説] 報酬計算より前に進めるので、報酬関数から見た jump_time は踏み切り直後のステップで 0.02 秒になる。
            self.jump_length_buf += self.jumping.to(gs.tc_int)
        self.base_pos = self.robot.get_pos()
        self.base_quat = self.robot.get_quat()
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(self.inv_base_init_quat, self.base_quat), rpy=True, degrees=True
        )
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel = transform_by_quat(self.robot.get_vel(), inv_base_quat)
        self.base_ang_vel = transform_by_quat(self.robot.get_ang(), inv_base_quat)
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat)
        self.dof_pos = self.robot.get_dofs_position(self.motors_dof_idx)
        self.dof_vel = self.robot.get_dofs_velocity(self.motors_dof_idx)

        # compute reward
        self.rew_buf.zero_()
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        if self.jump_enabled:
            # [解説] 報酬を計算し終えてから、時間切れ（1.0 秒）のジャンプを終わらせる。
            # [解説] そのあとの引き直しで、同じステップのうちに次のジャンプが始まることもある。
            self._end_jump(self.jump_length_buf >= self.max_jump_length)
        self._resample_commands(self.episode_length_buf % int(self.env_cfg["resampling_time_s"] / self.dt) == 0)

        # check termination and reset
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"]
        self.reset_buf |= torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"]
        self.reset_buf |= self.scene.rigid_solver.get_error_envs_mask()

        # Compute timeout
        self.extras["time_outs"] = (self.episode_length_buf > self.max_episode_length).to(dtype=gs.tc_float)

        # Reset environment if necessary
        self._reset_idx(self.reset_buf)

        # update observations
        self._update_observation()

        self.last_actions.copy_(self.actions)
        self.last_dof_vel.copy_(self.dof_vel)

        return self.get_observations(), self.rew_buf, self.reset_buf, self.extras

    def get_observations(self):
        return TensorDict({"policy": self.obs_buf}, batch_size=[self.num_envs])

    def _reset_idx(self, envs_idx=None):
        # reset state
        self.robot.set_qpos(self.init_qpos, envs_idx=envs_idx, zero_velocity=True, skip_forward=True)

        # reset buffers
        if envs_idx is None:
            self.base_pos.copy_(self.init_base_pos)
            self.base_quat.copy_(self.init_base_quat)
            self.projected_gravity.copy_(self.init_projected_gravity)
            self.dof_pos.copy_(self.init_dof_pos)
            self.base_lin_vel.zero_()
            self.base_ang_vel.zero_()
            self.dof_vel.zero_()
            self.actions.zero_()
            self.last_actions.zero_()
            self.last_dof_vel.zero_()
            self.episode_length_buf.zero_()
            self.reset_buf.fill_(True)
        else:
            torch.where(envs_idx[:, None], self.init_base_pos, self.base_pos, out=self.base_pos)
            torch.where(envs_idx[:, None], self.init_base_quat, self.base_quat, out=self.base_quat)
            torch.where(
                envs_idx[:, None], self.init_projected_gravity, self.projected_gravity, out=self.projected_gravity
            )
            torch.where(envs_idx[:, None], self.init_dof_pos, self.dof_pos, out=self.dof_pos)
            self.base_lin_vel.masked_fill_(envs_idx[:, None], 0.0)
            self.base_ang_vel.masked_fill_(envs_idx[:, None], 0.0)
            self.dof_vel.masked_fill_(envs_idx[:, None], 0.0)
            self.actions.masked_fill_(envs_idx[:, None], 0.0)
            self.last_actions.masked_fill_(envs_idx[:, None], 0.0)
            self.last_dof_vel.masked_fill_(envs_idx[:, None], 0.0)
            self.episode_length_buf.masked_fill_(envs_idx, 0)
            self.reset_buf.masked_fill_(envs_idx, True)

        # fill extras
        n_envs = envs_idx.sum() if envs_idx is not None else self.num_envs
        self.extras["episode"] = {}
        for key, value in self.episode_sums.items():
            if envs_idx is None:
                mean = value.mean()
            else:
                mean = torch.where(n_envs > 0, value[envs_idx].sum() / n_envs, 0.0)
            self.extras["episode"]["rew_" + key] = mean / self.env_cfg["episode_length_s"]
            if envs_idx is None:
                value.zero_()
            else:
                value.masked_fill_(envs_idx, 0.0)

        # random sample command upon reset
        if self.jump_enabled:
            # [解説] リセットされた環境はジャンプ途中でも打ち切る。その直後の引き直しで、確率 jump_prob で新しいジャンプが始まる。
            self._end_jump(self._all_envs() if envs_idx is None else envs_idx)
        self._resample_commands(envs_idx)

    def _update_observation(self):
        obs = [
            self.base_ang_vel * self.obs_scales["ang_vel"],  # 3
            self.projected_gravity,  # 3
            self.commands * self.commands_scale,  # 3 (+1 jump flag)
            (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],  # 12
            self.dof_vel * self.obs_scales["dof_vel"],  # 12
            self.actions,  # 12
        ]
        if self.jump_enabled:
            # [解説] ポリシーの入力に「ジャンプの進み具合」を 0〜1 の数で足す（0.5 なら 0.5 秒経過）。
            # [解説] 跳べフラグは上の commands に入っているので、ポリシーには「跳べ」と「今どの段階か」の両方が伝わる。
            # [解説] これで観測は 45 次元から 47 次元になる。
            obs.append((self.jump_length_buf / self.max_jump_length)[:, None])  # 1, jump phase in [0, 1)
        self.obs_buf = torch.concatenate(obs, dim=-1)

    def reset(self):
        self._reset_idx()
        self._update_observation()
        return self.get_observations()

    # Reward functions.
    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_similar_to_default(self):
        # Penalize joint poses far away from default pose
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        return torch.square(self.base_pos[:, 2] - self.reward_cfg["base_height_target"])
