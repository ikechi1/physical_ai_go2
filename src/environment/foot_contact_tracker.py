import torch

import genesis

# [解説] 接触力（上向き）がこの値 [N] を超えたら、床に触れているとみなす。
CONTACT_FORCE_THRESHOLD = 1.0


# [解説] 足と胴体が床に触れているかを毎ステップ調べ、滞空（4 本とも浮いている状態）を記録するクラス。
class FootContactTracker:
    def __init__(self, robot, environment_count, step_seconds):
        self.robot = robot
        self.step_seconds = step_seconds
        # the feet are part of the calf links in this URDF
        # [解説] Go2 の URDF では足先が calf（すね）のリンクに含まれているので、calf の接触力で接地を判定する。
        self.foot_link_indices = []
        for leg_name in ["FR", "FL", "RR", "RL"]:
            self.foot_link_indices.append(robot.get_link(leg_name + "_calf").idx_local)
        self.base_link_index = robot.get_link("base").idx_local
        # [解説] 足ごとの接地（環境数 × 4）。
        self.foot_contact = torch.zeros((environment_count, 4), dtype=torch.bool, device=genesis.device)
        # [解説] 胴体（base リンク）が床に触れているか。
        self.base_contact = torch.zeros((environment_count,), dtype=torch.bool, device=genesis.device)
        # [解説] 4 本とも浮いているか。
        self.airborne = torch.zeros((environment_count,), dtype=torch.bool, device=genesis.device)
        # [解説] 前のステップは接地、今のステップは空中 → 跳び上がった瞬間。
        self.takeoff = torch.zeros((environment_count,), dtype=torch.bool, device=genesis.device)
        # steps all four feet have been off the ground in a row
        # [解説] 連続して空中にいるステップ数。着地したら 0 に戻る。
        self.airborne_steps = torch.zeros((environment_count,), dtype=genesis.tc_int, device=genesis.device)

    # [解説] シミュレーションを 1 ステップ進めたあとに呼ぶ。接触力から接地・滞空・跳び上がりを更新する。
    def update(self):
        contact_forces = self.robot.get_links_net_contact_force()
        self.foot_contact = contact_forces[:, self.foot_link_indices, 2] > CONTACT_FORCE_THRESHOLD
        self.base_contact = contact_forces[:, self.base_link_index, 2] > CONTACT_FORCE_THRESHOLD
        airborne_now = ~self.foot_contact.any(dim=1)
        self.takeoff = airborne_now & ~self.airborne
        self.airborne = airborne_now
        self.airborne_steps = torch.where(self.airborne, self.airborne_steps + 1, 0)

    # [解説] 連続して空中にいる秒数（どれかの足が触れている間は 0）。
    def airborne_seconds(self):
        return self.airborne_steps * self.step_seconds

    # spawned above the ground, so the first fall is not counted as a takeoff
    # [解説] リセットした環境は空中から始まるので airborne を True にしておく。開始時の落下を「跳び上がった」と数えないため。
    def reset(self, reset_mask):
        self.airborne.masked_fill_(reset_mask, True)
        self.airborne_steps.masked_fill_(reset_mask, 0)
