import math

import torch

import genesis


# [解説] 「跳べフラグ」の状態を環境ごとに持つクラス。
# [解説] フラグは立ててから duration_seconds（1.0 秒）たつと自動で下がる。その間がジャンプ 1 回ぶん。
class JumpCommand:
    def __init__(self, environment_count, duration_seconds, step_seconds):
        # [解説] フラグ（0.0 か 1.0）。観測にはこの値がそのまま入る。
        self.flag = torch.zeros((environment_count,), dtype=genesis.tc_float, device=genesis.device)
        # steps since the jump flag was raised (0 while not jumping); the flag drops after duration_steps
        # [解説] 環境ごとの「フラグが立ってから何ステップたったか」のカウンタ（1 ステップ = 0.02 秒）。
        self.elapsed_steps = torch.zeros((environment_count,), dtype=genesis.tc_int, device=genesis.device)
        # takeoffs since the jump flag was raised (0 while not jumping)
        # [解説] 跳べフラグが立ってから何回跳び上がったか（4 本とも浮いた瞬間の回数）を数える。
        self.takeoff_count = torch.zeros((environment_count,), dtype=genesis.tc_int, device=genesis.device)
        # [解説] ジャンプ 1 回の長さをステップ数に直したもの。1.0 秒 / 0.02 秒 = 50 ステップ。
        self.duration_steps = math.ceil(duration_seconds / step_seconds)
        self.step_seconds = step_seconds

    # [解説] 跳べフラグが立っているかを True / False で返す。報酬関数で「ジャンプ中だけ効かせる」ときに掛け算に使える。
    def is_jumping(self):
        return self.flag > 0

    # [解説] フラグが立ってからの秒数。報酬関数で「踏み切りは最初の 0.5 秒」のように時間で区切るときに使う。
    def elapsed_seconds(self):
        return self.elapsed_steps * self.step_seconds

    # [解説] ジャンプの進み具合を 0〜1 の数で返す（0.5 なら 0.5 秒経過）。観測に入れる。
    def phase(self):
        return self.elapsed_steps / self.duration_steps

    # [解説] フラグを下げる直前のステップ（ジャンプ 1 回ぶんの時間を使い切った）かどうか。
    def is_last_step(self):
        return self.is_jumping() & (self.elapsed_steps >= self.duration_steps)

    # [解説] start_mask が True の環境でジャンプを始める（フラグを 1 にし、カウンタを 0 に戻す）。
    def start(self, start_mask):
        # [解説] すでにジャンプ中の環境は除外する。途中でカウンタが 0 に戻って、同じジャンプが延長されないようにするため。
        new_jump_mask = start_mask & ~self.is_jumping()
        self.flag.masked_fill_(new_jump_mask, 1.0)
        self.elapsed_steps.masked_fill_(new_jump_mask, 0)
        self.takeoff_count.masked_fill_(new_jump_mask, 0)

    # [解説] end_mask が True の環境でジャンプを終える（フラグを 0 にし、カウンタを 0 に戻す）。
    def end(self, end_mask):
        self.flag.masked_fill_(end_mask, 0.0)
        self.elapsed_steps.masked_fill_(end_mask, 0)
        self.takeoff_count.masked_fill_(end_mask, 0)

    # [解説] ジャンプの時間を使い切った環境のジャンプを終える。
    def end_finished_jumps(self):
        self.end(self.elapsed_steps >= self.duration_steps)

    # [解説] ジャンプ中の環境だけカウンタを 1 進める（True は 1、False は 0 として足される）。
    def advance(self):
        self.elapsed_steps += self.is_jumping().to(genesis.tc_int)

    # [解説] ジャンプ中に跳び上がった環境の回数を 1 増やす。
    def count_takeoffs(self, takeoff_mask):
        self.takeoff_count += (takeoff_mask & self.is_jumping()).to(genesis.tc_int)
