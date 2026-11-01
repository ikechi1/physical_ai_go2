"""
Render the submission video: each scene runs a trained policy with scripted commands and an offscreen camera.

uv run go2-make-video                  # videos/go2_walk_jump.mp4
uv run go2-make-video --only s7        # render one scene to videos/s7.mp4 (for checking)
"""

import argparse
import glob
import os
import pickle
import re

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from rsl_rl.runners import OnPolicyRunner

import genesis as gs

from physical_ai_go2.go2_env import Go2Env

FPS = 50  # one frame per control step (dt = 0.02 s), so 1x speed
RES = (1280, 720)
FONT = "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc"

# [解説] 動画の構成。card は文字だけの画面、scene は学習済みポリシーを動かして撮る場面。
# [解説] events は (秒, 動作) の並び。("cmd", vx, vy, yaw) で速度指令、("jump",) で跳べフラグを立てる。
# [解説] slow が 4 なら 1 フレームを 4 回書き、1/4 倍速のスローにする。
SCENES = [
    {"id": "c0", "card": ["Unitree Go2 の歩行とジャンプ", "強化学習（Genesis + PPO）で「報酬の仮説」を試した記録"], "sec": 5},
    {
        "id": "s1",
        "exp": "go2-walking",
        "sec": 12,
        "events": [(1, "cmd", 0.8, 0, 0), (5, "cmd", 0.5, 0, 0.8), (9, "cmd", -0.4, 0, 0)],
        "caption": "① 歩行：速度指令に追従（前進 0.8m/s → 旋回しながら前進 → 後退）",
    },
    {"id": "c1", "card": ["② その場ジャンプ", "「跳べ」フラグ（1 秒間）で跳ぶ報酬を、仮説を立てて 20 回試した"], "sec": 4},
    {
        "id": "s2",
        "exp": "go2-jump",
        "sec": 5,
        "events": [(1, "jump")],
        "caption": "仮説1：フラグ中は体高の目標を +15cm → 背伸びするだけで跳ばない",
    },
    {
        "id": "s3",
        "exp": "go2-jump-upvel-noheight",
        "sec": 6,
        "events": [(1, "jump")],
        "caption": "仮説4：上向きの速度に報酬 → 高く跳ぶ（0.77m）が、着地で傾いて転倒",
    },
    {
        "id": "s4",
        "exp": "go2-jump-upvel-capped",
        "sec": 5,
        "events": [(1, "jump")],
        "caption": "仮説5：報酬に時間と高さの制限 → 何度も跳ねて報酬を稼ぐ（約 3.7 回）",
    },
    {
        "id": "s5",
        "exp": "go2-jump-crouch",
        "sec": 5,
        "events": [(1, "jump")],
        "caption": "仮説8：踏み切りの窓を 0.5 秒に → 胴体から床に落とし、反動で上向きの速度を稼ぐ",
    },
    {
        "id": "s6",
        "exp": "go2-jump-airfirst",
        "sec": 5,
        "events": [(1, "jump")],
        "caption": "仮説18：滞空の報酬は 1 回目だけ → 本ジャンプのあと着地で小さく跳ねる",
    },
    {
        "id": "s7",
        "exp": "go2-jump-stick-500",
        "sec": 13,
        "events": [(1, "jump"), (5, "jump"), (9, "jump")],
        "caption": "仮説20：1 回だけ跳んで着地を保つと報酬 → 1 回だけ跳ぶ（最高 約 0.49m、転倒 0）",
    },
    {
        "id": "s8",
        "exp": "go2-jump-stick-500",
        "sec": 1.6,
        "events": [(0.8, "jump")],
        "skip": 0.8,
        "slow": 4,
        "caption": "仮説20：スロー（1/4 倍速）",
    },
    {"id": "c2", "card": ["③ 歩行 ＋ ジャンプ", "歩きながら、フラグが立ったら跳ぶ"], "sec": 4},
    {
        "id": "s9",
        "exp": "go2-walk-jump-w2",
        "sec": 5,
        "events": [(0.5, "cmd", 0.5, 0, 0)],
        "caption": "W2：転倒に罰 → 転ぶのを避けて、指令 0.5m/s でも立ち止まる",
    },
    {
        "id": "s10",
        "exp": "go2-walk-jump-w3",
        "sec": 6,
        "events": [(0.5, "cmd", 0.5, 0, 0), (2.5, "jump")],
        "caption": "W3：歩行の学習済み重みから開始 → 歩けるが、フラグが立っても跳ばない",
    },
    {
        "id": "s11",
        "exp": "go2-walk-jump-w4",
        "sec": 12,
        "events": [(0.5, "cmd", 0.5, 0, 0), (2.5, "jump"), (6.5, "jump"), (10.0, "jump")],
        "caption": "W4：ジャンプの練習量を増やす → 歩きながら跳ぶ（1 回のフラグで小さく 3 回）",
    },
    {
        "id": "s12",
        "exp": "go2-walk-jump-w4",
        "sec": 3.6,
        "events": [(0.5, "cmd", 0.5, 0, 0), (2.5, "jump")],
        "skip": 2.4,
        "slow": 4,
        "caption": "W4：スロー（1/4 倍速）",
    },
    {
        "id": "c3",
        "card": [
            "まとめ",
            "・報酬は「抜け穴」を突かれる：回数で稼ぐ、時間をずらす、落下の反動を使う",
            "・仮説 → 学習 → 評価 → 原因の分析 を繰り返し、その場ジャンプは 1 回跳びに",
            "・歩行＋ジャンプは、歩きながら跳ぶところまで（1 回跳び・障害物越えは今後）",
        ],
        "sec": 9,
    },
]


def font(size):
    return ImageFont.truetype(FONT, size)


def card_frames(lines, sec):
    img = Image.new("RGB", RES, (14, 22, 34))
    d = ImageDraw.Draw(img)
    y = 230 if len(lines) <= 2 else 170
    d.text((RES[0] // 2, y), lines[0], font=font(56), fill=(240, 240, 240), anchor="mm")
    for i, line in enumerate(lines[1:]):
        anchor, x = ("mm", RES[0] // 2) if len(lines) <= 2 else ("lm", 140)
        d.text((x, y + 110 + i * 70), line, font=font(32), fill=(190, 205, 225), anchor=anchor)
    frame = np.asarray(img)
    return [frame] * int(sec * FPS)


def overlay(frame, caption, cmd, jumping, slow):
    img = Image.fromarray(frame)
    d = ImageDraw.Draw(img, "RGBA")
    d.rectangle((0, RES[1] - 80, RES[0], RES[1]), fill=(0, 0, 0, 170))
    d.text((RES[0] // 2, RES[1] - 40), caption, font=font(28), fill=(255, 255, 255), anchor="mm")
    speed = "1/4 倍速" if slow == 4 else "1 倍速"
    d.text((24, 20), f"{speed}   指令 vx={cmd[0]:+.1f}m/s  yaw={cmd[2]:+.1f}rad/s", font=font(24), fill=(255, 255, 255))
    if jumping:
        d.rounded_rectangle((RES[0] - 210, 16, RES[0] - 24, 64), radius=10, fill=(230, 80, 40, 230))
        d.text((RES[0] - 117, 40), "跳べフラグ ON", font=font(24), fill=(255, 255, 255), anchor="mm")
    return np.asarray(img)


def latest_ckpt(log_dir):
    return max(glob.glob(f"{log_dir}/model_*.pt"), key=lambda p: int(re.search(r"model_(\d+)\.pt", p).group(1)))


def scene_frames(scene):
    log_dir = f"logs/{scene['exp']}"
    with open(f"{log_dir}/cfgs.pkl", "rb") as f:
        env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(f)
    reward_cfg["reward_scales"] = {}
    env_cfg["episode_length_s"] = 1e6  # only falls / tilt reset the robot
    env = Go2Env(1, env_cfg, obs_cfg, reward_cfg, command_cfg, camera_res=RES)
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    runner.load(latest_ckpt(log_dir))
    policy = runner.get_inference_policy(device=gs.device)

    cmd = (0.0, 0.0, 0.0)
    env.set_commands(*cmd)
    obs = env.reset()
    events = sorted(scene["events"])
    slow = scene.get("slow", 1)
    frames = []
    with torch.no_grad():
        for step in range(int(scene["sec"] * FPS)):
            t = step / FPS
            while events and events[0][0] <= t:
                _, kind, *args = events.pop(0)
                if kind == "cmd":
                    cmd = tuple(args)
                    env.set_commands(*cmd)
                elif env.jump_enabled:
                    env.jump()
            obs, _, _, _ = env.step(policy(obs))
            if t < scene.get("skip", 0):
                continue
            # [解説] カメラはロボットの真横 2m から追いかける。高さは固定して、ジャンプの上下が見えるようにする。
            x, y = env.base_pos[0, :2].tolist()
            env.camera.set_pose(pos=(x, y - 2.0, 0.6), lookat=(x, y, 0.3))
            rgb = np.asarray(env.camera.render()[0])
            frames += [overlay(rgb, scene["caption"], cmd, bool(env.jumping[0]) if env.jump_enabled else False, slow)] * slow
    return frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", type=str, default=None, help="Render one scene id (e.g. s7) to videos/<id>.mp4")
    parser.add_argument("--out", type=str, default="videos/go2_walk_jump.mp4")
    args = parser.parse_args()

    gs.init(backend=gs.cpu, logging_level="warning", seed=0)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    scenes = [s for s in SCENES if args.only in (None, s["id"])]
    out = f"videos/{args.only}.mp4" if args.only else args.out
    writer = imageio.get_writer(out, fps=FPS, codec="libx264", quality=8, macro_block_size=8)
    for scene in scenes:
        frames = card_frames(scene["card"], scene["sec"]) if "card" in scene else scene_frames(scene)
        for frame in frames:
            writer.append_data(frame)
        print(f"{scene['id']}: {len(frames) / FPS:.1f} s")
    writer.close()
    print(f"saved {out}")


if __name__ == "__main__":
    main()
