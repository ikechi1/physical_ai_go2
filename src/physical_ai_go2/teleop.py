"""
Drive the Go2 in the Genesis viewer with the keyboard, using a policy trained by train_walk.py / train_jump.py / train_walk_jump.py.

Keyboard Controls (click the viewer window first):
Up / Down     - Walk forward / backward
Left / Right  - Turn left / right
q / e         - Side step left / right
j             - Jump (policies trained with the jump flag only)
space         - Stop
u             - Reset robot
esc           - Quit

Plus all default viewer controls (press 'i' to see them)
"""

import argparse
import glob
import os
import pickle
import re

import torch
from rsl_rl.runners import OnPolicyRunner

import genesis as gs
from genesis.vis.keybindings import Key, KeyAction, Keybind

from physical_ai_go2.go2_env import Go2Env

# commands sent while a key is held (kept inside the trained command ranges)
LIN_VEL_X_MAX, LIN_VEL_X_MIN = 1.0, -0.5
LIN_VEL_Y_MAX = 0.5
ANG_VEL_MAX = 1.0


def latest_ckpt(log_dir):
    ckpts = glob.glob(os.path.join(log_dir, "model_*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint in {log_dir}. Train first with: uv run go2-train-walk (or go2-train-jump / go2-train-walk-jump)")
    return max(ckpts, key=lambda p: int(re.search(r"model_(\d+)\.pt", p).group(1)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e", "--exp-name", type=str, default="go2-walking", help="Experiment name; also the log directory under logs/"
    )
    parser.add_argument("--ckpt", type=int, default=None, help="Checkpoint iteration to load (default: latest)")
    args = parser.parse_args()

    gs.init(backend=gs.cpu, logging_level="warning")

    log_dir = f"logs/{args.exp_name}"
    with open(f"{log_dir}/cfgs.pkl", "rb") as f:
        env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(f)
    reward_cfg["reward_scales"] = {}
    env_cfg["episode_length_s"] = 1e6  # no time-out reset while driving

    env = Go2Env(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=True,
    )
    env.scene.viewer.follow_entity(env.robot, smoothing=0.1)

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    ckpt = os.path.join(log_dir, f"model_{args.ckpt}.pt") if args.ckpt is not None else latest_ckpt(log_dir)
    runner.load(ckpt)
    policy = runner.get_inference_policy(device=gs.device)
    print(f"Loaded {ckpt}")
    print(__doc__)

    # keys currently held down
    held = {name: False for name in ("fwd", "back", "left", "right", "side_l", "side_r")}
    state = {"running": True, "reset": False}

    def set_held(name, value):
        held[name] = value

    def stop():
        for name in held:
            held[name] = False

    # [解説] j キーを押したときに呼ばれる。フラグ付きで学習したポリシーなら env.jump() でフラグを立てる。
    # [解説] フラグは env 側で 1.0 秒後に自動で下がるので、キーを離したときの処理はいらない。
    def jump():
        if env.jump_enabled:
            env.jump()
            print("jump")
        else:
            print("this policy was trained without the jump flag")

    def reset():
        state["reset"] = True

    def quit_():
        state["running"] = False

    keybinds = []
    for name, key in (
        ("fwd", Key.UP),
        ("back", Key.DOWN),
        ("left", Key.LEFT),
        ("right", Key.RIGHT),
        ("side_l", Key.Q),
        ("side_r", Key.E),
    ):
        keybinds.append(Keybind(f"go2_{name}_press", key, KeyAction.PRESS, callback=set_held, args=(name, True)))
        keybinds.append(Keybind(f"go2_{name}_release", key, KeyAction.RELEASE, callback=set_held, args=(name, False)))
    keybinds += [
        # [解説] j キーを押した瞬間（PRESS）に jump() を呼ぶ。矢印キーと違い、押し続けても意味はない。
        Keybind("go2_jump", Key.J, KeyAction.PRESS, callback=jump),
        Keybind("go2_stop", Key.SPACE, KeyAction.PRESS, callback=stop),
        Keybind("go2_reset", Key.U, KeyAction.RELEASE, callback=reset),
        Keybind("go2_quit", Key.ESCAPE, KeyAction.RELEASE, callback=quit_),
    ]
    env.scene.viewer.register_keybinds(*keybinds, overwrite=True)

    obs_dict = env.reset()
    last_cmd = None
    with torch.no_grad():
        while state["running"] and env.scene.viewer.is_alive():
            if state["reset"]:
                state["reset"] = False
                obs_dict = env.reset()

            cmd = (
                LIN_VEL_X_MAX * held["fwd"] + LIN_VEL_X_MIN * held["back"],
                LIN_VEL_Y_MAX * (held["side_l"] - held["side_r"]),
                ANG_VEL_MAX * (held["left"] - held["right"]),
            )
            if cmd != last_cmd:
                print(f"command: vx={cmd[0]:+.2f} m/s  vy={cmd[1]:+.2f} m/s  yaw={cmd[2]:+.2f} rad/s")
                last_cmd = cmd
            env.set_commands(*cmd)

            actions = policy(obs_dict)
            obs_dict, _, _, _ = env.step(actions)


if __name__ == "__main__":
    main()
