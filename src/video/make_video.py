"""
Render the submission video: each scene runs a trained policy with scripted commands and an offscreen camera.

uv run go2-make-video                  # videos/go2_walk_jump.mp4
uv run go2-make-video --only s7        # render one scene to videos/s7.mp4 (for checking)
"""

import argparse
import os

import imageio.v2
import numpy
import torch

import genesis

from control.policy_loading import LoadedPolicy
from video.video_drawing import VideoFrameDrawer
from video.video_scenes import SCENES

FRAMES_PER_SECOND = 50  # one frame per control step (0.02 s), so 1x speed
VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720
# [解説] カメラはロボットの真横（横方向に 2m 離れた、高さ 0.6m の位置）から、高さ 0.3m の点を見る。
CAMERA_SIDE_DISTANCE = 2.0
CAMERA_HEIGHT = 0.6
CAMERA_TARGET_HEIGHT = 0.3


# [解説] 文字だけの画面を、場面の秒数ぶん並べる。
def record_card(scene, drawer):
    frame = drawer.draw_card(scene["card_lines"])
    frames = []
    for frame_number in range(int(scene["seconds"] * FRAMES_PER_SECOND)):
        frames.append(frame)
    return frames


# [解説] 学習済みポリシーを、場面に書かれた指令とジャンプの時刻どおりに動かして撮影する。
def record_policy_scene(scene, drawer):
    loaded_policy = LoadedPolicy(scene["experiment"], 1, camera_resolution=(VIDEO_WIDTH, VIDEO_HEIGHT))
    environment = loaded_policy.environment
    velocity_command = {"forward": 0.0, "sideways": 0.0, "turning": 0.0}
    environment.set_velocity_commands(0.0, 0.0, 0.0)
    observation = environment.reset()

    pending_commands = list(scene["velocity_commands"])
    pending_jump_times = list(scene["jump_times"])
    frames = []
    with torch.no_grad():
        for step_number in range(int(scene["seconds"] * FRAMES_PER_SECOND)):
            time = step_number / FRAMES_PER_SECOND
            while len(pending_commands) > 0 and pending_commands[0]["time"] <= time:
                velocity_command = pending_commands.pop(0)
                environment.set_velocity_commands(
                    velocity_command["forward"], velocity_command["sideways"], velocity_command["turning"]
                )
            while len(pending_jump_times) > 0 and pending_jump_times[0] <= time:
                pending_jump_times.pop(0)
                if environment.jump_enabled:
                    environment.start_jump_for_all()
            observation, rewards, dones, extras = environment.step(loaded_policy.policy(observation))
            if time < scene["skip_seconds"]:
                continue
            camera_frame = capture_following_camera(environment)
            is_jumping = bool(environment.is_jumping()[0])
            frame = drawer.draw_overlay(
                camera_frame, scene["caption"], velocity_command, is_jumping, scene["slow_factor"]
            )
            for copy_number in range(scene["slow_factor"]):
                frames.append(frame)
    return frames


# [解説] カメラをロボットの真横に動かして撮る。高さは固定して、ジャンプの上下が見えるようにする。
def capture_following_camera(environment):
    robot_x = float(environment.base_position[0, 0])
    robot_y = float(environment.base_position[0, 1])
    environment.camera.set_pose(
        pos=(robot_x, robot_y - CAMERA_SIDE_DISTANCE, CAMERA_HEIGHT),
        lookat=(robot_x, robot_y, CAMERA_TARGET_HEIGHT),
    )
    return numpy.asarray(environment.camera.render()[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", type=str, default=None, help="Render one scene name (e.g. s7) to videos/<name>.mp4")
    parser.add_argument("--out", type=str, default="videos/go2_walk_jump.mp4")
    arguments = parser.parse_args()

    genesis.init(backend=genesis.cpu, logging_level="warning", seed=0)
    output_path = arguments.out
    if arguments.only is not None:
        output_path = "videos/" + arguments.only + ".mp4"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    drawer = VideoFrameDrawer(VIDEO_WIDTH, VIDEO_HEIGHT)
    writer = imageio.v2.get_writer(output_path, fps=FRAMES_PER_SECOND, codec="libx264", quality=8, macro_block_size=8)
    for scene in SCENES:
        if arguments.only is not None and scene["name"] != arguments.only:
            continue
        if "card_lines" in scene:
            frames = record_card(scene, drawer)
        else:
            frames = record_policy_scene(scene, drawer)
        for frame in frames:
            writer.append_data(frame)
        print("{}: {:.1f} s".format(scene["name"], len(frames) / FRAMES_PER_SECOND))
    writer.close()
    print("saved " + output_path)


if __name__ == "__main__":
    main()
