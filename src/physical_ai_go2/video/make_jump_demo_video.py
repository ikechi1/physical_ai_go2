"""
Render a 30-second jump demo: one key press for each hypothesis whose motion clearly changed.

uv run go2-make-jump-demo-video        # videos/go2_jump_hypotheses.mp4
"""

import argparse
import os

import imageio.v2
import numpy
import torch

import genesis

from physical_ai_go2.control.keyboard_controller import KeyboardController
from physical_ai_go2.control.policy_loading import LoadedPolicy
from physical_ai_go2.video.jump_demo_overlay import JumpDemoOverlayDrawer
from physical_ai_go2.video.jump_demo_timeline import (
    AFTER_SLOW_SECONDS,
    JUMP_DEMO_SCENES,
    KEY_DISPLAY_SECONDS,
    SETTLE_SECONDS,
    SLOW_FACTOR,
    SLOW_SECONDS,
    STAND_SECONDS,
)

STEPS_PER_SECOND = 50  # one control step is 0.02 s
FRAMES_PER_SECOND = 50
VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720
# [解説] カメラはロボットの真横に置く。高さは固定して、ジャンプの上下が床との距離で分かるようにする。
CAMERA_SIDE_DISTANCE = 2.8
CAMERA_HEIGHT = 0.7
# [解説] 最高点（約 0.7m）で上の表示に重ならず、立ち姿が下のパネルに隠れない高さを見る。
CAMERA_TARGET_HEIGHT = 0.2


def capture_side_camera(environment):
    robot_x = float(environment.base_position[0, 0])
    robot_y = float(environment.base_position[0, 1])
    environment.camera.set_pose(
        pos=(robot_x, robot_y - CAMERA_SIDE_DISTANCE, CAMERA_HEIGHT),
        lookat=(robot_x, robot_y, CAMERA_TARGET_HEIGHT),
    )
    return numpy.asarray(environment.camera.render()[0])


# [解説] 開始時は空中から落ちてくるので、撮影の前に立つまで進めておく。
def settle_robot(loaded_policy, observation):
    for step_number in range(int(SETTLE_SECONDS * STEPS_PER_SECOND)):
        observation, rewards, dones, extras = loaded_policy.environment.step(loaded_policy.policy(observation))
    return observation


# [解説] 1 つの仮説を撮影する。j キーを押した扱いにして、teleop と同じ KeyboardController.jump() を呼ぶ。
def record_scene(scene, scene_number, overlay_drawer, writer):
    loaded_policy = LoadedPolicy(scene["experiment"], 1, camera_resolution=(VIDEO_WIDTH, VIDEO_HEIGHT))
    environment = loaded_policy.environment
    controller = KeyboardController(environment)
    environment.set_velocity_commands(0.0, 0.0, 0.0)
    observation = settle_robot(loaded_policy, environment.reset())

    press_step = int(STAND_SECONDS * STEPS_PER_SECOND)
    key_display_end_step = press_step + int(KEY_DISPLAY_SECONDS * STEPS_PER_SECOND)
    slow_end_step = press_step + int(SLOW_SECONDS * STEPS_PER_SECOND)
    total_steps = slow_end_step + int(AFTER_SLOW_SECONDS * STEPS_PER_SECOND)

    max_height = None
    reset_steps = []
    for step_number in range(total_steps):
        if step_number == press_step:
            controller.jump()
        observation, rewards, dones, extras = environment.step(loaded_policy.policy(observation))
        if bool(dones[0]):
            reset_steps.append(step_number - press_step)

        height = float(environment.base_position[0, 2])
        if step_number >= press_step:
            if max_height is None or height > max_height:
                max_height = height
        is_slow = step_number >= press_step and step_number < slow_end_step
        status = {
            "scene_number": scene_number,
            "scene_count": len(JUMP_DEMO_SCENES),
            "is_key_pressed": step_number >= press_step and step_number < key_display_end_step,
            "is_jumping": bool(environment.is_jumping()[0]),
            "is_slow": is_slow,
            "height": height,
            "max_height": max_height,
        }
        frame = overlay_drawer.draw(capture_side_camera(environment), scene, status)
        frame_copies = 1
        if is_slow:
            frame_copies = SLOW_FACTOR
        for copy_number in range(frame_copies):
            writer.append_data(frame)
    print(
        "{}: max height {:.3f} m, resets at steps after the key press {}".format(
            scene["experiment"], max_height, reset_steps
        )
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="videos/go2_jump_hypotheses.mp4")
    arguments = parser.parse_args()

    genesis.init(backend=genesis.cpu, logging_level="warning", seed=0)
    os.makedirs(os.path.dirname(arguments.out), exist_ok=True)
    writer = imageio.v2.get_writer(arguments.out, fps=FRAMES_PER_SECOND, codec="libx264", quality=8, macro_block_size=8)
    overlay_drawer = JumpDemoOverlayDrawer(VIDEO_WIDTH, VIDEO_HEIGHT)
    with torch.no_grad():
        for scene_index in range(len(JUMP_DEMO_SCENES)):
            record_scene(JUMP_DEMO_SCENES[scene_index], scene_index + 1, overlay_drawer, writer)
    writer.close()
    print("saved " + arguments.out)


if __name__ == "__main__":
    main()
