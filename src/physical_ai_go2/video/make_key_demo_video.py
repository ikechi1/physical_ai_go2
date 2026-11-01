"""
Render a 30-second walking demo that shows which key is being pressed.

uv run go2-make-key-demo-video        # videos/go2_walk_keyboard.mp4
"""

import argparse
import os

import imageio.v2
import numpy
import torch

import genesis

from physical_ai_go2.control.keyboard_controller import KeyboardController
from physical_ai_go2.control.policy_loading import LoadedPolicy
from physical_ai_go2.video.key_demo_timeline import DEMO_SECONDS, find_pressed_key
from physical_ai_go2.video.key_input_overlay import KeyInputOverlayDrawer

FRAMES_PER_SECOND = 50  # one frame per control step (0.02 s), so 1x speed
VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720
# [解説] カメラはロボットの後ろ少し右上に置き、向きは固定する。キーの左右と画面の左右がおおむね一致する。
CAMERA_BEHIND_DISTANCE = 2.6
CAMERA_RIGHT_DISTANCE = 0.9
CAMERA_HEIGHT = 1.5
# [解説] 字幕のパネルでロボットが隠れないよう、ロボットより低い点を見て、ロボットを画面の上寄りに映す。
CAMERA_TARGET_HEIGHT = -0.05


# [解説] カメラの位置をロボットに合わせて動かし、1 フレーム撮る。
def capture_following_camera(environment):
    robot_x = float(environment.base_position[0, 0])
    robot_y = float(environment.base_position[0, 1])
    environment.camera.set_pose(
        pos=(robot_x - CAMERA_BEHIND_DISTANCE, robot_y - CAMERA_RIGHT_DISTANCE, CAMERA_HEIGHT),
        lookat=(robot_x, robot_y, CAMERA_TARGET_HEIGHT),
    )
    return numpy.asarray(environment.camera.render()[0])


# [解説] キーごとに、指令への追従を確かめるための実測値（ロボットから見た速度）を足し込む。
def add_measured_velocity(measured_totals, pressed_key, environment):
    if pressed_key not in measured_totals:
        measured_totals[pressed_key] = {"forward": 0.0, "sideways": 0.0, "turning": 0.0, "count": 0}
    totals = measured_totals[pressed_key]
    totals["forward"] += float(environment.base_linear_velocity[0, 0])
    totals["sideways"] += float(environment.base_linear_velocity[0, 1])
    totals["turning"] += float(environment.base_angular_velocity[0, 2])
    totals["count"] += 1


def print_measured_velocities(measured_totals):
    for key_name, totals in measured_totals.items():
        print(
            "{}: forward {:+.2f} m/s  sideways {:+.2f} m/s  turning {:+.2f} rad/s".format(
                key_name,
                totals["forward"] / totals["count"],
                totals["sideways"] / totals["count"],
                totals["turning"] / totals["count"],
            )
        )


# [解説] 台本どおりにキーを押した状態にして、teleop と同じ KeyboardController で速度指令を作って撮影する。
def record_demo(experiment_name, writer):
    loaded_policy = LoadedPolicy(experiment_name, 1, camera_resolution=(VIDEO_WIDTH, VIDEO_HEIGHT))
    environment = loaded_policy.environment
    controller = KeyboardController(environment)
    overlay_drawer = KeyInputOverlayDrawer(VIDEO_WIDTH, VIDEO_HEIGHT)
    environment.set_velocity_commands(0.0, 0.0, 0.0)
    observation = environment.reset()

    measured_totals = {}
    reset_count = 0
    with torch.no_grad():
        for step_number in range(int(DEMO_SECONDS * FRAMES_PER_SECOND)):
            pressed_key = find_pressed_key(step_number / FRAMES_PER_SECOND)
            controller.stop()
            if pressed_key is not None:
                controller.set_held(pressed_key, True)
            velocity_command = controller.velocity_command()
            environment.set_velocity_commands(velocity_command[0], velocity_command[1], velocity_command[2])

            observation, rewards, dones, extras = environment.step(loaded_policy.policy(observation))
            if bool(dones[0]):
                reset_count += 1
            add_measured_velocity(measured_totals, pressed_key, environment)

            camera_frame = capture_following_camera(environment)
            writer.append_data(overlay_drawer.draw(camera_frame, pressed_key, velocity_command))
    print_measured_velocities(measured_totals)
    print("resets during the demo: " + str(reset_count))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp-name", type=str, default="go2-walking", help="Experiment name under logs/")
    parser.add_argument("--out", type=str, default="videos/go2_walk_keyboard.mp4")
    arguments = parser.parse_args()

    genesis.init(backend=genesis.cpu, logging_level="warning", seed=0)
    os.makedirs(os.path.dirname(arguments.out), exist_ok=True)
    writer = imageio.v2.get_writer(arguments.out, fps=FRAMES_PER_SECOND, codec="libx264", quality=8, macro_block_size=8)
    record_demo(arguments.exp_name, writer)
    writer.close()
    print("saved " + arguments.out)


if __name__ == "__main__":
    main()
