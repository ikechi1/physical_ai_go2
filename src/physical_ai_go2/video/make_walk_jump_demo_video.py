"""
Render the walking + jumping demo: every key is pressed in turn, and the video shows what did not work.

uv run go2-make-walk-jump-demo-video        # videos/go2_walk_jump_keyboard.mp4
"""

import argparse
import os

import imageio.v2
import numpy
import torch

import genesis

from physical_ai_go2.control.keyboard_controller import KeyboardController
from physical_ai_go2.control.policy_loading import LoadedPolicy
from physical_ai_go2.video.segment_motion_tracker import SegmentMotionTracker
from physical_ai_go2.video.walk_jump_demo_overlay import WalkJumpDemoOverlayDrawer
from physical_ai_go2.video.walk_jump_demo_timeline import EXPERIMENT_NAME, SUMMARY_SECONDS, WALK_JUMP_DEMO_SEGMENTS
from physical_ai_go2.video.walk_jump_summary_card import WalkJumpSummaryCardDrawer

STEPS_PER_SECOND = 50  # one control step is 0.02 s
STEP_SECONDS = 0.02
FRAMES_PER_SECOND = 50
VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720
# [解説] 跳べフラグが立っている間だけ、跳ねる回数が見えるよう 1/2 倍速にする。
SLOW_FACTOR = 2
# [解説] キーを押している表示を出す秒数。ジャンプの指令は押した瞬間に 1 回だけ出る。
JUMP_KEY_DISPLAY_SECONDS = 0.3
# [解説] カメラは歩行のデモ（make_key_demo_video.py）と同じ位置から撮る。
CAMERA_BEHIND_DISTANCE = 2.6
CAMERA_RIGHT_DISTANCE = 0.9
CAMERA_HEIGHT = 1.5
CAMERA_TARGET_HEIGHT = -0.05


def capture_following_camera(environment):
    robot_x = float(environment.base_position[0, 0])
    robot_y = float(environment.base_position[0, 1])
    environment.camera.set_pose(
        pos=(robot_x - CAMERA_BEHIND_DISTANCE, robot_y - CAMERA_RIGHT_DISTANCE, CAMERA_HEIGHT),
        lookat=(robot_x, robot_y, CAMERA_TARGET_HEIGHT),
    )
    return numpy.asarray(environment.camera.render()[0])


# [解説] 台本の区間のキーを押した状態にして、teleop と同じ KeyboardController で指令を作る。
def press_segment_keys(controller, environment, segment):
    controller.stop()
    for key_name in segment["keys"]:
        controller.set_held(key_name, True)
    velocity_command = controller.velocity_command()
    environment.set_velocity_commands(velocity_command[0], velocity_command[1], velocity_command[2])
    return velocity_command


# [解説] 1 区間を撮影して tracker を返す。戻り値の observation は次の区間に引き継ぐ。
def record_segment(segment, loaded_policy, controller, observation, overlay_drawer, writer):
    environment = loaded_policy.environment
    velocity_command = press_segment_keys(controller, environment, segment)
    tracker = SegmentMotionTracker(segment, velocity_command, STEP_SECONDS)

    jump_step = None
    jump_key_end_step = None
    if segment["jump_offset"] is not None:
        jump_step = int(segment["jump_offset"] * STEPS_PER_SECOND)
        jump_key_end_step = jump_step + int(JUMP_KEY_DISPLAY_SECONDS * STEPS_PER_SECOND)

    reset_count = 0
    for step_number in range(int(segment["seconds"] * STEPS_PER_SECOND)):
        if step_number == jump_step:
            controller.jump()
            tracker.mark_jump_pressed()
        observation, rewards, dones, extras = environment.step(loaded_policy.policy(observation))
        if bool(dones[0]):
            reset_count += 1
        tracker.update(environment)

        pressed_keys = list(segment["keys"])
        if jump_step is not None:
            if step_number >= jump_step and step_number < jump_key_end_step:
                pressed_keys.append("jump")
        is_jumping = bool(environment.is_jumping()[0])
        status = {"pressed_keys": pressed_keys, "is_jumping": is_jumping, "is_slow": is_jumping}
        frame = overlay_drawer.draw(capture_following_camera(environment), segment, tracker, status)
        frame_copies = 1
        if is_jumping:
            frame_copies = SLOW_FACTOR
        for copy_number in range(frame_copies):
            writer.append_data(frame)
    if reset_count > 0:
        print("resets in {}: {}".format(segment["name"], reset_count))
    return observation, tracker


def print_result(segment, tracker):
    print(
        "{}: command {:+.2f}, average {:+.2f}, expected {:+.2f}, actual {:+.2f}, bounces {}, flights {}, max jump height {}".format(
            segment["name"],
            tracker.commanded_velocity,
            tracker.average_velocity(),
            tracker.expected_progress(),
            tracker.actual_progress,
            tracker.bounce_count,
            tracker.flight_count,
            tracker.max_jump_height,
        )
    )


def record_demo(writer):
    loaded_policy = LoadedPolicy(EXPERIMENT_NAME, 1, camera_resolution=(VIDEO_WIDTH, VIDEO_HEIGHT))
    environment = loaded_policy.environment
    controller = KeyboardController(environment)
    overlay_drawer = WalkJumpDemoOverlayDrawer(VIDEO_WIDTH, VIDEO_HEIGHT)
    environment.set_velocity_commands(0.0, 0.0, 0.0)
    observation = environment.reset()

    results = []
    with torch.no_grad():
        for segment in WALK_JUMP_DEMO_SEGMENTS:
            observation, tracker = record_segment(
                segment, loaded_policy, controller, observation, overlay_drawer, writer
            )
            if segment["name"] != "idle":
                results.append({"segment": segment, "tracker": tracker})
                print_result(segment, tracker)
    return results


def write_summary_card(results, writer):
    card = WalkJumpSummaryCardDrawer(VIDEO_WIDTH, VIDEO_HEIGHT).draw(results)
    for frame_number in range(SUMMARY_SECONDS * FRAMES_PER_SECOND):
        writer.append_data(card)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="videos/go2_walk_jump_keyboard.mp4")
    arguments = parser.parse_args()

    genesis.init(backend=genesis.cpu, logging_level="warning", seed=0)
    os.makedirs(os.path.dirname(arguments.out), exist_ok=True)
    writer = imageio.v2.get_writer(arguments.out, fps=FRAMES_PER_SECOND, codec="libx264", quality=8, macro_block_size=8)
    results = record_demo(writer)
    write_summary_card(results, writer)
    writer.close()
    print("saved " + arguments.out)


if __name__ == "__main__":
    main()
