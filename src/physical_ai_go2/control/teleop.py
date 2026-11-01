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

import torch

import genesis

from physical_ai_go2.control.keyboard_controller import KeyboardController
from physical_ai_go2.control.policy_loading import LoadedPolicy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e", "--exp-name", type=str, default="go2-walking", help="Experiment name; also the log directory under logs/"
    )
    parser.add_argument("--ckpt", type=int, default=None, help="Checkpoint iteration to load (default: latest)")
    arguments = parser.parse_args()

    genesis.init(backend=genesis.cpu, logging_level="warning")

    loaded_policy = LoadedPolicy(arguments.exp_name, 1, checkpoint_iteration=arguments.ckpt, show_viewer=True)
    environment = loaded_policy.environment
    viewer = environment.scene.viewer
    viewer.follow_entity(environment.robot, smoothing=0.1)
    print("Loaded " + loaded_policy.checkpoint_path)
    print(__doc__)

    controller = KeyboardController(environment)
    controller.register(viewer)

    observation = environment.reset()
    last_command = None
    with torch.no_grad():
        while controller.is_running and viewer.is_alive():
            if controller.reset_requested:
                controller.reset_requested = False
                observation = environment.reset()

            command = controller.velocity_command()
            if command != last_command:
                print(
                    "command: vx={:+.2f} m/s  vy={:+.2f} m/s  yaw={:+.2f} rad/s".format(
                        command[0], command[1], command[2]
                    )
                )
                last_command = command
            environment.set_velocity_commands(command[0], command[1], command[2])

            actions = loaded_policy.policy(observation)
            observation, rewards, dones, extras = environment.step(actions)


if __name__ == "__main__":
    main()
