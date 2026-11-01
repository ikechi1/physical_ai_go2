from genesis.vis.keybindings import Key, KeyAction, Keybind

# commands sent while a key is held (kept inside the trained command ranges)
FORWARD_VELOCITY_MAX = 1.0
BACKWARD_VELOCITY_MAX = -0.5
SIDEWAYS_VELOCITY_MAX = 0.5
TURNING_VELOCITY_MAX = 1.0

# [解説] 押している間だけ動くキー。名前とキーの対応。
HELD_KEYS = {
    "forward": Key.UP,
    "backward": Key.DOWN,
    "turn_left": Key.LEFT,
    "turn_right": Key.RIGHT,
    "side_left": Key.Q,
    "side_right": Key.E,
}


# [解説] ビューアのキー入力を受け取り、速度指令とジャンプ・リセット・終了の要求に変えるクラス。
class KeyboardController:
    def __init__(self, environment):
        self.environment = environment
        # keys currently held down
        self.held = {}
        for name in HELD_KEYS.keys():
            self.held[name] = False
        self.is_running = True
        self.reset_requested = False

    def register(self, viewer):
        keybinds = []
        for name, key in HELD_KEYS.items():
            keybinds.append(
                Keybind("go2_" + name + "_press", key, KeyAction.PRESS, callback=self.set_held, args=(name, True))
            )
            keybinds.append(
                Keybind("go2_" + name + "_release", key, KeyAction.RELEASE, callback=self.set_held, args=(name, False))
            )
        # [解説] j キーを押した瞬間（PRESS）に jump() を呼ぶ。矢印キーと違い、押し続けても意味はない。
        keybinds.append(Keybind("go2_jump", Key.J, KeyAction.PRESS, callback=self.jump))
        keybinds.append(Keybind("go2_stop", Key.SPACE, KeyAction.PRESS, callback=self.stop))
        keybinds.append(Keybind("go2_reset", Key.U, KeyAction.RELEASE, callback=self.request_reset))
        keybinds.append(Keybind("go2_quit", Key.ESCAPE, KeyAction.RELEASE, callback=self.quit))
        viewer.register_keybinds(*keybinds, overwrite=True)

    def set_held(self, name, is_held):
        self.held[name] = is_held

    def stop(self):
        for name in self.held.keys():
            self.held[name] = False

    # [解説] 跳べフラグ付きで学習したポリシーなら、跳べフラグを立てる。フラグは 1.0 秒後に自動で下がる。
    def jump(self):
        if self.environment.jump_enabled:
            self.environment.start_jump_for_all()
            print("jump")
        else:
            print("this policy was trained without the jump flag")

    def request_reset(self):
        self.reset_requested = True

    def quit(self):
        self.is_running = False

    # [解説] 押しているキーから速度指令（前後の速度、左右の速度、旋回速度）を作る。反対向きのキーを同時に押すと打ち消し合う。
    def velocity_command(self):
        forward_velocity = 0.0
        if self.held["forward"]:
            forward_velocity += FORWARD_VELOCITY_MAX
        if self.held["backward"]:
            forward_velocity += BACKWARD_VELOCITY_MAX
        sideways_velocity = 0.0
        if self.held["side_left"]:
            sideways_velocity += SIDEWAYS_VELOCITY_MAX
        if self.held["side_right"]:
            sideways_velocity -= SIDEWAYS_VELOCITY_MAX
        turning_velocity = 0.0
        if self.held["turn_left"]:
            turning_velocity += TURNING_VELOCITY_MAX
        if self.held["turn_right"]:
            turning_velocity -= TURNING_VELOCITY_MAX
        return (forward_velocity, sideways_velocity, turning_velocity)
