import math

# [解説] これより短い滞空は、足がわずかに浮いただけとみなし、跳ねた回数に数えない（3 ステップ = 0.06 秒）。
MIN_FLIGHT_STEPS = 3
# [解説] 胴体がこの高さ [m] を上に超えたら、1 回上下したと数える。W4 の普段の歩行では 0.29m までしか上がらない。
BOUNCE_HEIGHT = 0.33
# [解説] 量の名前と、速度指令のタプル（前後、左右、旋回）での位置の対応。
COMMAND_INDEX = {"forward": 0, "sideways": 1, "turning": 2}


# [解説] 台本の 1 区間のあいだ、指令に対して実際にどれだけ動いたかと、跳ねた回数を数えるクラス。
class SegmentMotionTracker:
    def __init__(self, segment, velocity_command, step_seconds):
        self.quantity = segment["quantity"]
        self.step_seconds = step_seconds
        self.commanded_velocity = 0.0
        if self.quantity is not None:
            self.commanded_velocity = velocity_command[COMMAND_INDEX[self.quantity]]
        self.elapsed_seconds = 0.0
        self.actual_progress = 0.0
        self.is_jump_pressed = False
        self.is_above_bounce_height = False
        self.bounce_count = 0
        self.flight_count = 0
        self.max_jump_height = None

    # [解説] j キーを押したあとの動きだけを数える。フラグが下がったあとの跳ねも含める。
    def mark_jump_pressed(self):
        self.is_jump_pressed = True

    def update(self, environment):
        self.elapsed_seconds += self.step_seconds
        if self.quantity is not None:
            self.actual_progress += self._read_velocity(environment) * self.step_seconds
        if self.is_jump_pressed:
            self._update_jump(environment)

    def _read_velocity(self, environment):
        if self.quantity == "forward":
            return float(environment.base_linear_velocity[0, 0])
        if self.quantity == "sideways":
            return float(environment.base_linear_velocity[0, 1])
        return float(environment.base_angular_velocity[0, 2])

    def _update_jump(self, environment):
        height = float(environment.base_position[0, 2])
        if self.max_jump_height is None or height > self.max_jump_height:
            self.max_jump_height = height
        is_above = height > BOUNCE_HEIGHT
        if is_above and not self.is_above_bounce_height:
            self.bounce_count += 1
        self.is_above_bounce_height = is_above
        if int(environment.foot_contact_tracker.airborne_steps[0]) == MIN_FLIGHT_STEPS:
            self.flight_count += 1

    def average_velocity(self):
        if self.elapsed_seconds == 0.0:
            return 0.0
        return self.actual_progress / self.elapsed_seconds

    def expected_progress(self):
        return self.commanded_velocity * self.elapsed_seconds

    # [解説] 表示用の単位に直す。旋回は角度 [度]、それ以外は距離 [m] のまま。
    def to_display_progress(self, progress):
        if self.quantity == "turning":
            return math.degrees(progress)
        return progress
