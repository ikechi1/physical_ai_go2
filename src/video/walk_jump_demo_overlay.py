import numpy
from PIL import Image, ImageDraw

from video.key_input_overlay import KEY_LABELS
from video.video_drawing import VideoFrameDrawer

PANEL_HEIGHT = 200
KEY_WIDTH = 70
KEY_HEIGHT = 56
KEY_GAP = 8
KEY_IDLE_COLOR = (70, 78, 92, 255)
KEY_PRESSED_COLOR = (245, 158, 40, 255)
TEXT_COLOR = (255, 255, 255)
SUBTEXT_COLOR = (200, 212, 228)
IDLE_TEXT_COLOR = (170, 180, 195)
# [解説] 判定ごとの色。good は緑、partial は黄、bad は赤。
LEVEL_COLORS = {
    "good": (40, 150, 90, 255),
    "partial": (205, 135, 20, 255),
    "bad": (205, 55, 45, 255),
}
# [解説] 量ごとの表示名と単位。
QUANTITY_LABELS = {
    "forward": {"name": "前後の速度", "velocity_unit": "m/s", "progress_unit": "m", "decimals": 2},
    "sideways": {"name": "左右の速度", "velocity_unit": "m/s", "progress_unit": "m", "decimals": 2},
    "turning": {"name": "旋回の速度", "velocity_unit": "rad/s", "progress_unit": "度", "decimals": 0},
}
# [解説] キーの並び。パネルの右端を基準にした列と行で置く（Q、E、J は左の組、矢印キーは右の組）。
KEY_POSITIONS = {
    "side_left": {"column": 0, "row": 0, "cap": "Q"},
    "side_right": {"column": 1, "row": 0, "cap": "E"},
    "jump": {"column": 0.5, "row": 1, "cap": "J"},
    "forward": {"column": 3.5, "row": 0, "cap": KEY_LABELS["forward"]["cap"]},
    "turn_left": {"column": 2.5, "row": 1, "cap": KEY_LABELS["turn_left"]["cap"]},
    "backward": {"column": 3.5, "row": 1, "cap": KEY_LABELS["backward"]["cap"]},
    "turn_right": {"column": 4.5, "row": 1, "cap": KEY_LABELS["turn_right"]["cap"]},
}
KEY_COLUMN_COUNT = 5.5


# [解説] 映像に、押しているキー、判定、指令と実測の差、跳ねた回数、跳べフラグ、再生速度を重ねるクラス。
class WalkJumpDemoOverlayDrawer:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.font_source = VideoFrameDrawer(width, height)

    def draw(self, frame, segment, tracker, status):
        image = Image.fromarray(frame)
        drawing = ImageDraw.Draw(image, "RGBA")
        panel_top = self.height - PANEL_HEIGHT
        drawing.rectangle((0, panel_top, self.width, self.height), fill=(10, 14, 22, 215))
        self._draw_header(drawing, status)
        self._draw_keys(drawing, panel_top, status["pressed_keys"])
        if segment["name"] == "idle":
            drawing.text((40, panel_top + 62), "入力なし（停止）", font=self.font_source.font(44), fill=IDLE_TEXT_COLOR)
            return numpy.asarray(image)
        self._draw_verdict(drawing, panel_top, segment)
        row_top = panel_top + 74
        if segment["quantity"] is not None:
            self._draw_comparison(drawing, row_top, tracker)
            row_top += 66
        if segment["jump_offset"] is not None:
            self._draw_jump_result(drawing, row_top, tracker, status["is_jumping"])
        return numpy.asarray(image)

    def _draw_header(self, drawing, status):
        self._draw_label(drawing, (28, 22), "Go2 歩行＋ジャンプ（W4）をキーボードで操作")
        speed_text = "1 倍速"
        if status["is_slow"]:
            speed_text = "1/2 倍速"
        self._draw_label(drawing, (self.width - 150, 22), speed_text)

    def _draw_label(self, drawing, position, text):
        font = self.font_source.font(26)
        box = drawing.textbbox(position, text, font=font)
        drawing.rounded_rectangle(
            (box[0] - 12, box[1] - 8, box[2] + 12, box[3] + 8), radius=8, fill=(10, 14, 22, 190)
        )
        drawing.text(position, text, font=font, fill=TEXT_COLOR)

    def _draw_verdict(self, drawing, panel_top, segment):
        badge_font = self.font_source.font(24)
        badge_box = drawing.textbbox((52, panel_top + 24), segment["verdict"], font=badge_font)
        drawing.rounded_rectangle(
            (40, panel_top + 16, badge_box[2] + 12, panel_top + 60), radius=9, fill=LEVEL_COLORS[segment["level"]]
        )
        drawing.text((52, panel_top + 24), segment["verdict"], font=badge_font, fill=TEXT_COLOR)
        title = segment["name"] + "：" + segment["caption"]
        drawing.text((badge_box[2] + 28, panel_top + 20), title, font=self.font_source.font(26), fill=TEXT_COLOR)

    def _draw_comparison(self, drawing, top, tracker):
        labels = QUANTITY_LABELS[tracker.quantity]
        velocity_text = "{}   指令 {:+.1f} {}   →   実測（平均） {:+.2f} {}".format(
            labels["name"],
            tracker.commanded_velocity,
            labels["velocity_unit"],
            tracker.average_velocity(),
            labels["velocity_unit"],
        )
        drawing.text((40, top), velocity_text, font=self.font_source.font(22), fill=SUBTEXT_COLOR)
        progress_format = "動いた量     指令どおりなら {:+." + str(labels["decimals"]) + "f} {}   →   実際 {:+."
        progress_format += str(labels["decimals"]) + "f} {}"
        progress_text = progress_format.format(
            tracker.to_display_progress(tracker.expected_progress()),
            labels["progress_unit"],
            tracker.to_display_progress(tracker.actual_progress),
            labels["progress_unit"],
        )
        drawing.text((40, top + 32), progress_text, font=self.font_source.font(22), fill=(255, 190, 90))

    def _draw_jump_result(self, drawing, top, tracker, is_jumping):
        flag_text = "跳べフラグ OFF"
        flag_color = SUBTEXT_COLOR
        if is_jumping:
            flag_text = "跳べフラグ ON"
            flag_color = (255, 120, 80)
        result_text = "上下 {} 回 / 4 本とも浮いた {} 回".format(tracker.bounce_count, tracker.flight_count)
        if tracker.max_jump_height is not None:
            result_text += " / 最高 {:.2f} m".format(tracker.max_jump_height)
        drawing.text((40, top), flag_text, font=self.font_source.font(22), fill=flag_color)
        drawing.text((230, top), result_text, font=self.font_source.font(22), fill=(255, 190, 90))

    def _draw_keys(self, drawing, panel_top, pressed_keys):
        keys_width = KEY_COLUMN_COUNT * (KEY_WIDTH + KEY_GAP)
        keys_left = self.width - 40 - keys_width
        keys_top = panel_top + int((PANEL_HEIGHT - (2 * KEY_HEIGHT + KEY_GAP)) / 2)
        for key_name, position in KEY_POSITIONS.items():
            left = keys_left + position["column"] * (KEY_WIDTH + KEY_GAP)
            top = keys_top + position["row"] * (KEY_HEIGHT + KEY_GAP)
            fill_color = KEY_IDLE_COLOR
            text_color = TEXT_COLOR
            if key_name in pressed_keys:
                fill_color = KEY_PRESSED_COLOR
                text_color = (20, 20, 20)
            drawing.rounded_rectangle((left, top, left + KEY_WIDTH, top + KEY_HEIGHT), radius=9, fill=fill_color)
            drawing.text(
                (left + KEY_WIDTH / 2, top + KEY_HEIGHT / 2),
                position["cap"],
                font=self.font_source.font(30),
                fill=text_color,
                anchor="mm",
            )
