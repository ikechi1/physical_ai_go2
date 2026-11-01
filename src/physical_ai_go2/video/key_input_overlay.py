import numpy
from PIL import Image, ImageDraw

from physical_ai_go2.video.video_drawing import VideoFrameDrawer

PANEL_HEIGHT = 200
KEY_WIDTH = 70
KEY_HEIGHT = 56
KEY_GAP = 8
KEY_IDLE_COLOR = (70, 78, 92, 255)
KEY_PRESSED_COLOR = (245, 158, 40, 255)
TEXT_COLOR = (255, 255, 255)
SUBTEXT_COLOR = (200, 212, 228)
IDLE_TEXT_COLOR = (170, 180, 195)

# [解説] キーの名前ごとの、キーの上に書く文字と、押したときの動作の名前。
KEY_LABELS = {
    "forward": {"cap": "↑", "action": "前進"},
    "backward": {"cap": "↓", "action": "後進"},
    "turn_left": {"cap": "←", "action": "左旋回"},
    "turn_right": {"cap": "→", "action": "右旋回"},
    "side_left": {"cap": "Q", "action": "左へ横移動"},
    "side_right": {"cap": "E", "action": "右へ横移動"},
}
# [解説] キーの並び。パネルの右端を基準にした列と行で置く（Q と E は左の組、矢印キーは右の組）。
KEY_POSITIONS = {
    "side_left": {"column": 0, "row": 0},
    "side_right": {"column": 1, "row": 0},
    "forward": {"column": 3.5, "row": 0},
    "turn_left": {"column": 2.5, "row": 1},
    "backward": {"column": 3.5, "row": 1},
    "turn_right": {"column": 4.5, "row": 1},
}
KEY_COLUMN_COUNT = 5.5


# [解説] 撮影した映像に、押しているキーと、その動作、ロボットへの速度指令を重ねるクラス。
class KeyInputOverlayDrawer:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.font_source = VideoFrameDrawer(width, height)

    def draw(self, frame, pressed_key, velocity_command):
        image = Image.fromarray(frame)
        drawing = ImageDraw.Draw(image, "RGBA")
        panel_top = self.height - PANEL_HEIGHT
        drawing.rectangle((0, panel_top, self.width, self.height), fill=(10, 14, 22, 215))
        self._draw_title(drawing)
        self._draw_action(drawing, panel_top, pressed_key)
        self._draw_velocity_command(drawing, panel_top, velocity_command)
        self._draw_keys(drawing, panel_top, pressed_key)
        return numpy.asarray(image)

    def _draw_title(self, drawing):
        title = "Go2 歩行ポリシーをキーボードで操作（1 倍速）"
        title_font = self.font_source.font(26)
        title_box = drawing.textbbox((28, 22), title, font=title_font)
        drawing.rounded_rectangle(
            (title_box[0] - 12, title_box[1] - 8, title_box[2] + 12, title_box[3] + 8),
            radius=8,
            fill=(10, 14, 22, 190),
        )
        drawing.text((28, 22), title, font=title_font, fill=TEXT_COLOR)

    def _draw_action(self, drawing, panel_top, pressed_key):
        drawing.text((40, panel_top + 20), "入力中のキー", font=self.font_source.font(22), fill=SUBTEXT_COLOR)
        if pressed_key is None:
            drawing.text((40, panel_top + 56), "なし（停止）", font=self.font_source.font(56), fill=IDLE_TEXT_COLOR)
            return
        label = KEY_LABELS[pressed_key]
        action_text = label["cap"] + "  " + label["action"]
        drawing.text((40, panel_top + 56), action_text, font=self.font_source.font(56), fill=(255, 190, 90))

    def _draw_velocity_command(self, drawing, panel_top, velocity_command):
        command_text = "速度指令   前後 {:+.1f} m/s    左右 {:+.1f} m/s    旋回 {:+.1f} rad/s".format(
            velocity_command[0], velocity_command[1], velocity_command[2]
        )
        drawing.text((40, panel_top + 146), command_text, font=self.font_source.font(22), fill=SUBTEXT_COLOR)

    def _draw_keys(self, drawing, panel_top, pressed_key):
        keys_width = KEY_COLUMN_COUNT * (KEY_WIDTH + KEY_GAP)
        keys_left = self.width - 40 - keys_width
        keys_top = panel_top + int((PANEL_HEIGHT - (2 * KEY_HEIGHT + KEY_GAP)) / 2)
        for key_name, position in KEY_POSITIONS.items():
            left = keys_left + position["column"] * (KEY_WIDTH + KEY_GAP)
            top = keys_top + position["row"] * (KEY_HEIGHT + KEY_GAP)
            fill_color = KEY_IDLE_COLOR
            text_color = TEXT_COLOR
            if key_name == pressed_key:
                fill_color = KEY_PRESSED_COLOR
                text_color = (20, 20, 20)
            drawing.rounded_rectangle((left, top, left + KEY_WIDTH, top + KEY_HEIGHT), radius=9, fill=fill_color)
            drawing.text(
                (left + KEY_WIDTH / 2, top + KEY_HEIGHT / 2),
                KEY_LABELS[key_name]["cap"],
                font=self.font_source.font(30),
                fill=text_color,
                anchor="mm",
            )
