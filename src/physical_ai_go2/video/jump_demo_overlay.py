import numpy
from PIL import Image, ImageDraw

from physical_ai_go2.video.video_drawing import VideoFrameDrawer

PANEL_HEIGHT = 200
KEY_SIZE = 84
TEXT_COLOR = (255, 255, 255)
SUBTEXT_COLOR = (200, 212, 228)
RESULT_COLOR = (255, 190, 90)
KEY_IDLE_COLOR = (70, 78, 92, 255)
KEY_PRESSED_COLOR = (245, 158, 40, 255)
FLAG_ON_COLOR = (230, 80, 40, 240)
FLAG_OFF_COLOR = (70, 78, 92, 255)


# [解説] 撮影した映像に、仮説とその結果、j キーの入力、跳べフラグ、胴体の高さ、再生速度を重ねるクラス。
class JumpDemoOverlayDrawer:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.font_source = VideoFrameDrawer(width, height)

    def draw(self, frame, scene, status):
        image = Image.fromarray(frame)
        drawing = ImageDraw.Draw(image, "RGBA")
        panel_top = self.height - PANEL_HEIGHT
        drawing.rectangle((0, panel_top, self.width, self.height), fill=(10, 14, 22, 215))
        self._draw_header(drawing, status)
        drawing.text((40, panel_top + 22), scene["hypothesis"], font=self.font_source.font(30), fill=TEXT_COLOR)
        drawing.text((40, panel_top + 72), "→ " + scene["result"], font=self.font_source.font(36), fill=RESULT_COLOR)
        self._draw_height(drawing, panel_top, status)
        self._draw_jump_key(drawing, panel_top, status["is_key_pressed"])
        self._draw_jump_flag(drawing, panel_top, status["is_jumping"])
        return numpy.asarray(image)

    def _draw_header(self, drawing, status):
        header = "Go2 その場ジャンプ：報酬の仮説で動きが変わる（{}/{}）".format(
            status["scene_number"], status["scene_count"]
        )
        self._draw_label(drawing, (28, 22), header)
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

    def _draw_height(self, drawing, panel_top, status):
        height_text = "胴体の高さ   いま {:.2f} m".format(status["height"])
        if status["max_height"] is not None:
            height_text += "    入力後の最高 {:.2f} m".format(status["max_height"])
        drawing.text((40, panel_top + 142), height_text, font=self.font_source.font(22), fill=SUBTEXT_COLOR)

    def _draw_jump_key(self, drawing, panel_top, is_key_pressed):
        left = self.width - 40 - 210
        top = panel_top + 22
        fill_color = KEY_IDLE_COLOR
        text_color = TEXT_COLOR
        if is_key_pressed:
            fill_color = KEY_PRESSED_COLOR
            text_color = (20, 20, 20)
        key_left = left + int((210 - KEY_SIZE) / 2)
        drawing.rounded_rectangle((key_left, top, key_left + KEY_SIZE, top + KEY_SIZE), radius=12, fill=fill_color)
        drawing.text(
            (key_left + KEY_SIZE / 2, top + KEY_SIZE / 2),
            "J",
            font=self.font_source.font(42),
            fill=text_color,
            anchor="mm",
        )

    def _draw_jump_flag(self, drawing, panel_top, is_jumping):
        left = self.width - 40 - 210
        top = panel_top + 124
        fill_color = FLAG_OFF_COLOR
        flag_text = "跳べフラグ OFF"
        if is_jumping:
            fill_color = FLAG_ON_COLOR
            flag_text = "跳べフラグ ON"
        drawing.rounded_rectangle((left, top, left + 210, top + 48), radius=10, fill=fill_color)
        drawing.text(
            (left + 105, top + 24), flag_text, font=self.font_source.font(24), fill=TEXT_COLOR, anchor="mm"
        )
