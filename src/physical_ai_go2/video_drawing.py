import numpy
from PIL import Image, ImageDraw, ImageFont

FONT_PATH = "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc"
CARD_BACKGROUND_COLOR = (14, 22, 34)


# [解説] 動画のフレームに文字を描くクラス。文字だけの画面（カード）と、撮影した映像に重ねる字幕を作る。
class VideoFrameDrawer:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        # [解説] 同じ大きさのフォントを何度も読み込まないよう、大きさごとに覚えておく。
        self.fonts_by_size = {}

    def font(self, size):
        if size not in self.fonts_by_size:
            self.fonts_by_size[size] = ImageFont.truetype(FONT_PATH, size)
        return self.fonts_by_size[size]

    # [解説] 1 行目を見出し、2 行目以降を本文として描いた画面を 1 枚作る。
    # [解説] 2 行までなら中央ぞろえ、3 行以上なら本文を左ぞろえにする。
    def draw_card(self, lines):
        image = Image.new("RGB", (self.width, self.height), CARD_BACKGROUND_COLOR)
        drawing = ImageDraw.Draw(image)
        is_short = len(lines) <= 2
        top = 170
        if is_short:
            top = 230
        drawing.text((int(self.width / 2), top), lines[0], font=self.font(56), fill=(240, 240, 240), anchor="mm")
        for line_number in range(1, len(lines)):
            y = top + 110 + (line_number - 1) * 70
            if is_short:
                position = (int(self.width / 2), y)
                anchor = "mm"
            else:
                position = (140, y)
                anchor = "lm"
            drawing.text(position, lines[line_number], font=self.font(32), fill=(190, 205, 225), anchor=anchor)
        return numpy.asarray(image)

    # [解説] 撮影した映像に、下の字幕、左上の倍速と指令、右上の「跳べフラグ ON」を重ねる。
    def draw_overlay(self, frame, caption, velocity_command, is_jumping, slow_factor):
        image = Image.fromarray(frame)
        drawing = ImageDraw.Draw(image, "RGBA")
        drawing.rectangle((0, self.height - 80, self.width, self.height), fill=(0, 0, 0, 170))
        drawing.text(
            (int(self.width / 2), self.height - 40), caption, font=self.font(28), fill=(255, 255, 255), anchor="mm"
        )
        speed_label = "1 倍速"
        if slow_factor == 4:
            speed_label = "1/4 倍速"
        status_text = "{}   指令 vx={:+.1f}m/s  yaw={:+.1f}rad/s".format(
            speed_label, velocity_command["forward"], velocity_command["turning"]
        )
        drawing.text((24, 20), status_text, font=self.font(24), fill=(255, 255, 255))
        if is_jumping:
            drawing.rounded_rectangle((self.width - 210, 16, self.width - 24, 64), radius=10, fill=(230, 80, 40, 230))
            drawing.text((self.width - 117, 40), "跳べフラグ ON", font=self.font(24), fill=(255, 255, 255), anchor="mm")
        return numpy.asarray(image)
