import numpy
from PIL import Image, ImageDraw

from video.video_drawing import CARD_BACKGROUND_COLOR, VideoFrameDrawer
from video.walk_jump_demo_overlay import LEVEL_COLORS, QUANTITY_LABELS

TEXT_COLOR = (240, 240, 240)
SUBTEXT_COLOR = (190, 205, 225)
RULE_COLOR = (70, 78, 92)
ROW_HEIGHT = 50
TABLE_TOP = 118
# [解説] 表の列の左端の位置。
COLUMN_ACTION = 60
COLUMN_COMMAND = 400
COLUMN_MEASURED = 620
COLUMN_WALKING_ONLY = 840
COLUMN_VERDICT = 1060


# [解説] 動画の最後に出す、キーごとの指令・実測・判定の表を描くクラス。値は撮影で測ったものを使う。
class WalkJumpSummaryCardDrawer:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.font_source = VideoFrameDrawer(width, height)

    # [解説] results は「segment と、その区間を測った tracker」の組の並び。
    def draw(self, results):
        image = Image.new("RGB", (self.width, self.height), CARD_BACKGROUND_COLOR)
        drawing = ImageDraw.Draw(image)
        drawing.text(
            (COLUMN_ACTION, 34), "歩行＋ジャンプ（W4）：キー操作の結果", font=self.font_source.font(38), fill=TEXT_COLOR
        )
        self._draw_table_header(drawing)
        row_number = 0
        for result in results:
            self._draw_row(drawing, TABLE_TOP + 44 + row_number * ROW_HEIGHT, result["segment"], result["tracker"])
            row_number += 1
        conclusion_top = TABLE_TOP + 44 + row_number * ROW_HEIGHT + 14
        drawing.text(
            (COLUMN_ACTION, conclusion_top),
            "歩行のみのポリシーでできていた後進・その場旋回・右への横移動が、W4 ではほぼできない。",
            font=self.font_source.font(26),
            fill=(255, 190, 90),
        )
        drawing.text(
            (COLUMN_ACTION, conclusion_top + 44),
            "ジャンプは、1 回の入力で 1 回跳びにならない。",
            font=self.font_source.font(26),
            fill=(255, 190, 90),
        )
        return numpy.asarray(image)

    def _draw_table_header(self, drawing):
        header_font = self.font_source.font(22)
        drawing.text((COLUMN_ACTION, TABLE_TOP), "キー操作", font=header_font, fill=SUBTEXT_COLOR)
        drawing.text((COLUMN_COMMAND, TABLE_TOP), "指令", font=header_font, fill=SUBTEXT_COLOR)
        drawing.text((COLUMN_MEASURED, TABLE_TOP), "W4 の実測", font=header_font, fill=SUBTEXT_COLOR)
        drawing.text((COLUMN_WALKING_ONLY, TABLE_TOP), "歩行のみ", font=header_font, fill=SUBTEXT_COLOR)
        drawing.text((COLUMN_VERDICT, TABLE_TOP), "判定", font=header_font, fill=SUBTEXT_COLOR)
        drawing.line((COLUMN_ACTION, TABLE_TOP + 34, self.width - 60, TABLE_TOP + 34), fill=RULE_COLOR, width=2)

    def _draw_row(self, drawing, top, segment, tracker):
        row_font = self.font_source.font(24)
        drawing.text((COLUMN_ACTION, top), segment["name"], font=row_font, fill=TEXT_COLOR)
        if segment["quantity"] is not None:
            unit = QUANTITY_LABELS[segment["quantity"]]["velocity_unit"]
            drawing.text(
                (COLUMN_COMMAND, top),
                "{:+.1f} {}".format(tracker.commanded_velocity, unit),
                font=row_font,
                fill=TEXT_COLOR,
            )
        measured_text = self._measured_text(segment, tracker)
        drawing.text((COLUMN_MEASURED, top), measured_text, font=row_font, fill=TEXT_COLOR)
        if segment.get("walking_only") is not None:
            drawing.text(
                (COLUMN_WALKING_ONLY, top),
                "{:+.2f} {}".format(segment["walking_only"], QUANTITY_LABELS[segment["quantity"]]["velocity_unit"]),
                font=row_font,
                fill=SUBTEXT_COLOR,
            )
        self._draw_verdict_badge(drawing, top, segment)

    # [解説] ジャンプの行は速度ではなく、1 回の入力で胴体が上下した回数と、4 本とも浮いた回数を出す。
    def _measured_text(self, segment, tracker):
        if segment["jump_offset"] is not None:
            return "上下 {} 回（4 本とも浮いた {} 回）".format(tracker.bounce_count, tracker.flight_count)
        unit = QUANTITY_LABELS[segment["quantity"]]["velocity_unit"]
        return "{:+.2f} {}".format(tracker.average_velocity(), unit)

    def _draw_verdict_badge(self, drawing, top, segment):
        badge_font = self.font_source.font(20)
        text_box = drawing.textbbox((COLUMN_VERDICT + 10, top + 2), segment["verdict"], font=badge_font)
        color = LEVEL_COLORS[segment["level"]]
        drawing.rounded_rectangle(
            (COLUMN_VERDICT, top - 4, text_box[2] + 10, top + 32), radius=7, fill=(color[0], color[1], color[2])
        )
        drawing.text((COLUMN_VERDICT + 10, top + 2), segment["verdict"], font=badge_font, fill=(255, 255, 255))
