# [解説] 歩行＋ジャンプ（W4）のキー操作デモ動画の台本（make_walk_jump_demo_video.py が読む）。
# [解説] keys は押し続けるキー（keyboard_controller.py の HELD_KEYS の名前）、jump_offset は j キーを押す時刻 [秒]。
# [解説] quantity は指令と実測を比べる量（forward / sideways / turning）。None なら比べない。
# [解説] level と verdict は判定の表示。good は指令どおり、partial は一部だけ、bad は動かない。
# [解説] walking_only は同じ指令を歩行のみのポリシーに与えた実測の平均（go2-make-key-demo-video の結果）。
EXPERIMENT_NAME = "go2-walk-jump-w4"
SUMMARY_SECONDS = 8


# [解説] キーを何も押さない区間を作る。
def idle_segment(seconds):
    return {"name": "idle", "keys": [], "seconds": seconds, "jump_offset": None, "quantity": None}


WALK_JUMP_DEMO_SEGMENTS = [
    idle_segment(1.2),
    {
        "name": "前進",
        "keys": ["forward"],
        "seconds": 4.0,
        "jump_offset": None,
        "quantity": "forward",
        "walking_only": 0.93,
        "level": "partial",
        "verdict": "ずれる",
        "caption": "前には進むが、左へそれていく",
    },
    idle_segment(0.8),
    {
        "name": "後進",
        "keys": ["backward"],
        "seconds": 4.0,
        "jump_offset": None,
        "quantity": "forward",
        "walking_only": -0.45,
        "level": "bad",
        "verdict": "動かない",
        "caption": "押しても、その場に立ったまま",
    },
    idle_segment(0.8),
    {
        "name": "左旋回",
        "keys": ["turn_left"],
        "seconds": 4.0,
        "jump_offset": None,
        "quantity": "turning",
        "walking_only": 0.96,
        "level": "bad",
        "verdict": "動かない",
        "caption": "押しても、その場で回れない",
    },
    idle_segment(0.8),
    {
        "name": "右旋回",
        "keys": ["turn_right"],
        "seconds": 4.0,
        "jump_offset": None,
        "quantity": "turning",
        "walking_only": -0.99,
        "level": "bad",
        "verdict": "動かない",
        "caption": "押しても、その場で回れない",
    },
    idle_segment(0.8),
    {
        "name": "左へ横移動",
        "keys": ["side_left"],
        "seconds": 4.0,
        "jump_offset": None,
        "quantity": "sideways",
        "walking_only": 0.48,
        "level": "partial",
        "verdict": "足りない",
        "caption": "動くが、指令より遅い",
    },
    idle_segment(0.8),
    {
        "name": "右へ横移動",
        "keys": ["side_right"],
        "seconds": 4.0,
        "jump_offset": None,
        "quantity": "sideways",
        "walking_only": -0.47,
        "level": "bad",
        "verdict": "ほぼ動かない",
        "caption": "右へはほとんど動かない",
    },
    idle_segment(0.8),
    {
        "name": "その場でジャンプ",
        "keys": [],
        "seconds": 4.0,
        "jump_offset": 0.5,
        "quantity": None,
        "level": "partial",
        "verdict": "止まらない",
        "caption": "跳ぶが、何度も跳ねる",
    },
    {
        "name": "前進しながらジャンプ",
        "keys": ["forward"],
        "seconds": 6.0,
        "jump_offset": 2.0,
        "quantity": "forward",
        "walking_only": None,
        "level": "partial",
        "verdict": "上下が続く",
        "caption": "1 回跳びにならない",
    },
    idle_segment(1.0),
]
