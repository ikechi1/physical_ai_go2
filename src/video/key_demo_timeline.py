# [解説] キー操作デモ動画で、どのキーをいつ押すかの台本（make_key_demo_video.py が読む）。
# [解説] key は keyboard_controller.py の HELD_KEYS の名前。start 秒に押して end 秒に離す。
# [解説] 左右の旋回と左右の横移動を続けて並べ、ロボットの向きを最初に近い状態へ戻す。
DEMO_SECONDS = 30.0
KEY_PRESSES = [
    {"key": "forward", "start": 1.2, "end": 5.2},
    {"key": "backward", "start": 6.0, "end": 10.0},
    {"key": "turn_left", "start": 10.8, "end": 14.8},
    {"key": "turn_right", "start": 15.6, "end": 19.6},
    {"key": "side_left", "start": 20.4, "end": 24.4},
    {"key": "side_right", "start": 25.2, "end": 29.2},
]


# [解説] その時刻に押しているキーの名前を返す。何も押していなければ None を返す。
def find_pressed_key(time):
    for key_press in KEY_PRESSES:
        if key_press["start"] <= time and time < key_press["end"]:
            return key_press["key"]
    return None
