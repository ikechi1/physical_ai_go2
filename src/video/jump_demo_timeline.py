# [解説] ジャンプのキー操作デモ動画の台本（make_jump_demo_video.py が読む）。
# [解説] どの場面も、立っている → j キーを押す → スローで跳ぶ様子を見せる → 1 倍速で着地後を見せる、の順。
# [解説] 1 場面の長さは 0.6 + 1.2 × 2 + 2.0 = 5 秒で、6 場面で 30 秒になる。
SETTLE_SECONDS = 1.0
STAND_SECONDS = 0.6
SLOW_SECONDS = 1.2
SLOW_FACTOR = 2
AFTER_SLOW_SECONDS = 2.0
# [解説] キーを押している表示を出す秒数。ジャンプの指令は押した瞬間に 1 回だけ出る。
KEY_DISPLAY_SECONDS = 0.3

# [解説] 動きがはっきり変わった仮説だけを選ぶ。experiment は logs/ の下の実験名。
JUMP_DEMO_SCENES = [
    {
        "experiment": "go2-jump",
        "hypothesis": "仮説1：フラグ中は体高の目標を +15cm",
        "result": "背伸びするだけで、跳ばない",
    },
    {
        "experiment": "go2-jump-upvel-noheight",
        "hypothesis": "仮説4：上向きの速度に報酬",
        "result": "高く跳ぶが、空中で傾きが 10° を超え、開始姿勢に戻される",
    },
    {
        "experiment": "go2-jump-upvel-capped",
        "hypothesis": "仮説5：上向き速度の報酬に時間と高さの制限",
        "result": "転ばないが、1 回の入力で何度も跳ねる",
    },
    {
        "experiment": "go2-jump-crouch",
        "hypothesis": "仮説8：踏み切りの時間を 0.3 → 0.5 秒に",
        "result": "胴体から床に落とし、その反動で跳ぶ",
    },
    {
        "experiment": "go2-jump-airfirst",
        "hypothesis": "仮説18：滞空の報酬は 1 回目の飛行だけ",
        "result": "しっかり跳ぶが、着地で小さく跳ねる",
    },
    {
        "experiment": "go2-jump-stick-500",
        "hypothesis": "仮説20：着地して 4 本足で立つ時間に報酬",
        "result": "1 回だけ跳んで、着地して立つ",
    },
]
