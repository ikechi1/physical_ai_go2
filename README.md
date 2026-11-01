# physical-ai-go2

Genesis シミュレーター上で Unitree Go2 をキーボード操作するプロジェクト。

速度指令（前後・横・旋回）に追従する歩行ポリシーと、ジャンプのポリシーを強化学習（rsl-rl の PPO）で学習し、
その指令をキーボードから与えて操作する。学習したポリシーの動きは動画にもできる。

## 使い方

```bash
uv sync

# 1. ポリシーの学習（Apple Silicon の Metal で約5分）
uv run go2-train-walk        # 歩行          → logs/go2-walking/
uv run go2-train-jump        # ジャンプ      → logs/go2-jump/
uv run go2-train-walk-jump   # 歩行＆ジャンプ → logs/go2-walk-jump/
#   既にログがあると止まる。別名で学習: uv run go2-train-walk -e go2-v2
#   消して学習し直す: uv run go2-train-walk --overwrite

# 2. キーボード操作（最新のチェックポイントを読み込む。-e で使うポリシーを選ぶ）
uv run go2-teleop                   # logs/go2-walking/
uv run go2-teleop -e go2-walk-jump

# 3. 動画の作成（logs/ の学習済みポリシーを使い、videos/ に保存する。--out で保存先を変える）
uv run go2-make-video                 # 提出用の動画       → videos/go2_walk_jump.mp4
uv run go2-make-video --only s7       # 1 シーンだけ確認   → videos/s7.mp4
uv run go2-make-key-demo-video        # 歩行のキー操作     → videos/go2_walk_keyboard.mp4
uv run go2-make-jump-demo-video       # 仮説ごとのジャンプ → videos/go2_jump_hypotheses.mp4
```

キーボード操作は、ビューアのウィンドウをクリックしてから行う。

| キー | 動作 |
| --- | --- |
| ↑ / ↓ | 前進 / 後退 |
| ← / → | 左旋回 / 右旋回 |
| q / e | 左 / 右へ横歩き |
| j | ジャンプ（跳べフラグ付きで学習したポリシーのみ） |
| space | 停止 |
| u | ロボットをリセット |
| esc | 終了 |

## ファイル

`src/physical_ai_go2/` の中身。

### `environment/`: 強化学習の環境

| ファイル | 役割 |
| --- | --- |
| `go2_environment.py` | Go2 の強化学習環境（Genesis の `examples/locomotion/go2_env.py` が元） |
| `velocity_commands.py` | 速度指令（前後・横・旋回）の引き直しと外からの設定 |
| `jump_command.py` | 跳べフラグ（1.0 秒間）とジャンプ中の経過時間・跳び上がった回数 |
| `foot_contact_tracker.py` | 足と胴体の接地、滞空時間、跳び上がった瞬間の判定 |
| `rewards.py` | 報酬の計算（報酬の名前と関数の対応） |

### `training/`: 学習

| ファイル | 役割 |
| --- | --- |
| `training_config.py` | 学習の共通設定（PPO・ロボット・観測） |
| `training_runner.py` | 学習の実行（引数の解析、ログフォルダ、学習ループ） |
| `walking_weights.py` | 歩行ポリシーの重みを歩行＆ジャンプの初期値にする（`--init-from`） |
| `train_walk.py` / `train_jump.py` / `train_walk_jump.py` | 歩行・その場ジャンプ・歩行＆ジャンプの報酬と指令 |

### `control/`: 学習済みポリシーの読み込みとキーボード操作

| ファイル | 役割 |
| --- | --- |
| `policy_loading.py` | 学習済みポリシーと環境の読み込み（操作・動画用） |
| `keyboard_controller.py` / `teleop.py` | キーボード操作 |

### `video/`: 動画の作成

| ファイル | 役割 |
| --- | --- |
| `video_drawing.py` | 字幕の描画（各動画で共通） |
| `video_scenes.py` / `make_video.py` | 提出用の動画の構成・撮影 |
| `key_demo_timeline.py` / `key_input_overlay.py` / `make_key_demo_video.py` | キー入力を表示する歩行デモ動画の台本・表示の描画・撮影 |
| `jump_demo_timeline.py` / `jump_demo_overlay.py` / `make_jump_demo_video.py` | 仮説ごとのジャンプをキー入力の表示付きで見せるデモ動画の台本・表示の描画・撮影 |
