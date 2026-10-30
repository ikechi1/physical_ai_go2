# physical-ai-go2

Genesis シミュレーター上で Unitree Go2 をキーボード操作するプロジェクト。

速度指令（前後・横・旋回）に追従する歩行ポリシーを強化学習（rsl-rl の PPO）で学習し、
その指令をキーボードから与えて操作する。

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
```

ビューアのウィンドウをクリックしてから操作する。

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

- `src/physical_ai_go2/go2_env.py` — Go2 環境（Genesis の `examples/locomotion/go2_env.py` に外部からの速度指令設定を追加）
- `src/physical_ai_go2/train_common.py` — 学習の共通部分（PPO・ロボット・観測の設定と学習ループ）
- `src/physical_ai_go2/train_walk.py` — 歩行ポリシーの報酬と指令（前後・横・旋回）
- `src/physical_ai_go2/train_jump.py` — ジャンプポリシーの報酬と指令（跳べフラグのみ、その場で跳ぶ）
- `src/physical_ai_go2/train_walk_jump.py` — 歩行＆ジャンプポリシーの報酬と指令（速度指令＋跳べフラグ）
- `src/physical_ai_go2/teleop.py` — キーボード操作スクリプト
