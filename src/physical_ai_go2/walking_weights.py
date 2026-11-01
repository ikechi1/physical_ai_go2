import torch

# [解説] 歩行だけのポリシーの観測の次元数。跳べフラグとジャンプの進み具合を足すと 47 次元になる。
WALKING_OBSERVATION_SIZE = 45
# [解説] 45 次元の観測で、速度指令（3 次元）が終わる位置。47 次元の観測では、ここに跳べフラグが入る。
JUMP_FLAG_COLUMN = 9


# [解説] 歩行のポリシー（観測 45 次元）の重みを、歩行＆ジャンプ（観測 47 次元）の初期値として読み込む。
# [解説] 47 次元の観測は、45 次元の 9 番目（速度指令の直後）に跳べフラグ、最後にジャンプの進み具合が増えた形。
# [解説] 最初の層の重みのその 2 列を 0 にして足すので、読み込んだ直後は 2 つの入力を無視して、元の歩行と同じ動きをする。
def load_walking_weights(runner, model_path, device):
    saved_model = torch.load(model_path, weights_only=False, map_location=device)
    for network_name in ["actor_state_dict", "critic_state_dict"]:
        first_layer_weights = saved_model[network_name]["mlp.0.weight"]
        if first_layer_weights.shape[1] == WALKING_OBSERVATION_SIZE:
            saved_model[network_name]["mlp.0.weight"] = add_jump_input_columns(first_layer_weights)
    # [解説] 最適化の状態（Adam の統計）と学習回数は引き継がず、ネットワークの重みと行動のばらつき（std）だけ使う。
    runner.alg.load(saved_model, {"actor": True, "critic": True}, strict=True)


# [解説] 最初の層の重み（出力数 × 45）に、跳べフラグとジャンプの進み具合の 2 列を 0 で足して、出力数 × 47 にする。
def add_jump_input_columns(first_layer_weights):
    output_count = first_layer_weights.shape[0]
    zero_column = torch.zeros(output_count, 1, device=first_layer_weights.device)
    columns_before_flag = first_layer_weights[:, :JUMP_FLAG_COLUMN]
    columns_after_flag = first_layer_weights[:, JUMP_FLAG_COLUMN:]
    return torch.cat([columns_before_flag, zero_column, columns_after_flag, zero_column], dim=1)
