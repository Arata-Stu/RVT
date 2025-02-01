import os

# 固定パラメータ
gpu_ids = 0
batch_size_per_gpu = 1
mdl_cfg = "tiny"  # MDL_CFGの値を指定
base_data_dir = "./datasets/pre_gifu"  # DATA_DIRの値を指定

input_channels = 3  # 入力チャンネル数

# dtごとに異なるckptを指定する辞書
dt_ckpt_map = {
    5: "./ckpt/gen4_5.ckpt",
    10: "./ckpt/gen4_10.ckpt",
    20: "./ckpt/gen4_20.ckpt",
    50: "./ckpt/gen4_50.ckpt",
    100: "./ckpt/gen4_100.ckpt",
    # 必要に応じて追加
}

# ループ処理
for dt, ckpt_path in dt_ckpt_map.items():
    data_dir = f"{base_data_dir}_{dt}"
    command = f"""
        python3 validation.py dataset=gifu dataset.path={data_dir} checkpoint="'{ckpt_path}'" \
        +experiment/gifu="{mdl_cfg}.yaml" hardware.gpus={gpu_ids} \
        batch_size.eval={batch_size_per_gpu} use_test_set=1 \
        dataset.ev_repr_name="'event_frame_dt={dt}'" model.backbone.input_channels={input_channels} model.postprocess.confidence_threshold=0.001
        """

    print(f"Running command for gifu event_frame_dt={dt} with checkpoint {ckpt_path}")
    os.system(command)  # 実際にコマンドを実行
