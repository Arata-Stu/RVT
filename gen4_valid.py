import os

# 固定パラメータ
gpu_ids = 0
batch_size_per_gpu = 4
mdl_cfg = "tiny"  # MDL_CFGの値を指定
data_dir = "/home/metis/Arata_repos/pre_gen4"  # DATA_DIRの値を指定
ckpt_path = ".ckpt"  # CKPT_PATHの値を指定

input_channels = 3  # 入力チャンネル数
event_frame_dts = [50]  # 必要に応じて値を追加


# ループ処理
for dt in event_frame_dts:
    command = f"""
        python3 validation.py dataset=gen4 dataset.path={data_dir} checkpoint="'{ckpt_path}'" \
        +experiment/gen4="{mdl_cfg}.yaml" hardware.gpus={gpu_ids} \
        batch_size.eval={batch_size_per_gpu} use_test_set=1 \
        dataset.ev_repr_name="'event_frame_dt={dt}'" model.backbone.input_channels={input_channels} model.postprocess.confidence_threshold=0.001
        """


    print(f"Running command for gen4 event_frame_dt={dt}")
    os.system(command)  # 実際にコマンドを実行
