import os

# 固定パラメータ
gpu_ids = 0
batch_size_per_gpu = 1
train_workers_per_gpu = 12
eval_workers_per_gpu = 4
mdl_cfg = "yolox"  # MDL_CFGの値を指定
base_data_dir = "./datasets/pre_dsec"  # DATA_DIRの値を指定

sampling = "random"
input_channels = 3  # 入力チャンネル数

# event_frame_dts に対応する artifact_name を指定
event_frame_dts = [5, 10, 20, 50, 100]  
artifact_names = {
    5: "iasl_at-gifu-university/part2_YOLOX_gen4_frame_5/checkpoint-v80goh29-last:v39",
    10: "iasl_at-gifu-university/part2_YOLOX_gen4_frame_10/checkpoint-xd7j5hjw-last:v39",
    20: "iasl_at-gifu-university/part2_YOLOX_gen4_frame_20/checkpoint-q2r9731y-last:v39",
    50: "iasl_at-gifu-university/part2_YOLOX_gen4_frame_50/checkpoint-ihxc5u2m-last:v39",
    100: "iasl_at-gifu-university/part2_YOLOX_gen4_frame_100/checkpoint-2c76g26a-last:v39",
}

# ループ処理
for dt in event_frame_dts:
    data_dir = f"{base_data_dir}_{dt}"
    artifact_name = artifact_names.get(dt, "default-artifact")  # デフォルト値を設定する場合

    command = f"""
    python3 train.py model=yolox dataset=dsec dataset.path={data_dir} wandb.project_name=part2_YOLOX_dsec_frame_{dt} \
    wandb.group_name=dsec +experiment/dsec={mdl_cfg}.yaml hardware.gpus={gpu_ids} \
    batch_size.train={batch_size_per_gpu} batch_size.eval={batch_size_per_gpu} \
    hardware.num_workers.train={train_workers_per_gpu} hardware.num_workers.eval={eval_workers_per_gpu} \
    dataset.ev_repr_name="'event_frame_dt={dt}'" model.backbone.input_dim={input_channels} dataset.train.sampling={sampling}\
    training.max_steps=100000 wandb.resume_only_weights=True wandb.artifact_name="{artifact_name}" dataset.sequence_length=16 \
    """
    print(f"Running command for dsec event_frame_dt={dt} with artifact {artifact_name}")
    os.system(command)  # 実際にコマンドを実行
