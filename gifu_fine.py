import os

# 固定パラメータ
gpu_ids = 0
batch_size_per_gpu = 1
train_workers_per_gpu = 12
eval_workers_per_gpu = 4
mdl_cfg = "tiny"  # MDL_CFGの値を指定
base_data_dir = "./datasets/pre_gifu"  # DATA_DIRの値を指定

sampling = "random"
input_channels = 3  # 入力チャンネル数

# event_frame_dts に対応する artifact_name を指定
event_frame_dts = [5, 10, 20, 50, 100]  
artifact_names = {
    5: "iasl_at-gifu-university/part2_RVT_gen4_frame_5/checkpoint-ysxxqiip-last:v19",
    10: "iasl_at-gifu-university/part2_RVT_gen4_frame_10/checkpoint-5c3hgzh2-last:v19",
    20: "iasl_at-gifu-university/part2_RVT_gen4_frame_20/checkpoint-4s14997x-last:v19",
    50: "iasl_at-gifu-university/part2_RVT_gen4_frame50/checkpoint-fbjr3fbh-last:v39",
    100: "iasl_at-gifu-university/part2_RVT_gen4_frame_100/checkpoint-4ohzg5or-last:v0",
}

# ループ処理
for dt in event_frame_dts:
    data_dir = f"{base_data_dir}_{dt}"
    artifact_name = artifact_names.get(dt, "default-artifact")  # デフォルト値を設定する場合

    command = f"""
    python3 train.py model=rnndet dataset=gifu dataset.path={data_dir} wandb.project_name=part2_RVT_gifu_frame_{dt} \
    wandb.group_name=gifu +experiment/gifu={mdl_cfg}.yaml hardware.gpus={gpu_ids} \
    batch_size.train={batch_size_per_gpu} batch_size.eval={batch_size_per_gpu} \
    hardware.num_workers.train={train_workers_per_gpu} hardware.num_workers.eval={eval_workers_per_gpu} \
    dataset.ev_repr_name="'event_frame_dt={dt}'" model.backbone.input_channels={input_channels} dataset.train.sampling={sampling}\
    training.max_steps=200000 wandb.resume_only_weights=True wandb.artifact_name="{artifact_name}" \
    """
    print(f"Running command for gifu event_frame_dt={dt} with artifact {artifact_name}")
    os.system(command)  # 実際にコマンドを実行
