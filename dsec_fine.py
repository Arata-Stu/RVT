import os

# 固定パラメータ
gpu_ids = 0
batch_size_per_gpu = 8
train_workers_per_gpu = 12
eval_workers_per_gpu = 4
mdl_cfg = "tiny"  # MDL_CFGの値を指定
base_data_dir = "./datasets/pre_dsec"  # DATA_DIRの値を指定
artifact_name = "iasl_at-gifu-university/part2_RVT_gen4_frame50/checkpoint-fbjr3fbh-last:v39"

sampling = "mixed"
input_channels = 3  # 入力チャンネル数
# event_frame_dts = [5, 10, 20, 50, 100]  # 必要に応じて値を追加
event_frame_dts = [5]  # 必要に応じて値を追加

# ループ処理
for dt in event_frame_dts:
    data_dir = f"{base_data_dir}_{dt}"
    command = f"""
    python3 train.py model=rnndet dataset=sdsec dataset.path={data_dir} wandb.project_name=part2_RVT_dsec_frame_{dt} \
    wandb.group_name=dsec +experiment/dsec={mdl_cfg}.yaml hardware.gpus={gpu_ids} \
    batch_size.train={batch_size_per_gpu} batch_size.eval={batch_size_per_gpu} \
    hardware.num_workers.train={train_workers_per_gpu} hardware.num_workers.eval={eval_workers_per_gpu} \
    dataset.ev_repr_name="'event_frame_dt={dt}'" model.backbone.input_channels={input_channels} dataset.train.sampling={sampling}\
    training.max_steps=200000 wandb.resume_only_weights=True wandb.artifact_name="{artifact_name}" \
    """
    print(f"Running command for dsec event_frame_dt={dt}")
    os.system(command)  # 実際にコマンドを実行
