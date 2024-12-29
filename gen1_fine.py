import os

# 固定パラメータ
gpu_ids = 0
batch_size_per_gpu = 8
train_workers_per_gpu = 6
eval_workers_per_gpu = 2
mdl_cfg = "tiny"  # MDL_CFGの値を指定
data_dir = "/home/metis/Arata_repos/pre_gen1"  # DATA_DIRの値を指定
artifact_name = ""

input_channels = 3  # 入力チャンネル数
event_frame_dts = [5]  # 必要に応じて値を追加

# ループ処理
for dt in event_frame_dts:
    command = f"""
    python3 train.py model=rnndet dataset=gen1 dataset.path={data_dir} wandb.project_name=RVT_gen1_frame_{dt} \
    wandb.group_name=gen1 +experiment/gen1={mdl_cfg}.yaml hardware.gpus={gpu_ids} \
    batch_size.train={batch_size_per_gpu} batch_size.eval={batch_size_per_gpu} \
    hardware.num_workers.train={train_workers_per_gpu} hardware.num_workers.eval={eval_workers_per_gpu} \
    dataset.ev_repr_name="'event_frame_dt={dt}'" model.backbone.input_channels={input_channels} \
    training.max_steps=200000 wandb.resume_only_weights=True wandb.artifact_name="{artifact_name}" \
    """
    print(f"Running command for gen1 event_frame_dt={dt}")
    os.system(command)  # 実際にコマンドを実行
