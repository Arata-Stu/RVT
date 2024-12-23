import os

# 固定パラメータ
gpu_ids = 0
batch_size_per_gpu = 12
train_workers_per_gpu = 6
eval_workers_per_gpu = 2
mdl_cfg = "tiny"  # MDL_CFGの値を指定
data_dir = "/path/to/data"  # DATA_DIRの値を指定

# 繰り返し処理する値
event_frame_dts = [5, 10, 20, 33, 50, 100]  # 必要に応じて値を追加

# ループ処理
for dt in event_frame_dts:
    command = f"""
    python train.py model=rnndet dataset=gen4 dataset.path={data_dir} wandb.project_name=RVT_{dt} \
    wandb.group_name=1mpx +experiment/gen4={mdl_cfg}.yaml hardware.gpus={gpu_ids} \
    batch_size.train={batch_size_per_gpu} batch_size.eval={batch_size_per_gpu} \
    hardware.num_workers.train={train_workers_per_gpu} hardware.num_workers.eval={eval_workers_per_gpu} \
    dataset.ev_repr_name='event_frame_dt={dt}'
    """
    print(f"Running command for event_frame_dt={dt}")
    os.system(command)  # 実際にコマンドを実行
