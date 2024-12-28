import os

# 固定パラメータ
gpu_ids = [0]
batch_size_per_gpu = 16
train_workers_per_gpu = 8
eval_workers_per_gpu = 4
mdl_cfg = "yolox"  # MDL_CFGの値を指定
data_dir = "/home/ubuntu/gen4/pre_gen4"  # DATA_DIRの値を指定

input_channels = 3  # 入力チャンネル数
event_frame_dts = [5, 10, 20, 50, 100]  # 必要に応じて値を追加

gpu_ids_str = ",".join(map(str, gpu_ids))


# ループ処理
for dt in event_frame_dts:
    command = f"""
    python3 train.py model=yolox dataset=gen4 dataset.path={data_dir} wandb.project_name=YOLOX_gen4_frame_{dt} \
    wandb.group_name=gen4 +experiment/gen4={mdl_cfg}.yaml hardware.gpus="[ {gpu_ids_str} ]" \
    batch_size.train={batch_size_per_gpu} batch_size.eval={batch_size_per_gpu} \
    hardware.num_workers.train={train_workers_per_gpu} hardware.num_workers.eval={eval_workers_per_gpu} \
    dataset.ev_repr_name="'event_frame_dt={dt}'" dataset.train.sampling=random
    """
    print(f"Running command for gen4 event_frame_dt={dt}")
    os.system(command)  # 実際にコマンドを実行

