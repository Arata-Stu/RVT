import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from modules.utils.fetch import fetch_data_module
from omegaconf import OmegaConf, DictConfig
import cv2

import cv2
import numpy as np
from pathlib import Path

from data.utils.types import DataType
from utils.padding import InputPadderFromShape
from data.genx_utils.labels import ObjectLabels

def create_video_from_dataloader(yaml_path: str, output_path="output_video.mp4", fps=10, max_sequences=5):
    config = OmegaConf.load(yaml_path)

    data_module = fetch_data_module(config=config)
    data_module.setup("test")

    # 動画のWriterを初期化
    video_writer = None
    sequence_count = 0  # Trueの回数をカウント
    frame_size = None   # 動画のフレームサイズを記憶

    for batch in data_module.test_dataloader():
        data = batch["data"]
        ev_repr = data[DataType.EV_REPR]
        labels = data[DataType.OBJLABELS_SEQ]
        is_first_sample = data[DataType.IS_FIRST_SAMPLE]

        # シーケンスの区切りを判定
        if is_first_sample.any():
            sequence_count += 1
            if sequence_count > max_sequences:
                break  # 指定されたシーケンス数を超えたら終了

        # シーケンス内の各フレームを処理
        sequence_len = len(ev_repr)
        for tidx in range(sequence_len):
            ev_tensors = ev_repr[tidx]
            current_labels, valid_batch_indices = labels[tidx].get_valid_labels_and_batch_indices()

            # YOLOX形式のラベルを取得
            if len(current_labels) > 0:
                labels_yolox = ObjectLabels.get_labels_as_batched_tensor(obj_label_list=current_labels, format_='yolox')

            # 画像データの変換
            image = ev_tensors.squeeze(0).detach().cpu().numpy().astype('uint8').copy()
            if image.shape[0] == 3:
                image = np.transpose(image, (1, 2, 0))  # (C, H, W) -> (H, W, C)
            # if image.max() <= 1.0:
            #     image = (image * 255).astype(np.uint8)  # [0, 1] -> [0, 255]
            ## BGR
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image = image.astype(np.uint8).copy()

            for cls, cx, cy, w, h in labels_yolox[0]:
                # 不正なラベルをスキップ
                if any(val is None or np.isnan(val) for val in [cx, cy, w, h]):
                    continue

                # バウンディングボックスの範囲をクリップ
                x = max(0, int(cx - w / 2))
                y = max(0, int(cy - h / 2))
                x2 = min(image.shape[1] - 1, int(cx + w / 2))
                y2 = min(image.shape[0] - 1, int(cy + h / 2))

                # デバッグ: 座標と画像範囲を確認
                # print(f"x: {x}, y: {y}, x2: {x2}, y2: {y2}, image.shape: {image.shape}")

                # バウンディングボックスを描画
                color = (0, 255, 0)  # 緑色
                cv2.rectangle(image, (x, y), (x2, y2), color, 2)
                label = f"{cls}"
                cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # 動画のWriterを初期化
            if video_writer is None:
                frame_size = (image.shape[1], image.shape[0])  # 幅, 高さ
                video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, frame_size)

            # フレームを動画に書き込む
            video_writer.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))  # OpenCVはBGR形式

    # 動画Writerを解放
    if video_writer is not None:
        video_writer.release()
    print(f"動画が生成されました: {output_path}")

# 実行例
# create_video_from_dataloader(data_module, output_path="output_video.avi", fps=10, max_sequences=5)

