import cv2
import numpy as np
import torch 
from omegaconf import DictConfig
from typing import Tuple

from modules.utils.fetch import fetch_data_module, fetch_model_module
from utils.padding import InputPadderFromShape
from modules.utils.detection import RNNStates
from data.utils.types import DataType
from data.genx_utils.labels import ObjectLabels
from models.detection.yolox.utils.boxes import postprocess

class VideoWriter:
    def __init__(self, output_path: str, fps: int):
        self.output_path = output_path
        self.fps = fps
        self.video_writer = None

    def create_video_writer(self, frame_shape: Tuple[int, int]):
        self.video_writer = cv2.VideoWriter(self.output_path, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, frame_shape)
        # print(frame_shape)

    def visualize(self, ev_tensor, labels_yolox, predictions):
        
        ev_tensor = ev_tensor.squeeze(0).detach().cpu().numpy().astype('uint8').copy()

        if ev_tensor.shape[0] == 3:
            ev_tensor = np.transpose(ev_tensor, (1, 2, 0))  # (C, H, W) -> (H, W, C)

        if self.video_writer is None:
            # print(ev_tensor.shape)
            self.create_video_writer(frame_shape=(ev_tensor.shape[1], ev_tensor.shape[0]))

        ev_tensor = cv2.cvtColor(ev_tensor, cv2.COLOR_RGB2BGR).astype(np.uint8)


        if labels_yolox is not None:
            ## label = [cls, cx, cy, w, h]
            for cls, cx, cy, w, h in labels_yolox[0]:
                # 不正なラベルをスキップ
                if any(val is None or np.isnan(val) for val in [cx, cy, w, h]):
                    continue

                # バウンディングボックスの範囲をクリップ
                x = max(0, int(cx - w / 2))
                y = max(0, int(cy - h / 2))
                x2 = min(ev_tensor.shape[1] - 1, int(cx + w / 2))
                y2 = min(ev_tensor.shape[0] - 1, int(cy + h / 2))

                # バウンディングボックスを描画
                color = (0, 255, 0)  # 緑色
                cv2.rectangle(ev_tensor, (x, y), (x2, y2), color, 2)
                label = f"{cls}"
                cv2.putText(ev_tensor, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        
        if predictions is not None:
            ## predictions = [x1, y1, x2, y2, obj_conf, class_conf, class_id]
            for x1, y1, x2, y2, obj_conf, class_conf, class_id in predictions[0]:
                # バウンディングボックスの範囲をクリップ
                x1 = max(0, int(x1))
                y1 = max(0, int(y1))
                x2 = min(ev_tensor.shape[1] - 1, int(x2))
                y2 = min(ev_tensor.shape[0] - 1, int(y2))

                # バウンディングボックスを描画
                color = (255, 0, 0)

                cv2.rectangle(ev_tensor, (x1, y1), (x2, y2), color, 2)
                label = f"{class_id:.2f}"
                cv2.putText(ev_tensor, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


        self.video_writer.write(cv2.cvtColor(ev_tensor, cv2.COLOR_RGB2BGR))  # OpenCVはBGR形式


    def run(self, config: DictConfig, max_sequences: int = 1):
        
        data = fetch_data_module(config)
        data.setup("test")
        model = fetch_model_module(config)
        model.setup("test")
        model.eval()

        ckpt_path = config.ckpt_path
        if ckpt_path != "":
            ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
            model.load_state_dict(ckpt['state_dict'])

        sequence_count = 0

        rnn_state = RNNStates()
        input_padder = InputPadderFromShape((384, 640))

        for batch in data.test_dataloader():
            data = batch["data"]

            ev_repr = data[DataType.EV_REPR]
            labels = data[DataType.OBJLABELS_SEQ]
            is_first_sample = data[DataType.IS_FIRST_SAMPLE]

            rnn_state.reset(worker_id=0, indices_or_bool_tensor=is_first_sample)
            prev_states = rnn_state.get_states(worker_id=0)

            # シーケンスの区切りを判定
            if is_first_sample.any():
                sequence_count += 1
                if sequence_count > max_sequences:
                    break  # 指定されたシーケンス数を超えたら終了

            # シーケンス内の各フレームを処理
            sequence_len = len(ev_repr)
            
            for tidx in range(sequence_len):

                ev_tensors = ev_repr[tidx]
                ev_tensors = ev_tensors.to(torch.float32)
                ev_tensors_padded = input_padder.pad_tensor_ev_repr(ev_tensors)
                current_labels, valid_batch_indices = labels[tidx].get_valid_labels_and_batch_indices()

                # YOLOX形式のラベルを取得
                if len(current_labels) > 0:
                    labels_yolox = ObjectLabels.get_labels_as_batched_tensor(obj_label_list=current_labels, format_='yolox')

                backbone_features, states = model.mdl.forward_backbone(x=ev_tensors_padded, previous_states=prev_states)
                prev_states = states
                rnn_state.save_states_and_detach(worker_id=0, states=prev_states)

                predictions, _ = model.mdl.forward_detect(backbone_features=backbone_features)

                pred_processed = postprocess(prediction=predictions,
                                            num_classes=3,
                                            conf_thre=0.1,
                                            nms_thre=0.45)
                self.visualize(ev_tensors, labels_yolox, pred_processed)
                

        

