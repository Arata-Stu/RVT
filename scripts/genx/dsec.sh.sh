#!/bin/bash

# 実行に必要なディレクトリやパラメータ
DATA_DIR="/mnt/2TB_ssd/DSEC/"
BASE_DEST_DIR="/mnt/2TB_ssd/pre_dsec"
NUM_PROCESSES=5

# DURATION=("5" "10" "20" "50" "100")  # 配列から括弧を取り除く
DURATION=("5")
CONST_DURATION_DIR="conf_preprocess/extraction"

# event_frame.yaml のリスト
EVENT_FRAME_YAMLS=(
    "conf_preprocess/representation/event_frame.yaml"
    # "conf_preprocess/representation/stacked_hist.yaml"
)

# 繰り返し処理
for EVENT_FRAME in "${EVENT_FRAME_YAMLS[@]}"; do
    for DURATION_VALUE in "${DURATION[@]}"; do
        # 動的に const_duration.yaml のパスを生成
        CONST_DURATION="${CONST_DURATION_DIR}/const_duration_${DURATION_VALUE}.yaml"
        
        if [ ! -f "${CONST_DURATION}" ]; then
            echo "Warning: ${CONST_DURATION} does not exist. Skipping..."
            continue
        fi

        # 保存先ディレクトリを動的に設定
        DEST_DIR="${BASE_DEST_DIR}_${DURATION_VALUE}"

        # 実行ログ
        echo "Running with ${EVENT_FRAME} and ${CONST_DURATION}, saving to ${DEST_DIR}..."

        python preprocess_dataset_dsec.py "${DATA_DIR}" "${DEST_DIR}" \
            "${EVENT_FRAME}" "${CONST_DURATION}" \
            conf_preprocess/filter_dsec.yaml -ds dsec -np "${NUM_PROCESSES}"

        if [ $? -ne 0 ]; then
            echo "Error occurred while processing ${EVENT_FRAME} and ${CONST_DURATION}. Exiting..."
            exit 1
        fi
    done
done

echo "All tasks completed successfully."
