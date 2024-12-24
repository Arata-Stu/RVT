#!/bin/bash

# 実行に必要なディレクトリやパラメータ
DATA_DIR="/mnt/2TB_ssd/gen1/"
DEST_DIR="/media/arata/AT_2TB/pre_gen1"
NUM_PROCESSES=20

# const_duration.yaml のリスト
CONST_DURATION_YAMLS=(
    "conf_preprocess/extraction/const_duration_5.yaml"
    "conf_preprocess/extraction/const_duration_10.yaml"
    "conf_preprocess/extraction/const_duration_20.yaml"
    # "conf_preprocess/extraction/const_duration_33.yaml"
    "conf_preprocess/extraction/const_duration_50.yaml"
    "conf_preprocess/extraction/const_duration_100.yaml"
)

# event_frame.yaml のリスト
EVENT_FRAME_YAMLS=(
    "conf_preprocess/representation/event_frame.yaml"
    "conf_preprocess/representation/stacked_hist.yaml"
)

# 繰り返し処理
for EVENT_FRAME in "${EVENT_FRAME_YAMLS[@]}"; do
    for CONST_DURATION in "${CONST_DURATION_YAMLS[@]}"; do
        echo "Running with ${EVENT_FRAME} and ${CONST_DURATION}..."
        python preprocess_dataset.py "${DATA_DIR}" "${DEST_DIR}" \
            "${EVENT_FRAME}" "${CONST_DURATION}" \
            conf_preprocess/filter_gen4.yaml -ds gen1 -np "${NUM_PROCESSES}"

        if [ $? -ne 0 ]; then
            echo "Error occurred while processing ${EVENT_FRAME} and ${CONST_DURATION}. Exiting..."
            exit 1
        fi
    done
done

echo "All tasks completed successfully."
