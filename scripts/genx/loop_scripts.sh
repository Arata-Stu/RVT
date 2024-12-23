#!/bin/bash

# 実行に必要なディレクトリやパラメータ
DATA_DIR="/path/to/data_dir"
DEST_DIR="/path/to/dest_dir"
NUM_PROCESSES=10

# const_duration.yaml のリスト
CONST_DURATION_YAMLS=(
    "conf_preprocess/extraction/const_duration_5.yaml"
    "conf_preprocess/extraction/const_duration_10.yaml"
    "conf_preprocess/extraction/const_duration_20.yaml"
    "conf_preprocess/extraction/const_duration_33.yaml"
    "conf_preprocess/extraction/const_duration_50.yaml"
    "conf_preprocess/extraction/const_duration_100.yaml"
)

# 繰り返し処理
for CONST_DURATION in "${CONST_DURATION_YAMLS[@]}"; do
    echo "Running with ${CONST_DURATION}..."
    python preprocess_dataset.py "${DATA_DIR}" "${DEST_DIR}" \
        conf_preprocess/representation/event_frame.yaml "${CONST_DURATION}" \
        conf_preprocess/filter_gen4.yaml -ds gen4 -np "${NUM_PROCESSES}"

    if [ $? -ne 0 ]; then
        echo "Error occurred while processing ${CONST_DURATION}. Exiting..."
        exit 1
    fi
done

echo "All tasks completed successfully."
