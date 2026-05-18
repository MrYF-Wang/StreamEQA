#!/usr/bin/env bash
set -euo pipefail

# EVAL_MODEL=${EVAL_MODEL:-Qwen3VL}
# TOPIC=${TOPIC:-Perception}
# DATA_FILE=${DATA_FILE:-data/annotations/${TOPIC}.json}
# OUTPUT_FILE=${OUTPUT_FILE:-results/${TOPIC}_output_${EVAL_MODEL}.json}
# VIDEO_ROOT=${VIDEO_ROOT:-/path/to/videos}
# MODEL_PATH=${MODEL_PATH:-/path/to/Qwen3-VL-8B-Instruct}
# GPU=${GPU:-0}
# CUDA_VISIBLE_DEVICES=${GPU} python -m streameqa.eval \
#   --model_name "${EVAL_MODEL}" \
#   --model_path "${MODEL_PATH}" \
#   --video_root "${VIDEO_ROOT}" \
#   --data_file "${DATA_FILE}" \
#   --output_file "${OUTPUT_FILE}" 

# EVAL_MODEL=${EVAL_MODEL:-Qwen3VL}
# TOPIC=${TOPIC:-Interaction}
# DATA_FILE=${DATA_FILE:-data/annotations/${TOPIC}.json}
# OUTPUT_FILE=${OUTPUT_FILE:-results/${TOPIC}_output_${EVAL_MODEL}.json}
# VIDEO_ROOT=${VIDEO_ROOT:-/path/to/videos}
# MODEL_PATH=${MODEL_PATH:-/path/to/Qwen3-VL-8B-Instruct}
# GPU=${GPU:-0}
# CUDA_VISIBLE_DEVICES=${GPU} python -m streameqa.eval \
#   --model_name "${EVAL_MODEL}" \
#   --model_path "${MODEL_PATH}" \
#   --video_root "${VIDEO_ROOT}" \
#   --data_file "${DATA_FILE}" \
#   --output_file "${OUTPUT_FILE}" 

EVAL_MODEL=${EVAL_MODEL:-Qwen3VL}
TOPIC=${TOPIC:-Planning_all}
DATA_FILE=${DATA_FILE:-data/annotations/${TOPIC}.json}
OUTPUT_FILE=${OUTPUT_FILE:-results/${TOPIC}_output_${EVAL_MODEL}.json}
VIDEO_ROOT=${VIDEO_ROOT:-/path/to/videos}
MODEL_PATH=${MODEL_PATH:-/path/to/Qwen3-VL-8B-Instruct}
GPU=${GPU:-0}
CUDA_VISIBLE_DEVICES=${GPU} python -m streameqa.eval \
  --model_name "${EVAL_MODEL}" \
  --model_path "${MODEL_PATH}" \
  --video_root "${VIDEO_ROOT}" \
  --data_file "${DATA_FILE}" \
  --output_file "${OUTPUT_FILE}" 
