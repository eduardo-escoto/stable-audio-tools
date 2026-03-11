export PROJECT_ROOT="/home/eduardo/projects/synth_audio"

export DATASET_CONFIG="${PROJECT_ROOT}/configs/datasets/jamendo.json"

export MODEL_CONFIG="${PROJECT_ROOT}/configs/models/saosmall_finetune.json"
export PRETRAINED_CKPT_PATH="${PROJECT_ROOT}/.cache/models--stabilityai--stable-audio-open-small/snapshots/dc620d91535857b72ebb59b4ca45978db6d417f5/base_model.ckpt"

export SAVE_DIR="${PROJECT_ROOT}/runs"

export CKPT_PATH=""
export ENABLE_TORCH_COMPILE="1"
export USE_CHECKPOINTING="0"
export USE_LORA='false'

export NAME="jamendo_sao_small_finetune"
export CUDA_VISIBLE_DEVICES=2,3
export PYTHONWARNINGS="ignore"
uv run \
    python stable-audio-tools/train.py \
        --model-config $MODEL_CONFIG \
        --dataset-config $DATASET_CONFIG \
        --config-file "${PROJECT_ROOT}/stable-audio-tools/defaults.ini" \
        --pretrained-ckpt-path $PRETRAINED_CKPT_PATH \
        --save-dir $SAVE_DIR \
        --num-workers 32 --strategy ddp_find_unused_parameters_true --accum-batches 1 \
        --batch-size 128 --checkpoint-every 5000 --precision 16-mixed --name $NAME --use-lora $USE_LORA # --ckpt-path $CKPT_PATH \

