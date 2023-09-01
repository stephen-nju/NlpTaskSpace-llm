# 先激活环境

export PROJECT_PATH=/home/zhubin/code/NlpTaskSpace-llm/

cd ${PROJECT_PATH}

export PYTHONPATH=${PROJECT_PATH}

PRE_SEQ_LEN=128
LR=2e-2
export MODEL_PATH=/home/zhubin/model/chatglm-6b/
export TRAIN_DATA=/home/zhubin/train_data/AdvertiseGen/train.json
export VALID_DATA=/home/zhubin/train_data/AdvertiseGen/dev.json

CUDA_VISIBLE_DEVICES=2,3 python3 llm/chatglm-6b/lora/task_lora_hf_trainer_train.py \
    --do_train \
    --train_file ${TRAIN_DATA}\
    --validation_file ${VALID_DATA}\
    --prompt_column content \
    --response_column summary \
    --overwrite_cache \
    --model_name_or_path ${MODEL_PATH}\
    --output_dir output/adgen-chatglm-6b-pt-$PRE_SEQ_LEN-$LR \
    --overwrite_output_dir \
    --max_source_length 64 \
    --max_target_length 64 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --predict_with_generate \
    --max_steps 3000 \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate $LR \
    --pre_seq_len $PRE_SEQ_LEN \
    # --quantization_bit 4
