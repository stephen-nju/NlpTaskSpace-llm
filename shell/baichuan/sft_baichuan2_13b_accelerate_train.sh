# 先激活环境
export PROJECT_PATH=/home/zb/NlpTaskSpace-llm/

cd ${PROJECT_PATH}

export PYTHONPATH=${PROJECT_PATH}

export MODEL_PATH=/data/SHARE/MODELS/BAICHUAN/Baichuan2-13B-Base/

export TRAIN_DATA=/home/zb/train_data/baichuan_sft/train.json

export VALID_DATA=/home/zb/train_data/baichuan_sft/dev.json

CUDA_VISIBLE_DEVICES=5,6 accelerate launch --num_processes=2 --mixed_precision=fp16 --main_process_port=29601 task/baichuan/sft_baichuan2_13b_lora_accelerate.py \
	--train_file ${TRAIN_DATA} \
	--validation_file ${VALID_DATA} \
	--num_train_epochs 1 \
	--overwrite_cache \
	--model_name_or_path ${MODEL_PATH} \
	--output_dir output/baichuan-sft-128-v2 \
	--max_source_length 1024 \
	--max_target_length 128 \
	--num_warmup_steps 1000 \
	--per_device_train_batch_size 8 \
	--per_device_eval_batch_size 8 \
	--learning_rate 2e-5 \
	--with_tracking \
	--report_to tensorboard \
	--low_cpu_mem_usage
