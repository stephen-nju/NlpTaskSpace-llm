# 先激活环境
export PROJECT_PATH=/home/zhubin/code/NlpTaskSpace-llm/

cd ${PROJECT_PATH}

export PYTHONPATH=${PROJECT_PATH}

export MODEL_PATH=/home/zhubin/model/Baichuan-7B/

export TRAIN_DATA=/home/zhubin/train_data/baichuan/train.json
export VALID_DATA=/home/zhubin/train_data/baichuan/dev.json

CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes=2 task/finetune_baichuan_7b_lora_ner_accelerate.py \
	--train_file ${TRAIN_DATA} \
	--validation_file ${VALID_DATA} \
	--num_train_epochs 2 \
	--overwrite_cache \
	--model_name_or_path ${MODEL_PATH} \
	--output_dir output/baichuan-ner-128 \
	--max_source_length 64 \
	--max_target_length 64 \
	--num_warmup_steps 1000 \
	--per_device_train_batch_size 8 \
	--per_device_eval_batch_size 8 \
	--learning_rate 2e-5 \
	--low_cpu_mem_usage
