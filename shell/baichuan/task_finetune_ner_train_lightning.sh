# 先激活环境
export PROJECT_PATH=/home/zb/code/NlpTaskSpace-llm/

cd ${PROJECT_PATH}

export PYTHONPATH=${PROJECT_PATH}

export MODEL_PATH=/data/SHARE/MODELS/BAICHUAN/Baichuan-7B/

export TRAIN_DATA=/home/zb/train_data/baichuan/dev.json
export VALID_DATA=/home/zb/train_data/baichuan/dev.json

CUDA_VISIBLE_DEVICES=2,3 python3 task/finetune_baichuan_7b_lora_ner_lightning.py \
	--model_name_or_path ${MODEL_PATH} \
	--train_data ${TRAIN_DATA} \
	--dev_data ${VALID_DATA} \
	--max_epochs 1 \
	--overwrite_cache \
	--output_dir output/baichuan-ner-128 \
	--max_source_length 64 \
	--max_target_length 64 \
	--num_warmup_steps 1000 \
	--per_device_train_batch_size 8 \
	--per_device_eval_batch_size 8 \
	--learning_rate 2e-5 \
	--low_cpu_mem_usage
