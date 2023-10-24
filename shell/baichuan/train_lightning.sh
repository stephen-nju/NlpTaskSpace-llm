# 先激活环境
export PROJECT_PATH=/home/zb/NlpTaskSpace-llm/

cd ${PROJECT_PATH}

export PYTHONPATH=${PROJECT_PATH}

export MODEL_PATH=/data/SHARE/MODELS/BAICHUAN/Baichuan-7B/

export TRAIN_DATA=/home/zb/train_data/performance_test/dev.json

export VALID_DATA=/home/zb/train_data/performance_test/dev.json
export DS_CONFIG_STAGE_3=/home/zb/NlpTaskSpace-llm/config/lightning_deepspeed/zero_stage3_config.json

# CUDA_VISIBLE_DEVICES=3 python3 task/baichuan/sft_lightning.py \
# 	--deepspeed ${DS_CONFIG_STAGE_3} \
# 	--model_name_or_path ${MODEL_PATH} \
# 	--train_data ${TRAIN_DATA} \
# 	--dev_data ${VALID_DATA} \
# 	--max_epochs 4 \
# 	--overwrite_cache \
# 	--output_dir output/baichuan-ner-128 \
# 	--max_source_length 64 \
# 	--max_target_length 64 \
# 	--num_warmup_steps 1000 \
# 	--per_device_train_batch_size 8 \
# 	--per_device_eval_batch_size 8 \
# 	--learning_rate 2e-5 \
# 	--low_cpu_mem_usage \
# 	--use_lora true \
# 	--lora_target W_pack

MASTER_PORT=$(shuf -n 1 -i 10000-65535)

deepspeed --include=localhost:4,5 --master_port=${MASTER_PORT} --hostfile="" --no_local_rank task/baichuan/sft_lightning.py \
	--deepspeed ${DS_CONFIG_STAGE_3} \
	--template_name "baichuan2" \
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
	--low_cpu_mem_usage \
	--use_lora true \
	--save_steps 100 \
	--lora_target W_pack
