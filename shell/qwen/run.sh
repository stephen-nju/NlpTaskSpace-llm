# 先激活环境
export PROJECT_PATH=/home/zb/NlpTaskSpace-llm/
cd ${PROJECT_PATH}
export PYTHONPATH=${PROJECT_PATH}
export DS_CONFIG_STAGE_3=/home/zb/NlpTaskSpace-llm/config/lightning_deepspeed/zero_stage3_config.json
export DS_CONFIG_STAGE_2=/home/zb/NlpTaskSpace-llm/config/lightning_deepspeed/zero_stage2_config.json

MASTER_PORT=$(shuf -n 1 -i 10000-65535)
export TOKENIZERS_PARALLELISM=false
export MODEL_PATH=/home/zb/model/Qwen-14B/

# v4_plus 版本训练
export TRAIN_DATA="/home/zb/suningGit/zb/train_data/v4_plus/sn_generate_gpt_v4_plus_train.json"
export DEV_DATA="/home/zb/suningGit/zb/train_data/v1/sn_generate_gpt_v1_dev.json"

deepspeed --include=localhost:6,7 --master_port=${MASTER_PORT} --hostfile="" --no_local_rank task/qwen/supervised_finetuning_lightning_deepspeed.py \
	--deepspeed ${DS_CONFIG_STAGE_2} \
	--overwrite_cache \
	--model_name_or_path ${MODEL_PATH} \
	--output_dir /home/zb/saved_checkpoint/qwen_chat_sn_v4_plus_alpaca_2epoch_1e4 \
	--train_data ${TRAIN_DATA} \
	--dev_data ${DEV_DATA} \
	--max_epochs 2 \
	--max_source_length 512 \
	--max_target_length 512 \
	--warmup_proportion 0.1 \
	--per_device_train_batch_size 1 \
	--per_device_eval_batch_size 1 \
	--learning_rate 1e-4 \
	--gradient_accumulation_steps 16 \
	--preprocessing_num_workers 16 \
	--save_steps 500 \
	--use_lora true \
	--lora_target all \
	--lora_rank 8 \
	--lr_scheduler_type cosine \
	--low_cpu_mem_usage
