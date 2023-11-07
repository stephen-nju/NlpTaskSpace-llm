# 先激活环境
export PROJECT_PATH=/home/zb/NlpTaskSpace-llm/
cd ${PROJECT_PATH}
export PYTHONPATH=${PROJECT_PATH}
export DS_CONFIG_STAGE_3=/home/zb/NlpTaskSpace-llm/config/lightning_deepspeed/zero_stage3_config.json
export DS_CONFIG_STAGE_2=/home/zb/NlpTaskSpace-llm/config/lightning_deepspeed/aquila2-chat-34b/zero_stage2_config.json
MASTER_PORT=$(shuf -n 1 -i 10000-65535)
export TOKENIZERS_PARALLELISM=false
export MODEL_PATH=/data/SHARE/MODELS/BAAI/AquilaChat2-34B/

export TEST_DATA=/home/zb/train_data/baichuan_sft/single_task_sn_v2/split_task_v2/test.json

deepspeed --include=localhost:6,7 --master_port=${MASTER_PORT} --hostfile="" --no_local_rank task/aquila2/sft_deepspeed.py \
	--deepspeed ${DS_CONFIG_STAGE_2} \
	--overwrite_cache \
	--model_name_or_path ${MODEL_PATH} \
	--output_dir /home/zb/saved_checkpoint/aquila2-test \
	--train_data ${TEST_DATA} \
	--dev_data ${TEST_DATA} \
	--max_epochs 1 \
	--max_source_length 1024 \
	--max_target_length 1024 \
	--warmup_proportion 0.1 \
	--per_device_train_batch_size 2 \
	--per_device_eval_batch_size 2 \
	--learning_rate 2e-5 \
	--gradient_accumulation_steps 4 \
	--save_steps 500 \
	--use_lora true \
	--lora_target q_proj,v_proj \
	--use_slow_tokenizer \
	--lr_scheduler_type cosine \
	--low_cpu_mem_usage
