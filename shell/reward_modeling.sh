
export PROJECT_PATH=/home/zb/NlpTaskSpace-llm/
cd ${PROJECT_PATH}

# 当前版本的百川模型，在验证阶段无法使用deepspeed stage3 ,优先使用stage2
export PYTHONPATH=${PROJECT_PATH}
export TOKENIZERS_PARALLELISM=false
export DS_CONFIG_STAGE_3=/home/zb/NlpTaskSpace-llm/config/deepspeed/zero_stage3_config.json
export DS_CONFIG_STAGE_2=/home/zb/NlpTaskSpace-llm/config/deepspeed/zero_stage2_config.json

MASTER_PORT=$(shuf -n 1 -i 10000-65535)
# deepspeed 3 目前不兼容 configure_model,需要初始化加载，deepspeed 3 百川模型验证失败
deepspeed --include=localhost:4,5 --master_port=${MASTER_PORT} --hostfile="" --no_local_rank task/reward_modeling_hf.py \
	--deepspeed ${DS_CONFIG_STAGE_3} \
	--template_name qwen \
	--do_train \
	--overwrite_cache \
	--overwrite_output_dir \
	--model_name_or_path /home/zb/model/Qwen/Qwen-14B/ \
	--output_dir /home/zb/saved_checkpoint/base_sn_v5_lr5e5_epoch2_lora_all \
	--train_data /home/zb/suningGit/zb/train_data/v5/train/ \
	--dev_data /home/zb/suningGit/zb/train_data/v5/dev/ \
	--num_train_epochs 1 \
	--max_source_length 1024 \
	--max_target_length 2048 \
	--per_device_train_batch_size 4 \
	--per_device_eval_batch_size 4 \
	--learning_rate 5e-5 \
	--warmup_ratio 0.1 \
	--gradient_accumulation_steps 1 \
	--preprocessing_num_workers 16 \
	--save_steps 500 \
	--use_peft true \
	--lora_target all \
	--lora_rank 8 \
	--lr_scheduler_type cosine
