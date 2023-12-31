# 先激活环境
export PROJECT_PATH=/home/zb/NlpTaskSpace-llm/
cd ${PROJECT_PATH}
export PYTHONPATH=${PROJECT_PATH}
export DS_CONFIG_STAGE_3=/home/zb/NlpTaskSpace-llm/config/lightning_deepspeed/zero_stage3_config.json
export DS_CONFIG_STAGE_2=/home/zb/NlpTaskSpace-llm/config/lightning_deepspeed/zero_stage2_config.json

MASTER_PORT=$(shuf -n 1 -i 10000-65535)
export TOKENIZERS_PARALLELISM=false
export HF_ENDPOINT=https://hf-mirror.com
export MODEL_PATH=/home/zb/model/Qwen/Qwen-14B/

# v4_plus 版本训练
export TRAIN_DATA="/home/zb/suningGit/zb/train_data/v4_plus/sn_generate_gpt_v4_plus_train.json"
export DEV_DATA="/home/zb/suningGit/zb/train_data/v1/sn_generate_gpt_v1_dev.json"

# deepspeed error RuntimeError: torch.cat(): expected a non-empty list of Tensors
# CUDA_VISIBLE_DEVICES=2,3,4,5 python task/qwen/supervised_finetuning_lightning.py \
# 	--overwrite_cache \
# 	--model_name_or_path ${MODEL_PATH} \
# 	--output_dir /home/zb/saved_checkpoint/light_qwen_sft_sn_v5_tiger_lr2e4_1epoch \
# 	--train_data /home/zb/suningGit/zb/train_data/v5/train/,/home/zb/suningGit/zb/train_data/tigerbot_sft_zh \
# 	--dev_data /home/zb/suningGit/zb/train_data/v5/dev/ \
# 	--max_epochs 1 \
# 	--max_source_length 1024 \
# 	--max_target_length 2048 \
# 	--warmup_ratio 0.1 \
# 	--per_device_train_batch_size 4 \
# 	--per_device_eval_batch_size 4 \
# 	--learning_rate 1e-4 \
# 	--gradient_accumulation_steps 8 \
# 	--preprocessing_num_workers 16 \
# 	--save_steps 5000 \
# 	--use_lora true \
# 	--lora_target all \
# 	--lora_rank 8 \
# 	--lora_alpha 16 \
# 	--use_slow_tokenizer \
# 	--lr_scheduler_type cosine \
# 	--low_cpu_mem_usage

# source /root/venv/lightning/bin/activate
# # 激活虚拟环境
deepspeed --include=localhost:4,5 --master_port=${MASTER_PORT} --hostfile="" --no_local_rank task/qwen/supervised_finetuning_lightning.py \
	--deepspeed ${DS_CONFIG_STAGE_3} \
	--overwrite_cache \
	--model_name_or_path ${MODEL_PATH} \
	--output_dir /home/zb/saved_checkpoint/light_test \
	--train_data /home/zb/suningGit/zb/train_data/v5/train/ \
	--dev_data /home/zb/suningGit/zb/train_data/v5/dev/ \
	--max_epochs 1 \
	--max_source_length 1024 \
	--max_target_length 2048 \
	--warmup_ratio 0.1 \
	--per_device_train_batch_size 8 \
	--per_device_eval_batch_size 8 \
	--learning_rate 1e-4 \
	--gradient_accumulation_steps 1 \
	--preprocessing_num_workers 16 \
	--save_steps 5000 \
	--use_lora true \
	--lora_target all \
	--lora_rank 8 \
	--lora_alpha 16 \
	--use_slow_tokenizer \
	--lr_scheduler_type cosine \
	--low_cpu_mem_usage
