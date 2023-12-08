# 先激活环境

export PROJECT_PATH=/home/zb/NlpTaskSpace-llm/
cd ${PROJECT_PATH}

# 当前版本的百川模型，在验证阶段无法使用deepspeed stage3 ,优先使用stage2
export PYTHONPATH=${PROJECT_PATH}
export BAICHUAN2_13B=/data/SHARE/MODELS/BAICHUAN/Baichuan2-13B-Base/

export TOKENIZERS_PARALLELISM=false
export TEST_DATA=/home/zb/train_data/baichuan_sft/single_task_sn_v2/split_task_v2/test.json

export DS_CONFIG_STAGE_3=/home/zb/NlpTaskSpace-llm/config/deepspeed/zero_stage3_config.json

export DS_CONFIG_STAGE_2=/home/zb/NlpTaskSpace-llm/config/deepspeed/zero_stage2_config.json

export LIGHTNING_DS_CONFIG_STAGE_3=/home/zb/NlpTaskSpace-llm/config/lightning_deepspeed/zero_stage3_config.json
export LIGHTNING_DS_CONFIG_STAGE_2=/home/zb/NlpTaskSpace-llm/config/lightning_deepspeed/zero_stage2_config.json
# 运行baichuan1 7b的lora模型
MASTER_PORT=$(shuf -n 1 -i 10000-65535)
# ddp

# deepspeed 3 目前不兼容 configure_model,需要初始化加载，deepspeed 3 百川模型验证失败
# torch 版本为2.0.1 过高的版本训练可能出问题
source /root/venv/lightning/bin/activate
# 激活虚拟环境
deepspeed --include=localhost:4,5 --master_port=${MASTER_PORT} --hostfile="" --no_local_rank task/baichuan/supervised_finetuning_lightning.py \
	--deepspeed ${LIGHTNING_DS_CONFIG_STAGE_2} \
	--overwrite_cache \
	--model_name_or_path /home/zb/saved_checkpoint/light_baichuan_base_pt_lr1e4_2epoch/merge/ \
	--output_dir /home/zb/saved_checkpoint/light_baichuan_pt_v2_sft_sn_v5_tiger_lr2e4_1epoch \
	--train_data /home/zb/suningGit/zb/train_data/v5/train/,/home/zb/suningGit/zb/train_data/tigerbot_sft_zh \
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
