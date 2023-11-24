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
# deepspeed 3 目前不兼容 configure_model,需要初始化加载，deepspeed 3 百川模型验证失败
source /root/venv/lightning/bin/activate
# 激活虚拟环境
deepspeed --include=localhost:6,7 --master_port=${MASTER_PORT} --hostfile="" --no_local_rank task/pretraining_lightning.py \
	--deepspeed ${LIGHTNING_DS_CONFIG_STAGE_2} \
	--overwrite_cache \
	--model_name_or_path /home/zb/model/Baichuan2-13B-Base/ \
	--output_dir /home/zb/saved_checkpoint/light_base_pt_lr1e4_1epoch \
	--train_data /home/zb/suningGit/zb/pt/pt.txt \
	--max_epochs 1 \
	--block_size 2048 \
	--warmup_proportion 0.1 \
	--per_device_train_batch_size 8 \
	--per_device_eval_batch_size 8 \
	--learning_rate 1e-4 \
	--gradient_accumulation_steps 8 \
	--preprocessing_num_workers 16 \
	--save_steps 500 \
	--use_lora true \
	--lora_target all \
	--lora_rank 8 \
	--lora_alpha 16 \
	--use_slow_tokenizer \
	--lr_scheduler_type cosine \
	--low_cpu_mem_usage
