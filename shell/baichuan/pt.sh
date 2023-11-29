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
deepspeed --include=localhost:0,1,2,5 --master_port=${MASTER_PORT} --hostfile="" --no_local_rank task/pretraining_lightning.py \
	--deepspeed ${LIGHTNING_DS_CONFIG_STAGE_2} \
	--overwrite_cache \
	--model_name_or_path /home/zb/model/Baichuan2-13B-Base/ \
	--output_dir /home/zb/saved_checkpoint/light_baichuan_base_pt_lr1e4_2epoch \
	--train_data /home/zb/suningGit/zb/pt/v1/ \
	--dev_data /home/zb/suningGit/zb/pt/dev/ \
	--max_epochs 2 \
	--block_size 2048 \
	--warmup_proportion 0.1 \
	--per_device_train_batch_size 8 \
	--per_device_eval_batch_size 8 \
	--learning_rate 1e-4 \
	--gradient_accumulation_steps 1 \
	--preprocessing_num_workers 16 \
	--save_steps 500 \
	--use_lora true \
	--lora_target all \
	--lora_rank 8 \
	--lora_alpha 16 \
	--use_slow_tokenizer \
	--lr_scheduler_type cosine \
	--no_keep_linebreaks \
	--low_cpu_mem_usage

# pt hf trainer

# source activate baichuan13b

# deepspeed --include=localhost:0,1,5 --master_port=${MASTER_PORT} --hostfile="" --no_local_rank task/pretraining_hf.py \
# 	--deepspeed ${DS_CONFIG_STAGE_2} \
# 	--do_train \
# 	--do_eval \
# 	--model_type baichuan \
# 	--model_name_or_path ${BAICHUAN2_13B} \
# 	--train_file_dir /home/zb/suningGit/zb/pt/train_v1/ \
# 	--validation_file_dir /home/zb/suningGit/zb/pt/dev_v1/ \
# 	--report_to tensorboard \
# 	--overwrite_output_dir \
# 	--overwrite_cache \
# 	--output_dir /home/zb/saved_checkpoint/base_pt_1epoch_lr1e4 \
# 	--num_train_epochs 1 \
# 	--overwrite_cache \
# 	--block_size 1024 \
# 	--use_peft true \
# 	--target_modules all \
# 	--lora_rank 8 \
# 	--lora_alpha 16 \
# 	--eval_steps 100 \
# 	--evaluation_strategy steps \
# 	--warmup_ratio 0.1 \
# 	--logging_steps 10 \
# 	--preprocessing_num_workers 16 \
# 	--lr_scheduler_type cosine \
# 	--per_device_train_batch_size 8 \
# 	--per_device_eval_batch_size 8 \
# 	--save_steps 200 \
# 	--save_total_limit=2 \
# 	--learning_rate 1e-4 \
# 	--torch_dtype float16 \
# 	--bf16 true
