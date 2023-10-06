###
# @Author: stephen
# @Date: 2023-09-27 14:30:44
# @FilePath: \shells\run.sh
# @Description:
#
# Copyright (c) 2023 by ${git_name}, All Rights Reserved.
###

export PROJECT_PATH=/home/zb/NlpTaskSpace-llm/
cd ${PROJECT_PATH}

export PYTHONPATH=${PROJECT_PATH}
export BAICHUAN_7B=/data/SHARE/MODELS/BAICHUAN/Baichuan-7B
export BAICHUAN2_13B=/data/SHARE/MODELS/BAICHUAN/Baichuan2-13B-Base/
export DS_CONFIG=/home/zb/NlpTaskSpace-llm/config/deepspeed/zero_stage3_config.json
# 运行baichuan1 7b的lora模型
MASTER_PORT=$(shuf -n 1 -i 10000-65535)

# 持续预训练加上lora
# deepspeed --include=localhost:0,1,2,3,4,5,6,7 --master_port=29503 --hostfile="" task/baichuan/pt_baichuan2_13b_trainer.py \
# 	--deepspeed ${DS_CONFIG} \
# 	--do_train \
# 	--report_to tensorboard \
# 	--overwrite_output_dir \
# 	--overwrite_cache \
# 	--output_dir /home/zb/saved_checkpoint/baichuan_13b_pt_v1 \
# 	--train_file /home/zb/train_data/baichuan_pt/train.json \
# 	--validation_file /home/zb/train_data/baichuan_sft/single_task_jd/dev.json \
# 	--num_train_epochs 1 \
# 	--overwrite_cache \
# 	--model_name_or_path ${BAICHUAN2_13B} \
# 	--max_source_length 1024 \
# 	--max_target_length 128 \
# 	--block_size 1024 \
# 	--warmup_ratio 0.1 \
# 	--logging_steps 100 \
# 	--per_device_train_batch_size 4 \
# 	--per_device_eval_batch_size 4 \
# 	--lr_scheduler_type cosine \
# 	--save_steps 1000 \
# 	--save_total_limit=2 \
# 	--learning_rate 2e-5 \
# 	--bf16 true \
# 	--tf32 true \
# 	>/home/zb/saved_checkpoint/logs/baichuan_13b_pt_v1.txt

# wait
# 加载预训练完成的模型
deepspeed --include=localhost:4,5 --master_port=${MASTER_PORT} --hostfile="" task/baichuan/sft_baichuan2_13b_lora_trainer.py \
	--deepspeed ${DS_CONFIG} \
	--do_train \
	--report_to tensorboard \
	--overwrite_output_dir \
	--train_file /home/zb/train_data/baichuan_sft/multi_task/train.json \
	--validation_file /home/zb/train_data/baichuan_sft/multi_task/dev.json \
	--num_train_epochs 1 \
	--overwrite_cache \
	--model_name_or_path /home/zb/saved_checkpoint/baichuan_13b_pt_v1/checkpoint-1000/ \
	--output_dir /home/zb/saved_checkpoint/baichuan_13b_multi_task_pt_lora_v1 \
	--use_lora true \
	--lora_target W_pack \
	--max_source_length 1024 \
	--max_target_length 128 \
	--warmup_ratio 0.1 \
	--logging_steps 100 \
	--per_device_train_batch_size 8 \
	--per_device_eval_batch_size 8 \
	--save_steps 1000 \
	--save_total_limit=2 \
	--learning_rate 2e-5 \
	--bf16 true \
	--tf32 true \
	>/home/zb/saved_checkpoint/logs/baichuan_13b_multi_task_pt_lora_v1.txt
