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
export BAICHUAN_7B=/data/SHARE/MODELS/BAICHUAN/Baichuan-7B/
export BAICHUAN2_13B=/data/SHARE/MODELS/BAICHUAN/Baichuan2-13B-Base/
export DS_CONFIG=/home/zb/NlpTaskSpace-llm/config/deepspeed/zero_stage3_config.json
# 运行baichuan1 7b的lora模型
MASTER_PORT=$(shuf -n 1 -i 10000-65535)

# 分布式训练
# CUDA_VISIBLE_DEVICES=4,5 accelerate launch --num_processes 2 --main_process_port=${MASTER_PORT} task/baichuan/sft_baichuan2_13b_lora_trainer.py \
# 	--do_train \
# 	--model_name_or_path ${BAICHUAN_7B} \
# 	--report_to tensorboard \
# 	--overwrite_output_dir \
# 	--overwrite_cache \
# 	--train_file /home/zb/train_data/baichuan_sft/multi_task/train.json \
# 	--validation_file /home/zb/train_data/baichuan_sft/multi_task/dev.json \
# 	--num_train_epochs 1 \
# 	--overwrite_cache \
# 	--output_dir /home/zb/saved_checkpoint/baichuan_7b_mulit_task_lora_v1 \
# 	--use_lora true \
# 	--lora_target W_pack \
# 	--max_source_length 1024 \
# 	--max_target_length 128 \
# 	--warmup_ratio 0.1 \
# 	--logging_steps 100 \
# 	--per_device_train_batch_size 8 \
# 	--per_device_eval_batch_size 8 \
# 	--save_steps 1000 \
# 	--save_total_limit=2 \
# 	--learning_rate 2e-5 \
# 	--bf16 true \
# 	--tf32 true
# >/home/zb/saved_checkpoint/logs/baichuan_7b_mulit_task_lora_v1.txt

deepspeed --include=localhost:4,5 --master_port=${MASTER_PORT} --hostfile="" task/baichuan/sft_baichuan2_13b_lora_trainer.py \
	--deepspeed ${DS_CONFIG} \
	--do_train \
	--model_name_or_path ${BAICHUAN_7B} \
	--report_to tensorboard \
	--overwrite_output_dir \
	--overwrite_cache \
	--train_file /home/zb/train_data/baichuan_sft/multi_task/train.json \
	--validation_file /home/zb/train_data/baichuan_sft/multi_task/dev.json \
	--num_train_epochs 1 \
	--overwrite_cache \
	--output_dir /home/zb/saved_checkpoint/baichuan_7b_mulit_task_lora_v1 \
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
	>/home/zb/saved_checkpoint/logs/baichuan_7b_mulit_task_lora_v1.txt

# 评测训练结果
wait
# 运行baichuan2 13b的lora第一版本模型

MASTER_PORT=$(shuf -n 1 -i 10000-65535)
deepspeed --include=localhost:4,5 --master_port=${MASTER_PORT} --hostfile="" task/baichuan/sft_baichuan2_13b_lora_trainer.py \
	--deepspeed ${DS_CONFIG} \
	--do_train \
	--report_to tensorboard \
	--overwrite_cache \
	--overwrite_output_dir \
	--train_file /home/zb/train_data/baichuan_sft/multi_task/train.json \
	--validation_file /home/zb/train_data/baichuan_sft/multi_task/dev.json \
	--num_train_epochs 1 \
	--overwrite_cache \
	--model_name_or_path ${BAICHUAN2_13B} \
	--output_dir /home/zb/saved_checkpoint/baichuan_13b_multi_task_lora_v1 \
	--use_lora true \
	--lora_target W_pack \
	--max_source_length 1024 \
	--max_target_length 128 \
	--warmup_ratio 0.1 \
	--logging_steps 100 \
	--per_device_train_batch_size 8 \
	--per_device_eval_batch_size 8 \
	--learning_rate 2e-5 \
	--save_steps 1000 \
	--save_total_limit=2 \
	--bf16 true \
	--tf32 true \
	>/home/zb/saved_checkpoint/logs/baichuan_13b_multi_task_lora_v1.txt

# 运行baichuan2 13b lora 第二版本
wait

MASTER_PORT=$(shuf -n 1 -i 10000-65535)
deepspeed --include=localhost:4,5 --master_port=${MASTER_PORT} --hostfile="" task/baichuan/sft_baichuan2_13b_lora_trainer.py \
	--deepspeed ${DS_CONFIG} \
	--do_train \
	--report_to tensorboard \
	--overwrite_cache \
	--overwrite_output_dir \
	--train_file /home/zb/train_data/baichuan_sft/multi_task/train.json \
	--validation_file /home/zb/train_data/baichuan_sft/multi_task/dev.json \
	--num_train_epochs 1 \
	--overwrite_cache \
	--model_name_or_path ${BAICHUAN2_13B} \
	--output_dir /home/zb/saved_checkpoint/baichuan_13b_multi_task_lora_v2 \
	--use_lora true \
	--lora_target W_pack,o_proj,gate_proj \
	--max_source_length 1024 \
	--max_target_length 128 \
	--warmup_ratio 0.1 \
	--logging_steps 100 \
	--per_device_train_batch_size 8 \
	--per_device_eval_batch_size 8 \
	--learning_rate 2e-5 \
	--save_steps 1000 \
	--save_total_limit=2 \
	--bf16 true \
	--tf32 true \
	>/home/zb/saved_checkpoint/logs/baichuan_13b_multi_task_lora_v2.txt

# 单任务测试

wait
MASTER_PORT=$(shuf -n 1 -i 10000-65535)
deepspeed --include=localhost:4,5 --master_port=${MASTER_PORT} --hostfile="" task/baichuan/sft_baichuan2_13b_lora_trainer.py \
	--deepspeed ${DS_CONFIG} \
	--do_train \
	--report_to tensorboard \
	--overwrite_cache \
	--overwrite_output_dir \
	--train_file /home/zb/train_data/baichuan_sft/single_task_sn/train.json \
	--validation_file /home/zb/train_data/baichuan_sft/single_task_sn/dev.json \
	--num_train_epochs 1 \
	--overwrite_cache \
	--model_name_or_path ${BAICHUAN2_13B} \
	--output_dir /home/zb/saved_checkpoint/baichuan_13b_single_task_sn_lora_v1 \
	--use_lora true \
	--lora_target W_pack,o_proj,gate_proj \
	--max_source_length 1024 \
	--max_target_length 128 \
	--warmup_ratio 0.1 \
	--logging_steps 100 \
	--per_device_train_batch_size 8 \
	--per_device_eval_batch_size 8 \
	--learning_rate 2e-5 \
	--save_steps 1000 \
	--save_total_limit=2 \
	--bf16 true \
	--tf32 true \
	>/home/zb/saved_checkpoint/logs/baichuan_13b_single_task_sn_lora_v1.txt
# 不同量级的训练数据测试

wait
#少量数据训练
MASTER_PORT=$(shuf -n 1 -i 10000-65535)
deepspeed --include=localhost:4,5 --master_port=${MASTER_PORT} --hostfile="" task/baichuan/sft_baichuan2_13b_lora_trainer.py \
	--deepspeed ${DS_CONFIG} \
	--do_train \
	--report_to tensorboard \
	--overwrite_cache \
	--overwrite_output_dir \
	--train_file /home/zb/train_data/baichuan_sft/single_task_jd/train.json \
	--validation_file /home/zb/train_data/baichuan_sft/train_size/dev.json \
	--num_train_epochs 1 \
	--overwrite_cache \
	--model_name_or_path ${BAICHUAN2_13B} \
	--output_dir /home/zb/saved_checkpoint/baichuan_13b_single_task_jd_lora_v1 \
	--use_lora true \
	--lora_target W_pack \
	--max_source_length 1024 \
	--max_target_length 128 \
	--warmup_ratio 0.1 \
	--logging_steps 100 \
	--per_device_train_batch_size 8 \
	--per_device_eval_batch_size 8 \
	--learning_rate 2e-5 \
	--save_steps 1000 \
	--save_total_limit=2 \
	--bf16 true \
	--tf32 true \
	>/home/zb/saved_checkpoint/logs/baichuan_13b_single_task_jd_lora_v1.txt

wait
# medium 级别的数据训练
MASTER_PORT=$(shuf -n 1 -i 10000-65535)
deepspeed --include=localhost:4,5 --master_port=${MASTER_PORT} --hostfile="" task/baichuan/sft_baichuan2_13b_lora_trainer.py \
	--deepspeed ${DS_CONFIG} \
	--do_train \
	--report_to tensorboard \
	--overwrite_cache \
	--overwrite_output_dir \
	--train_file /home/zb/train_data/baichuan_sft/train_size/medium/train.json \
	--validation_file /home/zb/train_data/baichuan_sft/train_size/dev.json \
	--num_train_epochs 1 \
	--overwrite_cache \
	--model_name_or_path ${BAICHUAN2_13B} \
	--output_dir /home/zb/saved_checkpoint/baichuan_13b_medium_lora_v1 \
	--use_lora true \
	--lora_target W_pack \
	--max_source_length 1024 \
	--max_target_length 128 \
	--warmup_ratio 0.1 \
	--logging_steps 100 \
	--per_device_train_batch_size 8 \
	--per_device_eval_batch_size 8 \
	--learning_rate 2e-5 \
	--save_steps 1000 \
	--save_total_limit=2 \
	--bf16 true \
	--tf32 true \
	>/home/zb/saved_checkpoint/logs/baichuan_13b_medium_lora_v1.txt

wait
# 全量数据训练
MASTER_PORT=$(shuf -n 1 -i 10000-65535)
deepspeed --include=localhost:4,5 --master_port=${MASTER_PORT} --hostfile="" task/baichuan/sft_baichuan2_13b_lora_trainer.py \
	--deepspeed ${DS_CONFIG} \
	--do_train \
	--report_to tensorboard \
	--overwrite_cache \
	--overwrite_output_dir \
	--train_file /home/zb/train_data/baichuan_sft/train_size/all/train.json \
	--validation_file /home/zb/train_data/baichuan_sft/train_size/dev.json \
	--num_train_epochs 1 \
	--overwrite_cache \
	--model_name_or_path ${BAICHUAN2_13B} \
	--output_dir /home/zb/saved_checkpoint/baichuan_13b_all_lora_v1 \
	--use_lora true \
	--lora_target W_pack \
	--max_source_length 1024 \
	--max_target_length 128 \
	--warmup_ratio 0.1 \
	--logging_steps 100 \
	--per_device_train_batch_size 8 \
	--per_device_eval_batch_size 8 \
	--learning_rate 2e-5 \
	--save_steps 1000 \
	--save_total_limit=2 \
	--bf16 true \
	--tf32 true \
	>/home/zb/saved_checkpoint/logs/baichuan_13b_all_lora_v1.txt
