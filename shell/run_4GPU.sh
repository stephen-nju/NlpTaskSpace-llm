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

# 运行baichuan2 13b的全量微调，全量微调需要四张卡
deepspeed --include=localhost:0,1,2,3,4,5,6,7 --master_port=${MASTER_PORT} --hostfile="" task/baichuan/sft_baichuan2_13b_trainer.py \
	--do_train \
	--report_to tensorboard \
	--overwrite_output_dir \
	--train_file /home/zb/train_data/baichuan_sft/multi_task/train.json \
	--validation_file /home/zb/train_data/baichuan_sft/multi_task/dev.json \
	--num_train_epochs 1 \
	--overwrite_cache \
	--model_name_or_path ${BAICHUAN2_13B} \
	--output_dir /home/zb/saved_checkpoint/baichuan_13b_multi_task_full_tuning_v1 \
	--max_source_length 1024 \
	--max_target_length 128 \
	--warmup_ratio 0.1 \
	--logging_steps 100 \
	--lr_scheduler_type cosine \
	--per_device_train_batch_size 4 \
	--per_device_eval_batch_size 4 \
	--learning_rate 2e-5 \
	--deepspeed ${DS_CONFIG} \
	--bf16 true \
	--tf32 true \
	>/home/zb/saved_checkpoint/logs/baichuan_13b_multi_task_full_tuning_v1.txt
