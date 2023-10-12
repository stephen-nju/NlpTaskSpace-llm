# 先激活环境
source activate baichuan13b
export PROJECT_PATH=/home/zb/NlpTaskSpace-llm/
cd ${PROJECT_PATH}

# 当前版本的百川模型，在验证阶段无法使用deepspeed stage3 ,优先使用stage2
export PYTHONPATH=${PROJECT_PATH}
export BAICHUAN2_13B=/data/SHARE/MODELS/BAICHUAN/Baichuan2-13B-Base/
export DS_CONFIG_STAGE_3=/home/zb/NlpTaskSpace-llm/config/deepspeed/zero_stage3_config.json

export DS_CONFIG_STAGE_2=/home/zb/NlpTaskSpace-llm/config/deepspeed/zero_stage2_config.json
# 运行baichuan1 7b的lora模型
MASTER_PORT=$(shuf -n 1 -i 10000-65535)
# ddp
# 跑通千问14b的模型
# export QWen14B=/data/SHARE/MODELS/Qwen/Qwen-14B/Qwen-14B
# deepspeed --include=localhost:4,5 --master_port=${MASTER_PORT} --hostfile="" task/baichuan/sft_lora_trainer.py \
# 	--deepspeed ${DS_CONFIG_STAGE_3} \
# 	--do_train \
# 	--model_name_or_path ${QWen14B} \
# 	--report_to tensorboard \
# 	--overwrite_output_dir \
# 	--overwrite_cache \
# 	--train_file /home/zb/train_data/baichuan_sft/single_task_sn/train.json \
# 	--validation_file /home/zb/train_data/baichuan_sft/single_task_sn/dev.json \
# 	--num_train_epochs 2 \
# 	--overwrite_cache \
# 	--output_dir /home/zb/saved_checkpoint/qwen_14b_2epoch_sn \
# 	--use_lora true \
# 	--lora_target c_attn \
# 	--max_source_length 1024 \
# 	--max_target_length 128 \
# 	--warmup_ratio 0.1 \
# 	--logging_steps 1 \
# 	--lr_scheduler_type cosine \
# 	--per_device_train_batch_size 8 \
# 	--per_device_eval_batch_size 8 \
# 	--save_steps 100 \
# 	--save_total_limit=2 \
# 	--learning_rate 2e-5 \
# 	--bf16 true \
# 	--tf32 true

# wait

# baichuan13b 过拟合数据集
deepspeed --include=localhost:4,5 --master_port=${MASTER_PORT} --hostfile="" task/baichuan/sft_lora_trainer.py \
	--deepspeed ${DS_CONFIG_STAGE_3} \
	--do_train \
	--model_name_or_path ${BAICHUAN2_13B} \
	--report_to tensorboard \
	--overwrite_output_dir \
	--overwrite_cache \
	--train_file /home/zb/train_data/baichuan_sft/single_task_sn/overfit/train.json \
	--validation_file /home/zb/train_data/baichuan_sft/single_task_sn/dev.json \
	--num_train_epochs 20 \
	--overwrite_cache \
	--output_dir /home/zb/saved_checkpoint/baichuan_13b_overfit_5epoch \
	--use_lora true \
	--lora_target W_pack \
	--max_source_length 1024 \
	--max_target_length 128 \
	--warmup_ratio 0.1 \
	--logging_steps 10 \
	--lr_scheduler_type cosine \
	--per_device_train_batch_size 8 \
	--per_device_eval_batch_size 8 \
	--save_steps 100 \
	--save_total_limit=2 \
	--learning_rate 2e-5 \
	--bf16 true \
	--tf32 true

# wait
# baichuan13b 2个epoch用于对比测试学习率
# deepspeed --include=localhost:4,5 --master_port=${MASTER_PORT} --hostfile="" task/baichuan/sft_lora_trainer.py \
# 	--deepspeed ${DS_CONFIG_STAGE_2} \
# 	--do_train \
# 	--model_name_or_path ${BAICHUAN2_13B} \
# 	--report_to tensorboard \
# 	--overwrite_output_dir \
# 	--overwrite_cache \
# 	--train_file /home/zb/train_data/baichuan_sft/single_task_sn/train.json \
# 	--validation_file /home/zb/train_data/baichuan_sft/single_task_sn/dev.json \
# 	--num_train_epochs 2 \
# 	--overwrite_cache \
# 	--output_dir /home/zb/saved_checkpoint/baichuan_13b_2epoch \
# 	--use_lora true \
# 	--lora_target W_pack \
# 	--max_source_length 1024 \
# 	--max_target_length 128 \
# 	--warmup_ratio 0.1 \
# 	--logging_steps 10 \
# 	--lr_scheduler_type cosine \
# 	--per_device_train_batch_size 8 \
# 	--per_device_eval_batch_size 8 \
# 	--save_steps 100 \
# 	--save_total_limit=2 \
# 	--learning_rate 2e-5 \
# 	--bf16 true \
# 	--tf32 true
# wait
# baichuan一个epoch 调整lora学习率加十倍
# deepspeed --include=localhost:4,5 --master_port=${MASTER_PORT} --hostfile="" task/baichuan/sft_lora_trainer.py \
# 	--deepspeed ${DS_CONFIG_STAGE_2} \
# 	--do_train \
# 	--model_name_or_path ${BAICHUAN2_13B} \
# 	--report_to tensorboard \
# 	--overwrite_output_dir \
# 	--overwrite_cache \
# 	--train_file /home/zb/train_data/baichuan_sft/single_task_sn/train.json \
# 	--validation_file /home/zb/train_data/baichuan_sft/single_task_sn/dev.json \
# 	--num_train_epochs 2 \
# 	--overwrite_cache \
# 	--output_dir /home/zb/saved_checkpoint/baichuan_13b_2epoch_lr_e4 \
# 	--use_lora true \
# 	--lora_target W_pack \
# 	--max_source_length 1024 \
# 	--max_target_length 128 \
# 	--warmup_ratio 0.1 \
# 	--logging_steps 10 \
# 	--lr_scheduler_type cosine \
# 	--per_device_train_batch_size 8 \
# 	--per_device_eval_batch_size 8 \
# 	--save_steps 100 \
# 	--save_total_limit=2 \
# 	--learning_rate 2e-4 \
# 	--bf16 true \
# 	--tf32 true

# wait
# baichuan 2epoch 调整lora学习率减少10倍
# deepspeed --include=localhost:4,5 --master_port=${MASTER_PORT} --hostfile="" task/baichuan/sft_lora_trainer.py \
# 	--deepspeed ${DS_CONFIG_STAGE_3} \
# 	--do_train \
# 	--model_name_or_path ${BAICHUAN2_13B} \
# 	--report_to tensorboard \
# 	--overwrite_output_dir \
# 	--overwrite_cache \
# 	--train_file /home/zb/train_data/baichuan_sft/single_task_sn/train.json \
# 	--validation_file /home/zb/train_data/baichuan_sft/single_task_sn/dev.json \
# 	--num_train_epochs 2 \
# 	--overwrite_cache \
# 	--output_dir /home/zb/saved_checkpoint/baichuan_13b_2epoch_lr2e6 \
# 	--use_lora true \
# 	--lora_target W_pack \
# 	--max_source_length 1024 \
# 	--max_target_length 128 \
# 	--warmup_ratio 0.1 \
# 	--logging_steps 10 \
# 	--lr_scheduler_type cosine \
# 	--per_device_train_batch_size 8 \
# 	--per_device_eval_batch_size 8 \
# 	--save_steps 100 \
# 	--save_total_limit=2 \
# 	--learning_rate 2e-6 \
# 	--bf16 true \
# 	--tf32 true

# wait

# deepspeed --include=localhost:4,5 --master_port=${MASTER_PORT} --hostfile="" task/baichuan/sft_lora_trainer.py \
# 	--deepspeed ${DS_CONFIG_STAGE_3} \
# 	--do_train \
# 	--model_name_or_path ${BAICHUAN2_13B} \
# 	--report_to tensorboard \
# 	--overwrite_output_dir \
# 	--overwrite_cache \
# 	--train_file /home/zb/train_data/baichuan_sft/single_task_sn/train.json \
# 	--validation_file /home/zb/train_data/baichuan_sft/single_task_sn/dev.json \
# 	--num_train_epochs 2 \
# 	--overwrite_cache \
# 	--output_dir /home/zb/saved_checkpoint/baichuan_13b_2epoch_warmup0_lr2e6 \
# 	--use_lora true \
# 	--lora_target W_pack \
# 	--max_source_length 1024 \
# 	--max_target_length 128 \
# 	--warmup_ratio 0.0 \
# 	--logging_steps 10 \
# 	--lr_scheduler_type cosine \
# 	--per_device_train_batch_size 8 \
# 	--per_device_eval_batch_size 8 \
# 	--save_steps 100 \
# 	--save_total_limit=2 \
# 	--learning_rate 2e-6 \
# 	--bf16 true \
# 	--tf32 true

# wait

# deepspeed --include=localhost:4,5 --master_port=${MASTER_PORT} --hostfile="" task/baichuan/sft_lora_trainer.py \
# 	--deepspeed ${DS_CONFIG_STAGE_3} \
# 	--do_train \
# 	--model_name_or_path ${BAICHUAN2_13B} \
# 	--report_to tensorboard \
# 	--overwrite_output_dir \
# 	--overwrite_cache \
# 	--train_file /home/zb/train_data/baichuan_sft/single_task_sn/train.json \
# 	--validation_file /home/zb/train_data/baichuan_sft/single_task_sn/dev.json \
# 	--num_train_epochs 1 \
# 	--overwrite_cache \
# 	--output_dir /home/zb/saved_checkpoint/baichuan_13b_1epoch_warmup0 \
# 	--use_lora true \
# 	--lora_target W_pack \
# 	--max_source_length 1024 \
# 	--max_target_length 128 \
# 	--warmup_ratio 0.0 \
# 	--logging_steps 10 \
# 	--lr_scheduler_type cosine \
# 	--per_device_train_batch_size 8 \
# 	--per_device_eval_batch_size 8 \
# 	--save_steps 100 \
# 	--save_total_limit=2 \
# 	--learning_rate 2e-5 \
# 	--bf16 true \
# 	--tf32 true

# 下采样测试

deepspeed --include=localhost:4,5 --master_port=${MASTER_PORT} --hostfile="" task/baichuan/sft_lora_trainer.py \
	--deepspeed ${DS_CONFIG_STAGE_3} \
	--do_train \
	--model_name_or_path ${BAICHUAN2_13B} \
	--report_to tensorboard \
	--overwrite_output_dir \
	--overwrite_cache \
	--train_file /home/zb/train_data/baichuan_sft/single_task_sn/down_sample_1000/train.json \
	--validation_file /home/zb/train_data/baichuan_sft/single_task_sn/dev.json \
	--num_train_epochs 1 \
	--overwrite_cache \
	--output_dir /home/zb/saved_checkpoint/baichuan_13b_downsample_1000 \
	--use_lora true \
	--lora_target W_pack \
	--max_source_length 1024 \
	--max_target_length 128 \
	--warmup_ratio 0.0 \
	--logging_steps 10 \
	--lr_scheduler_type cosine \
	--per_device_train_batch_size 8 \
	--per_device_eval_batch_size 8 \
	--save_steps 100 \
	--save_total_limit=2 \
	--learning_rate 2e-5 \
	--bf16 true \
	--tf32 true

wait

deepspeed --include=localhost:4,5 --master_port=${MASTER_PORT} --hostfile="" task/baichuan/sft_lora_trainer.py \
	--deepspeed ${DS_CONFIG_STAGE_3} \
	--do_train \
	--model_name_or_path ${BAICHUAN2_13B} \
	--report_to tensorboard \
	--overwrite_output_dir \
	--overwrite_cache \
	--train_file /home/zb/train_data/baichuan_sft/single_task_sn/down_sample_3000/train.json \
	--validation_file /home/zb/train_data/baichuan_sft/single_task_sn/dev.json \
	--num_train_epochs 1 \
	--overwrite_cache \
	--output_dir /home/zb/saved_checkpoint/baichuan_13b_downsample_3000 \
	--use_lora true \
	--lora_target W_pack \
	--max_source_length 1024 \
	--max_target_length 128 \
	--warmup_ratio 0.0 \
	--logging_steps 10 \
	--lr_scheduler_type cosine \
	--per_device_train_batch_size 8 \
	--per_device_eval_batch_size 8 \
	--save_steps 100 \
	--save_total_limit=2 \
	--learning_rate 2e-5 \
	--bf16 true \
	--tf32 true
