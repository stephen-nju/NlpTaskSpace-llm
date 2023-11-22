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

# deepspeed --include=localhost:6,7 --master_port=${MASTER_PORT} --hostfile="" task/baichuan/sft_lora_trainer.py \
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
# 	--output_dir /home/zb/saved_checkpoint/baichuan_13b_sn_single_task \
# 	--use_lora true \
# 	--lora_target W_pack \
# 	--max_source_length 1024 \
# 	--max_target_length 1024 \
# 	--warmup_ratio 0.0 \
# 	--logging_steps 10 \
# 	--lr_scheduler_type cosine \
# 	--per_device_train_batch_size 8 \
# 	--per_device_eval_batch_size 8 \
# 	--save_steps 200 \
# 	--save_total_limit=2 \
# 	--learning_rate 2e-5 \
# 	--bf16 true \
# 	--tf32 true

# wait

# deepspeed --include=localhost:6,7 --master_port=${MASTER_PORT} --hostfile="" task/baichuan/sft_lora_trainer.py \
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
# 	--output_dir /home/zb/saved_checkpoint/baichuan_13b_sn_single_task_512 \
# 	--use_lora true \
# 	--lora_target W_pack \
# 	--max_source_length 512 \
# 	--max_target_length 512 \
# 	--warmup_ratio 0.0 \
# 	--logging_steps 10 \
# 	--lr_scheduler_type cosine \
# 	--per_device_train_batch_size 8 \
# 	--per_device_eval_batch_size 8 \
# 	--save_steps 200 \
# 	--save_total_limit=2 \
# 	--learning_rate 2e-5 \
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

# deepspeed --include=localhost:4,5 --master_port=${MASTER_PORT} --hostfile="" task/baichuan/sft_lora_trainer.py \
# 	--deepspeed ${DS_CONFIG_STAGE_3} \
# 	--do_train \
# 	--model_name_or_path ${BAICHUAN2_13B} \
# 	--report_to tensorboard \
# 	--overwrite_output_dir \
# 	--overwrite_cache \
# 	--train_file /home/zb/train_data/baichuan_sft/single_task_sn/down_sample_1000/train.json \
# 	--validation_file /home/zb/train_data/baichuan_sft/single_task_sn/dev.json \
# 	--num_train_epochs 1 \
# 	--overwrite_cache \
# 	--output_dir /home/zb/saved_checkpoint/baichuan_13b_downsample_1000 \
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

# wait

# deepspeed --include=localhost:4,5 --master_port=${MASTER_PORT} --hostfile="" task/baichuan/sft_lora_trainer.py \
# 	--deepspeed ${DS_CONFIG_STAGE_3} \
# 	--do_train \
# 	--model_name_or_path ${BAICHUAN2_13B} \
# 	--report_to tensorboard \
# 	--overwrite_output_dir \
# 	--overwrite_cache \
# 	--train_file /home/zb/train_data/baichuan_sft/single_task_sn/down_sample_3000/train.json \
# 	--validation_file /home/zb/train_data/baichuan_sft/single_task_sn/dev.json \
# 	--num_train_epochs 1 \
# 	--overwrite_cache \
# 	--output_dir /home/zb/saved_checkpoint/baichuan_13b_downsample_3000 \
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

# wait
# # baichuan13b 过拟合数据集
# deepspeed --include=localhost:4,5 --master_port=${MASTER_PORT} --hostfile="" task/baichuan/sft_lora_trainer.py \
# 	--deepspeed ${DS_CONFIG_STAGE_3} \
# 	--do_train \
# 	--model_name_or_path ${BAICHUAN2_13B} \
# 	--report_to tensorboard \
# 	--overwrite_output_dir \
# 	--overwrite_cache \
# 	--train_file /home/zb/train_data/baichuan_sft/single_task_sn/overfit/train.json \
# 	--validation_file /home/zb/train_data/baichuan_sft/single_task_sn/dev.json \
# 	--num_train_epochs 1 \
# 	--overwrite_cache \
# 	--output_dir /home/zb/saved_checkpoint/baichuan_13b_overfit_1epoch \
# 	--use_lora true \
# 	--lora_target W_pack \
# 	--max_source_length 1024 \
# 	--max_target_length 128 \
# 	--warmup_ratio 0.1 \
# 	--logging_teps 10 \
# 	--lr_scheduler_type cosine \
# 	--per_device_train_batch_size 8 \
# 	--per_device_eval_batch_size 8 \
# 	--save_steps 100 \
# 	--save_total_limit=2 \
# 	--learning_rate 2e-5 \
# 	--bf16 truje \
# 	--tf32 true

#更新prompt形式V1版本
# deepspeed --include=localhost:4,5 --master_port=${MASTER_PORT} --hostfile="" task/baichuan/sft_lora_trainer.py \
# 	--deepspeed ${DS_CONFIG_STAGE_3} \
# 	--do_train \
# 	--model_name_or_path ${BAICHUAN2_13B} \
# 	--report_to tensorboard \
# 	--overwrite_output_dir \
# 	--overwrite_cache \
# 	--train_file /home/zb/train_data/baichuan_sft/single_task_sn/enhance_prompt_v1/train.json \
# 	--validation_file /home/zb/train_data/baichuan_sft/single_task_sn/enhance_prompt_v1/dev.json \
# 	--num_train_epochs 1 \
# 	--overwrite_cache \
# 	--output_dir /home/zb/saved_checkpoint/baichuan_13b_enhance_prompt_v1 \
# 	--use_lora true \
# 	--lora_target W_pack \
# 	--max_source_length 1024 \
# 	--max_target_length 128 \
# 	--warmup_ratio 0.1 \
# 	--logging_steps 10 \
# 	--lr_scheduler_type cosine \
# 	--per_device_train_batch_size 8 \
# 	--per_device_eval_batch_size 8 \
# 	--save_steps 100
# 	--save_total_limit=2 \
# 	--learning_rate 2e-5 \
# 	--bf16 true \
# 	--tf32 true

# 更新prompt形式V2版本
# deepspeed --include=localhost:2,3 --master_port=${MASTER_PORT} --hostfile="" task/baichuan/sft_lora_trainer.py \
# 	--deepspeed ${DS_CONFIG_STAGE_3} \
# 	--do_train \
# 	--model_name_or_path ${BAICHUAN2_13B} \
# 	--report_to tensorboard \
# 	--overwrite_output_dir \
# 	--overwrite_cache \
# 	--train_file /home/zb/train_data/baichuan_sft/single_task_sn/merge_general_instruction/train.json \
# 	--validation_file /home/zb/train_data/baichuan_sft/single_task_sn/merge_general_instruction/train.json \
# 	--num_train_epochs 1 \
# 	--overwrite_cache \
# 	--output_dir /home/zb/saved_checkpoint/baichuan_13b_merge_instruction \
# 	--use_lora true \
# 	--lora_target W_pack \
# 	--max_source_length 1024 \
# 	--max_target_length 1024 \
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

# deepspeed --include=localhost:4,5 --master_port=${MASTER_PORT} --hostfile="" task/baichuan/sft_lora_trainer.py \
# 	--deepspeed ${DS_CONFIG_STAGE_3} \
# 	--do_train \
# 	--model_name_or_path ${BAICHUAN2_13B} \
# 	--report_to tensorboard \
# 	--overwrite_output_dir \
# 	--overwrite_cache \
# 	--train_file /home/zb/train_data/baichuan_sft/single_task_sn/train.json \
# 	--validation_file /home/zb/train_data/baichuan_sft/single_task_sn/dev.json \
# 	--num_train_epochs 100 \
# 	--overwrite_cache \
# 	--output_dir /home/zb/saved_checkpoint/baichuan_13b_overfit_1epoch \
# 	--use_lora true \
# 	--lora_target W_pack \
# 	--max_source_length 1024 \
# 	--max_target_length 128 \
# 	--warmup_ratio 0.1 \
# 	--logging_steps 10 \
# 	--lr_scheduler_type cosine \
# 	--per_device_train_batch_size 8 \
# 	--per_device_eval_batch_size 8 \
# 	--save_steps 10000 \
# 	--save_total_limit=5 \
# 	--learning_rate 2e-5 \
# 	--bf16 true \
# 	--tf32 true

# 淘宝数据集的任务
# deepspeed --include=localhost:4,5 --master_port=${MASTER_PORT} --hostfile="" task/baichuan/sft_lora_trainer.py \
# 	--deepspeed ${DS_CONFIG_STAGE_3} \
# 	--do_train \
# 	--model_name_or_path ${BAICHUAN2_13B} \
# 	--report_to tensorboard \
# 	--overwrite_output_dir \
# 	--overwrite_cache \
# 	--train_file /home/zb/train_data/baichuan_sft/single_task_tb/single_task_v1/train.json \
# 	--validation_file /home/zb/train_data/baichuan_sft/single_task_tb/single_task_v1/dev.json \
# 	--output_dir /home/zb/saved_checkpoint/baichuan_13b_tb_single_task \
# 	--num_train_epochs 1 \
# 	--overwrite_cache \
# 	--use_lora true \
# 	--lora_target W_pack \
# 	--max_source_length 1024 \
# 	--max_target_length 1024 \
# 	--warmup_ratio 0.1 \
# 	--logging_steps 10 \
# 	--lr_scheduler_type cosine \
# 	--per_device_train_batch_size 8 \
# 	--per_device_eval_batch_size 8 \
# 	--save_steps 500 \
# 	--save_total_limit=2 \
# 	--learning_rate 2e-5 \
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
# 	--train_file /home/zb/train_data/baichuan_sft/single_task_sn_v2/train.json \
# 	--validation_file /home/zb/train_data/baichuan_sft/single_task_sn_v2/dev.json \
# 	--output_dir /home/zb/saved_checkpoint/baichuan_13b_sn_single_task_v2_new_data \
# 	--num_train_epochs 1 \
# 	--overwrite_cache \
# 	--use_lora true \
# 	--lora_target W_pack \
# 	--max_source_length 1024 \
# 	--max_target_length 1024 \
# 	--warmup_ratio 0.1 \
# 	--logging_steps 10 \
# 	--lr_scheduler_type cosine \
# 	--per_device_train_batch_size 8 \
# 	--per_device_eval_batch_size 8 \
# 	--save_steps 200 \
# 	--save_total_limit=2 \
# 	--learning_rate 2e-5 \
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
# 	--train_file /home/zb/train_data/baichuan_sft/single_task_sn_v2/split_task_v2/train.json \
# 	--validation_file /home/zb/train_data/baichuan_sft/single_task_sn_v2/split_task_v2/dev.json \
# 	--output_dir /home/zb/saved_checkpoint/baichuan_13b_sn_split_task_v2_new_data \
# 	--num_train_epochs 1 \
# 	--overwrite_cache \
# 	--use_lora true \
# 	--lora_target W_pack \
# 	--max_source_length 1024 \
# 	--max_target_length 1024 \
# 	--warmup_ratio 0.1 \
# 	--logging_steps 10 \
# 	--lr_scheduler_type cosine \
# 	--per_device_train_batch_size 8 \
# 	--per_device_eval_batch_size 8 \
# 	--save_steps 200 \
# 	--save_total_limit=2 \
# 	--learning_rate 2e-5 \
# 	--bf16 true \
# 	--tf32 true

# wait

# deepspeed --include=localhost:0,1,2,3,6,7 --master_port=29503 --hostfile="" task/baichuan/pt_trainer.py \
# 	--deepspeed ${DS_CONFIG_STAGE_3} \
# 	--do_train \
# 	--report_to tensorboard \
# 	--overwrite_output_dir \
# 	--overwrite_cache \
# 	--train_file /home/zb/train_data/baichuan_pt/query_pretrain/train.json \
# 	--output_dir /home/zb/saved_checkpoint/baichuan_13b_sn_query_pt \
# 	--num_train_epochs 1 \
# 	--overwrite_cache \
# 	--model_name_or_path ${BAICHUAN2_13B} \
# 	--block_size 1024 \
# 	--warmup_ratio 0.1 \
# 	--logging_steps 100 \
# 	--per_device_train_batch_size 8 \
# 	--per_device_eval_batch_size 8 \
# 	--lr_scheduler_type cosine \
# 	--save_steps 1000 \
# 	--save_total_limit=2 \
# 	--learning_rate 2e-5 \
# 	--bf16 true \
# 	--tf32 true

# deepspeed --include=localhost:0,1,4,5,6,7 --master_port=${MASTER_PORT} --hostfile="" task/baichuan/sft_lora_trainer.py \
# 	--deepspeed ${DS_CONFIG_STAGE_3} \
# 	--do_train \
# 	--model_name_or_path ${BAICHUAN2_13B} \
# 	--report_to tensorboard \
# 	--overwrite_output_dir
# 	--overwrite_cache \
# 	--train_file /home/zb/train_data/baichuan_sft/single_task_sn_v2/merge_single_and_split/train.json \
# 	--validation_file /home/zb/train_data/baichuan_sft/single_task_sn_v2/split_task_v2/dev.json \
# 	--output_dir /home/zb/saved_checkpoint/baichuan_13b_sn_v2_merge_single_and_split \
# 	--num_train_epochs 1 \
# 	--overwrite_cache \
# 	--use_lora true \
# 	--lora_target W_pack \
# 	--max_source_length 1024 \
# 	--max_target_length 1024 \
# 	--warmup_ratio 0.1 \
# 	--logging_steps 10 \
# --lr_scheduler_type cosine \
# --save_total_limit=2 \
# --learning_rate 2e-5 # --save_steps 200 \
# --tf32 true # --per_device_eval_batch_size 8 \
# --bf16 true # 	--per_device_train_batch_size 8 \

# deepspeed 3 目前不兼容 configure_model,需要初始化加载，deepspeed 3 百川模型验证失败
source /root/venv/lightning/bin/activate
# 激活虚拟环境
deepspeed --include=localhost:6,7 --master_port=${MASTER_PORT} --hostfile="" --no_local_rank task/baichuan/supervised_finetuning_lightning.py \
	--deepspeed ${LIGHTNING_DS_CONFIG_STAGE_2} \
	--overwrite_cache \
	--model_name_or_path /home/zb/model/Baichuan2-13B-Base/ \
	--output_dir /home/zb/saved_checkpoint/base_sn_v5_lr5e5_epoch2_lora_all \
	--train_data /home/zb/suningGit/zb/train_data/v5/dev.json \
	--dev_data /home/zb/suningGit/zb/train_data/v5/dev.json \
	--max_epochs 2 \
	--max_source_length 1024 \
	--max_target_length 2048 \
	--warmup_proportion 0.1 \
	--per_device_train_batch_size 4 \
	--per_device_eval_batch_size 4 \
	--learning_rate 5e-5 \
	--gradient_accumulation_steps 8 \
	--preprocessing_num_workers 16 \
	--save_steps 500 \
	--use_lora true \
	--lora_target all \
	--lora_rank 8 \
	--use_slow_tokenizer \
	--lr_scheduler_type cosine \
	--low_cpu_mem_usage
