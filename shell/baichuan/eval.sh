# 验证千问的模型
export PROJECT_PATH=/home/zb/NlpTaskSpace-llm/
cd ${PROJECT_PATH}

export PYTHONPATH=${PROJECT_PATH}

#千问的
CUDA_VISIBLE_DEVICES=3 python evaluation/eval_baichuan_ner.py \
	--model_name_or_path=/data/SHARE/MODELS/Qwen/Qwen-14B/Qwen-14B \
	--experiment_name=qwen-14-sn \
	--lora_ckpt_path=/home/zb/saved_checkpoint/qwen_14b_2epoch_sn \
	--dev_data=/home/zb/train_data/baichuan_sft/single_task_sn/dev.json
wait

# 2e-5 两个epoch
CUDA_VISIBLE_DEVICES=3 python evaluation/eval_baichuan_ner.py \
	--model_name_or_path=/data/SHARE/MODELS/BAICHUAN/Baichuan2-13B-Base/ \
	--experiment_name=baichuan-13b-lr2e5-2epoch \
	--lora_ckpt_path=/home/zb/saved_checkpoint/baichuan_13b_2epoch_lr2e5 \
	--dev_data=/home/zb/train_data/baichuan_sft/single_task_sn/dev.json

wait

CUDA_VISIBLE_DEVICES=3 python evaluation/eval_baichuan_ner.py \
	--model_name_or_path=/data/SHARE/MODELS/BAICHUAN/Baichuan2-13B-Base/ \
	--experiment_name=baichuan-13b-lr2e4-2epoch \
	--lora_ckpt_path=/home/zb/saved_checkpoint/baichuan_13b_2epoch_lr_e4 \
	--dev_data=/home/zb/train_data/baichuan_sft/single_task_sn/dev.json

wait

# CUDA_VISIBLE_DEVICES=3 python evaluation/eval_baichuan_ner.py \
# 	--model_name_or_path=/data/SHARE/MODELS/BAICHUAN/Baichuan2-13B-Base/ \
# 	--experiment_name=baichuan-13b-lr2e5-2epoch-warmup0 \
# 	--lora_ckpt_path=/home/zb/saved_checkpoint/baichuan_13b_warmup0 \
# 	--dev_data=/home/zb/train_data/baichuan_sft/single_task_sn/dev.json

# wait

# CUDA_VISIBLE_DEVICES=3 python evaluation/eval_baichuan_ner.py \
# 	--model_name_or_path=/data/SHARE/MODELS/BAICHUAN/Baichuan2-13B-Base/ \
# 	--experiment_name=baichuan-13b-lr2e6-2epoch \
# 	--lora_ckpt_path=/home/zb/saved_checkpoint/test/baichuan_13b_2epoch_lr_e6 \
# 	--dev_data=/home/zb/train_data/baichuan_sft/single_task_sn/dev.json

# wait
