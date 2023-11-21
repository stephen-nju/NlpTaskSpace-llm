export PROJECT_PATH=/home/zb/NlpTaskSpace-llm/
cd ${PROJECT_PATH}

export PYTHONPATH=${PROJECT_PATH}
export BAICHUAN_7B=/data/SHARE/MODELS/BAICHUAN/Baichuan-7B
export BAICHUAN2_13B=/data/SHARE/MODELS/BAICHUAN/Baichuan2-13B-Base/
export DS_CONFIG=/home/zb/NlpTaskSpace-llm/config/deepspeed/zero_stage3_config.json
# 运行baichuan1 7b的lora模型

# v1=--lora_ckpt_path=/home/zb/saved_checkpoint/chat_sn_generate_v4_alpaca_8epoch/checkpoint-17200/ \

CUDA_VISIBLE_DEVICES=4 python scripts/merge_model_and_save_pretrain.py \
	--model_name_or_path=/data/SHARE/MODELS/BAICHUAN/Baichuan2-13B-Chat/ \
	--lora_ckpt_path=/home/zb/saved_checkpoint/chat_sn_v4_plus_alpaca_2epoch_lr1e4/ \
	--output_dir=/home/zb/saved_checkpoint/output_dir
