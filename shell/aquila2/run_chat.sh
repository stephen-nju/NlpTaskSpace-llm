# 先活环境
export PROJECT_PATH=/home/zb/NlpTaskSpace-llm/
cd ${PROJECT_PATH}
export PYTHONPATH=${PROJECT_PATH}
export DS_CONFIG_STAGE_3=/home/zb/NlpTaskSpace-llm/config/lightning_deepspeed/zero_stage3_config.json
export DS_CONFIG_STAGE_2=/home/zb/NlpTaskSpace-llm/config/lightning_deepspeed/aquila2-chat-34b/zero_stage2_config.json
MASTER_PORT=$(shuf -n 1 -i 10000-65535)
export TOKENIZERS_PARALLELISM=false
export MODEL_PATH=/data/SHARE/MODELS/BAAI/AquilaChat2-34B/

CUDA_VISIBLE_DEVICES=0 python task/aquila2/run_chat.py \
	--model_name_or_path=${MODEL_PATH} \
	--lora_ckpt_path=/home/zb/saved_checkpoint/aquila2-sn-v2/hf_model \
	--experiment_name=aquila-human_test_v3_2epoch \
	--data_path=/home/zb/suningGit/zb/LLaMA-Factory/data/llm_human_construction.json
