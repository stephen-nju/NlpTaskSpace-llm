source activate baichuan13b
export PROJECT_PATH=/home/zb/NlpTaskSpace-llm/
cd ${PROJECT_PATH}
export PYTHONPATH=${PROJECT_PATH}
export QWEN_14B_BASE=/data/SHARE/MODELS/Qwen/Qwen-14B/Qwen-14B/

CUDA_VISIBLE_DEVICES=4,5 python inference/xverse/run_chat.py \
	--model_name_or_path=/home/zb/saved_checkpoint/base_xverse_sn_v6_lora_lr1e4_2epoch/merge/ \
	--data_path=/data/SHARE/tmpt/concatData.json \
	--experiment_name=ceping_base_xverse_sn_v6_lora_lr1e4_2epoch
wait

CUDA_VISIBLE_DEVICES=4,5 python inference/xverse/run_chat.py \
	--model_name_or_path=/home/zb/saved_checkpoint/base_xverse_sn_v6_lora_lr5e5_2epoch/merge/ \
	--data_path=/data/SHARE/tmpt/concatData.json \
	--experiment_name=ceping_base_xverse_sn_v6_lora_lr5e5_2epoch
