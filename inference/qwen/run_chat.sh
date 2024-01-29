export PROJECT_PATH=/home/zb/NlpTaskSpace-llm/
cd ${PROJECT_PATH}
export PYTHONPATH=${PROJECT_PATH}
export QWEN_14B_BASE=/data/SHARE/MODELS/Qwen/Qwen-14B/Qwen-14B/

# CUDA_VISIBLE_DEVICES=0,1 python inference/qwen/run_chat.py \
# 	--model_name_or_path=/home/zb/saved_checkpoint/base_qwen_sn_v6_lora_lr1e4_2epoch/merge/ \
# 	--data_path=/data/SHARE/tmpt/concatData.json \
# 	--experiment_name=ceping_base_qwen_sn_v6_lora_lr1e4_2epoch

# wait

CUDA_VISIBLE_DEVICES=4 python inference/qwen/run_chat.py \
	--model_name_or_path=/home/zb/saved_checkpoint/base_qwen_sn_v11_lora_lr1e-4_3epoch/merge/ \
	--data_path=/home/zb/NlpTaskSpace-llm/data/chanping_test.json \
	--format=product \
	--experiment_name=chanping_test_v13

CUDA_VISIBLE_DEVICES=5 python inference/qwen/run_chat.py \
	--model_name_or_path=/home/zb/saved_checkpoint/base_qwen_sn_v11_lora_lr1e-4_3epoch/merge/ \
	--data_path=/data/SHARE/tmpt/sn_chat_dev.json \
	--format=raw \
	--experiment_name=sn_chat_dev
