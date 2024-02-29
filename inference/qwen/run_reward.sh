export PROJECT_PATH=/home/zb/NlpTaskSpace-llm/
cd ${PROJECT_PATH}
export PYTHONPATH=${PROJECT_PATH}
export QWEN_14B_BASE=/data/SHARE/MODELS/Qwen/Qwen-14B/Qwen-14B/

# CUDA_VISIBLE_DEVICES=0,1 python inference/qwen/run_chat.py \
# 	--model_name_or_path=/home/zb/saved_checkpoint/base_qwen_sn_v6_lora_lr1e4_2epoch/merge/ \
# 	--data_path=/data/SHARE/tmpt/concatData.json \
# 	--experiment_name=ceping_base_qwen_sn_v6_lora_lr1e4_2epoch

# wait

# CUDA_VISIBLE_DEVICES=4 python inference/qwen/run_chat.py \
# 	--model_name_or_path=/home/zb/saved_checkpoint/base_qwen_sn_v12_lora_lr1e4_2epoch/merge/ \
# 	--data_path=/home/zb/NlpTaskSpace-llm/data/chanping_test.json \
# 	--format=product \
# 	--experiment_name=chanping_test_v12_1e4

CUDA_VISIBLE_DEVICES=0 python inference/qwen/reward_score.py \
	--reward_model_name_or_path=/home/zb/saved_checkpoint/reward_qwen_sn_v12_lora_lr1e6_1epoch/merge \
	--vhead_path=/home/zb/saved_checkpoint/reward_qwen_sn_v12_lora_lr1e6_1epoch/ \
	--data_path=/home/zb/LLaMA-Factory/data/comparison_gpt4_data_zh.json \
	--format=product \
	--experiment_name=chanping_test_ppo_200

# --data_path=/home/zb/NlpTaskSpace-llm/data/chanping_test.json \
