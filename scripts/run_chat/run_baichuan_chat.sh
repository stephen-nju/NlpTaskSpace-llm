source activate baichuan13b
export PROJECT_PATH=/home/zb/NlpTaskSpace-llm/
cd ${PROJECT_PATH}

export PYTHONPATH=${PROJECT_PATH}
export BAICHUAN2_13B_CHAT=/data/SHARE/MODELS/BAICHUAN/Baichuan2-13B-Chat/

CUDA_VISIBLE_DEVICES=0,1 python scripts/run_chat/run_baichuan_chat.py \
	--model_name_or_path=${BAICHUAN2_13B_CHAT} \
	--lora_ckpt_path=/home/zb/saved_checkpoint/chat_sn_generate_v4_3epoch/ \
	--experiment_name=chat_sn_generate_v4_3epoch_default \
	--data_path=/home/zb/suningGit/zb/general_test_data/general_test.json
