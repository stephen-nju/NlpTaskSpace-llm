export PROJECT_PATH=/home/zb/NlpTaskSpace-llm/
cd ${PROJECT_PATH}

export PYTHONPATH=${PROJECT_PATH}
export SKYWORK=/data/SHARE/MODELS/SkyWork/Skywork-13B-base/

CUDA_VISIBLE_DEVICES=4,5 python task/skywork/run_chat.py \
	--model_name_or_path=${SKYWORK} \
	--lora_ckpt_path=/home/zb/saved_checkpoint/light_skywork_sft_sn_v5_lr1e4_1epoch/last.ckpt/ \
	--data_path=/home/zb/NlpTaskSpace-llm/data/badcase.jsonl \
	--experiment_name=badcase-skywork
