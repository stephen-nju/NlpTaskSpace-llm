
export PROJECT_PATH=/home/zb/NlpTaskSpace-llm/
cd ${PROJECT_PATH}
export PYTHONPATH=${PROJECT_PATH}
export QWEN_14B_BASE=/data/SHARE/MODELS/Qwen/Qwen-14B/Qwen-14B/

CUDA_VISIBLE_DEVICES=4,5 python inference/yi/run_chat.py \
	--model_name_or_path=/data/SHARE/MODELS/01ai/Yi-34B/ \
	--data_path=/data/SHARE/tmpt/concatData.json \
	--experiment_name=yi_test
