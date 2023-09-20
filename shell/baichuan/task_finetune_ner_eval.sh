# 先激活环境
export PROJECT_PATH=/home/zb/code/NlpTaskSpace-llm/

cd ${PROJECT_PATH}

export PYTHONPATH=${PROJECT_PATH}

export MODEL_PATH=/data/SHARE/MODELS/BAICHUAN/Baichuan2-13B-Base/
export LORA_PATH=/home/zb/code/NlpTaskSpace-llm/output/baichuan-ner-128-v2/

export TRAIN_DATA=/home/zb/train_data/baichuan/dev.json
export VALID_DATA=/home/zb/train_data/baichuan/dev.json

# ddp
CUDA_VISIBLE_DEVICES=2 python evaluation/eval_baichuan_7b_ner.py \
	--output_dir ${PROJECT_PATH}/output/trainer \
	--dev_data ${VALID_DATA} \
	--model_name_or_path ${MODEL_PATH} \
	--lora_ckpt_path ${LORA_PATH}
