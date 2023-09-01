# 先激活环境
export PROJECT_PATH=/home/zhubin/code/NlpTaskSpace-llm/

cd ${PROJECT_PATH}

export PYTHONPATH=${PROJECT_PATH}

export MODEL_PATH=/home/zhubin/model/Baichuan-7B/
export LORA_PATH=/home/zhubin/code/NlpTaskSpace-llm/output/baichuan-ner-128/

export TRAIN_DATA=/home/zhubin/train_data/baichuan/dev.json
export VALID_DATA=/home/zhubin/train_data/baichuan/dev.json

# ddp
CUDA_VISIBLE_DEVICES=0 python evaluation/eval_baichuan_7b_ner.py \
	--output_dir ${PROJECT_PATH}/output/trainer \
	--dev_data ${VALID_DATA} \
	--model_name_or_path ${MODEL_PATH} \
	--lora_ckpt_path ${LORA_PATH}
