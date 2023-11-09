source activate baichuan13b
export PROJECT_PATH=/home/zb/NlpTaskSpace-llm/
cd ${PROJECT_PATH}

export PYTHONPATH=${PROJECT_PATH}
export BAICHUAN2_13B_BASE=/data/SHARE/MODELS/BAICHUAN/Baichuan2-13B-Base/
CUDA_VISIBLE_DEVICES=0,1 python scripts/run_base/run_baichuan_base.py \
	--model_name_or_path=${BAICHUAN2_13B_BASE} \
	--lora_ckpt_path=/home/zb/saved_checkpoint/base_sn_generate_v4_alpaca_3epoch \
	--experiment_name=13b-base-humman-test-with-alpaca-4epoch-temperature-0.2-top-k5 \
	--data_path=/home/zb/suningGit/zb/LLaMA-Factory/data/llm_human_construction.json
