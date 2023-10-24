import datetime
import subprocess
import subprocess as sp
import time

from apscheduler.executors.pool import ThreadPoolExecutor
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.date import DateTrigger


# 创建调度器对象

scheduler = BlockingScheduler()


# 定义任务函数
def task_func1():
    out = sp.run(
        """
        export PROJECT_PATH=/home/zb/NlpTaskSpace-llm \
        && cd ${PROJECT_PATH} \
        && export PYTHONPATH=${PROJECT_PATH} \
        && export BAICHUAN2_13B=/data/SHARE/MODELS/BAICHUAN/Baichuan2-13B-Base/ \
        && export DS_CONFIG_STAGE_3=/home/zb/NlpTaskSpace-llm/config/deepspeed/zero_stage3_config.json \
        && export DS_CONFIG_STAGE_2=/home/zb/NlpTaskSpace-llm/config/deepspeed/zero_stage2_config.json \
        && MASTER_PORT=$(shuf -n 1 -i 10000-65535) \
        && deepspeed --include=localhost:0,1 --master_port=${MASTER_PORT} --hostfile="" task/baichuan/sft_lora_trainer.py \
            --deepspeed ${DS_CONFIG_STAGE_3} \
            --do_train \
            --model_name_or_path ${BAICHUAN2_13B} \
            --report_to tensorboard \
            --overwrite_output_dir \
            --overwrite_cache \
            --train_file /home/zb/train_data/baichuan_sft/single_task_sn_v2/split_task_v2/train.json \
            --validation_file /home/zb/train_data/baichuan_sft/single_task_sn_v2/split_task_v2/dev.json \
            --output_dir /home/zb/saved_checkpoint/baichuan_13b_sn_split_task_v2_new_data \
            --num_train_epochs 1 \
            --overwrite_cache \
            --use_lora true \
            --lora_target W_pack \
            --max_source_length 1024 \
            --max_target_length 1024 \
            --warmup_ratio 0.1 \
            --logging_steps 10 \
            --lr_scheduler_type cosine \
            --per_device_train_batch_size 8 \
            --per_device_eval_batch_size 8 \
            --save_steps 200 \
            --save_total_limit=2 \
            --learning_rate 2e-5 \
            --bf16 true \
            --tf32 true 
            """,
        shell=True,
        check=True,
    )
    print(out)


def task_new():
    out = sp.check_output(
        'export PROJECT_PATH=/home/zb/NlpTaskSpace-llm/ \
        && cd ${PROJECT_PATH} \
        && export PYTHONPATH=${PROJECT_PATH} \
        && export BAICHUAN2_13B=/data/SHARE/MODELS/BAICHUAN/Baichuan2-13B-Base/ \
        && export DS_CONFIG_STAGE_3=/home/zb/NlpTaskSpace-llm/config/deepspeed/zero_stage3_config.json \
        && export DS_CONFIG_STAGE_2=/home/zb/NlpTaskSpace-llm/config/deepspeed/zero_stage2_config.json \
        && MASTER_PORT=$(shuf -n 1 -i 10000-65535) \
        && deepspeed --include=localhost:0,1 --master_port=${MASTER_PORT} --hostfile="" task/baichuan/sft_lora_trainer.py \
            --deepspeed ${DS_CONFIG_STAGE_3} \
            --do_train \
            --model_name_or_path ${BAICHUAN2_13B} \
            --report_to tensorboard \
            --overwrite_output_dir \
            --overwrite_cache \
            --train_file /home/zb/train_data/baichuan_sft/single_task_sn_v2/split_task_v2/train.json \
            --validation_file /home/zb/train_data/baichuan_sft/single_task_sn_v2/split_task_v2/dev.json \
            --output_dir /home/zb/saved_checkpoint/baichuan_13b_sn_split_task_v2_new_data \
            --num_train_epochs 1 \
            --overwrite_cache \
            --use_lora true \
            --lora_target W_pack \
            --max_source_length 1024 \
            --max_target_length 1024 \
            --warmup_ratio 0.1 \
            --logging_steps 10 \
            --lr_scheduler_type cosine \
            --per_device_train_batch_size 8 \
            --per_device_eval_batch_size 8 \
            --save_steps 200 \
            --save_total_limit=2 \
            --learning_rate 2e-5 \
            --bf16 true \
            --tf32 true ',
        stderr=subprocess.STDOUT,
        shell=True,
    )


def task_func2():
    out = sp.check_output(
        'source activate baichuan13b \
        && export PROJECT_PATH=/home/zb/NlpTaskSpace-llm/ \
        && cd ${PROJECT_PATH} \
        && export PYTHONPATH=${PROJECT_PATH} \
        && export BAICHUAN2_13B=/data/SHARE/MODELS/BAICHUAN/Baichuan2-13B-Base/ \
        && export DS_CONFIG_STAGE_3=/home/zb/NlpTaskSpace-llm/config/deepspeed/zero_stage3_config.json \
        && export DS_CONFIG_STAGE_2=/home/zb/NlpTaskSpace-llm/config/deepspeed/zero_stage2_config.json \
        && MASTER_PORT=$(shuf -n 1 -i 10000-65535) \
        && deepspeed --include=localhost:0,1,2,3,4,5,6,7 --master_port=${MASTER_PORT} --hostfile="" task/baichuan/pt_trainer.py \
        --deepspeed ${DS_CONFIG_STAGE_3} \
        --do_train \
        --model_name_or_path ${BAICHUAN2_13B} \
        --report_to tensorboard \
        --overwrite_output_dir \
        --overwrite_cache \
        --train_file /home/zb/train_data/baichuan_pt/query_pretrain/train.json \
        --output_dir /home/zb/saved_checkpoint/baichuan_13b_sn_query_pt \
        --num_train_epochs 1 \
        --overwrite_cache \
        --max_source_length 1024 \
        --max_target_length 1024 \
        --block_size 1024 \
        --warmup_ratio 0.1 \
        --logging_steps 10 \
        --lr_scheduler_type cosine \
        --per_device_train_batch_size 8 \
        --per_device_eval_batch_size 8 \
        --save_steps 200 \
        --save_total_limit=2 \
        --learning_rate 2e-5 \
        --bf16 true \
        --tf32 true '
    )

    print(out)

    print("执行任务2...")


if __name__ == "__main__":
    # 为了防止全量和增量并发造成显存溢出，进而训练失败，设置同一时间只能有一个任务运行
    schedule = BlockingScheduler(executors={"default": ThreadPoolExecutor(1)})
    # schedule = BlockingScheduler()
    ##添加任务 依次往后顺延时间就行
    after_trigger1 = (datetime.datetime.now() + datetime.timedelta(seconds=2)).strftime("%Y-%m-%d %H:%M:%S")
    after_trigger2 = (datetime.datetime.now() + datetime.timedelta(hours=8)).strftime("%Y-%m-%d %H:%M:%S")
    trigger1 = DateTrigger(run_date=after_trigger1)
    trigger2 = DateTrigger(run_date=after_trigger2)
    # 添加定时任务
    schedule.add_job(task_func1, trigger1)
    # schedule.add_job(task_func2, trigger2)
    # 启动调度器
    schedule.print_jobs()

    schedule.start()
