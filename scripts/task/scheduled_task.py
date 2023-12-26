import datetime
import subprocess as sp

from apscheduler.executors.pool import ThreadPoolExecutor
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.date import DateTrigger

# 创建调度器对象
scheduler = BlockingScheduler()


# 定义任务函数
def task_func1():
    print("执行任务1....")
    out = sp.run(
        """
        export PROJECT_PATH=/home/zb/NlpTaskSpace-llm/ \
        && cd ${PROJECT_PATH} \
        && export PYTHONPATH=${PROJECT_PATH} \
        && export BAICHUAN2_13B=/data/SHARE/MODELS/BAICHUAN/Baichuan2-13B-Base/ \
        && CUDA_VISIBLE_DEVICES=5 python scripts/merge_model_and_save_pretrain.py \
            --model_name_or_path=/data/SHARE/MODELS/BAICHUAN/Baichuan2-13B-Base/ \
            --lora_ckpt_path=/home/zb/saved_checkpoint/light_baichuan_base_pt_lr1e4_2epoch/last.ckpt/ \
            --output_dir=/home/zb/saved_checkpoint/light_baichuan_base_pt_lr1e4_2epoch/merge/
        """,
        shell=True,
        check=True,
    )
    print(out)


def task_test():
    print("执行任务1....")
    out = sp.run(
        """
        echo hello word
        """,
        shell=True,
        check=True,
    )
    print(out)


def task_test2():
    print("执行任务2....")
    out = sp.run(
        """
        echo hello word 2
        """,
        shell=True,
        check=True,
    )
    print(out)


def task_func2():
    print("执行任务2......")
    out = sp.check_output(
        """
        export PROJECT_PATH=/home/zb/NlpTaskSpace-llm/ \
        && cd ${PROJECT_PATH} \
        && export PYTHONPATH=${PROJECT_PATH} \
        && export BAICHUAN2_13B=/data/SHARE/MODELS/BAICHUAN/Baichuan2-13B-Base/ \
        && export BAICHUAN2_13B_CHAT=/data/SHARE/MODELS/BAICHUAN/Baichuan2-13B-Chat/ \
        && export DS_CONFIG_STAGE_3=/home/zb/NlpTaskSpace-llm/config/deepspeed/zero_stage3_config.json \
        && export DS_CONFIG_STAGE_2=/home/zb/NlpTaskSpace-llm/config/deepspeed/zero_stage2_config.json \
        && MASTER_PORT=$(shuf -n 1 -i 10000-65535) \
        && echo ${MASTER_PORT} \
        && echo ${PYTHONPATH} \
        && deepspeed --include=localhost:0,1 --master_port=${MASTER_PORT} --hostfile="" --no_local_rank task/qwen/supervised_finetuning_lightning.py \
            --deepspeed ${DS_CONFIG_STAGE_2} \
            --overwrite_cache \
            --model_name_or_path ${MODEL_PATH} \
            --output_dir /home/zb/saved_checkpoint/light_baichuan_pt_v2_sft_sn_v5_tiger_lr2e4_1epoch \
            --train_data /home/zb/suningGit/zb/train_data/v5/train/,/home/zb/suningGit/zb/train_data/tigerbot_sft_zh \
            --dev_data /home/zb/suningGit/zb/train_data/v5/dev/ \
            --max_epochs 1 \
            --max_source_length 1024 \
            --max_target_length 2048 \
            --warmup_ratio 0.1 \
            --per_device_train_batch_size 8 \
            --per_device_eval_batch_size 8 \
            --learning_rate 1e-4 \
            --gradient_accumulation_steps 1 \
            --preprocessing_num_workers 16 \
            --save_steps 5000 \
            --use_lora true \
            --lora_target all \
            --lora_rank 8 \
            --lora_alpha 16 \
            --use_slow_tokenizer \
            --lr_scheduler_type cosine \
            --low_cpu_mem_usage
        """,
        stderr=sp.STDOUT,
        shell=True,
    )
    print(out)


if __name__ == "__main__":
    # 为了防止全量和增量并发造成显存溢出，进而训练失败，设置同一时间只能有一个任务运行
    schedule = BlockingScheduler(executors={"default": ThreadPoolExecutor(1)})
    # schedule = BlockingScheduler()
    ##添加任务 依次往后顺延时间就行
    after_trigger1 = (datetime.datetime.now() + datetime.timedelta(seconds=2)).strftime("%Y-%m-%d %H:%M:%S")
    after_trigger2 = (datetime.datetime.now() + datetime.timedelta(seconds=4)).strftime("%Y-%m-%d %H:%M:%S")
    trigger1 = DateTrigger(run_date=after_trigger1)
    trigger2 = DateTrigger(run_date=after_trigger2)
    # 添加定时任务
    schedule.add_job(task_test, trigger1)
    schedule.add_job(task_func2, trigger2)
    # 启动调度器
    schedule.print_jobs()

    try:
        schedule.start()
    except (KeyboardInterrupt, SystemExit):
        schedule.shutdown()
