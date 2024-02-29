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
        export PROJECT_PATH=/home/zb/code/LLaMA-Factory/ \
        && cd ${PROJECT_PATH} \
        && export PYTHONPATH=${PROJECT_PATH} \
        && export BAICHUAN2_13B=/data/SHARE/MODELS/BAICHUAN/Baichuan2-13B-Base/ \
        && export BAICHUAN2_13B_CHAT=/data/SHARE/MODELS/BAICHUAN/Baichuan2-13B-Chat/ \
        && export DS_CONFIG_STAGE_3=/home/zb/NlpTaskSpace-llm/config/deepspeed/zero_stage3_config.json \
        && export DS_CONFIG_STAGE_2=/home/zb/NlpTaskSpace-llm/config/deepspeed/zero_stage2_config.json \
        && MASTER_PORT=$(shuf -n 1 -i 10000-65535) \
        && deepspeed --include=localhost:0,1,2,3,4,5,6,7 --master_port=${MASTER_PORT} --hostfile="" src/train_bash.py \
            --deepspeed ${DS_CONFIG_STAGE_2} \
            --stage ppo \
            --do_train \
            --template qwen \
            --resize_vocab true \
            --dataset who_are_you,livestream,param_qa,alpaca_zh_retained,sn_generate_part0,sn_generate_part1,short_title_part0,short_title_part1,long_title_part0,long_title_part1,long_title_part2,sn_title,sn_xhs,sn_seo_phb,sn_seo_cp,sn_seo_other,sn_seo_zc,sn_chat_ir,sn_chat_rc \
            --overwrite_cache \
            --model_name_or_path /data/SHARE/MODELS/Qwen/Qwen-14B/ \
            --adapter_name_or_path /home/zb/saved_checkpoint/base_qwen_sft \
            --create_new_adapter \
            --report_to tensorboard \
            --output_dir /home/zb/saved_checkpoint/ppo_qwen_sn_v12_lora_lr1e5_1epoch \
            --overwrite_output_dir \
            --reward_model /home/zb/saved_checkpoint/reward_qwen_sn_v12_lora_lr1e6_1epoch/ \
            --reward_model_type lora \
            --cutoff 2048 \
            --top_k 0 \
            --top_p 0.9 \
            --finetuning_type lora \
            --lora_target all \
            --num_train_epochs 1 \
            --warmup_ratio 0.1 \
            --logging_steps 10 \
            --lr_scheduler_type cosine \
            --per_device_train_batch_size 4 \
            --gradient_accumulation_steps 4 \
            --save_steps 1000 \
            --save_total_limit 2 \
            --learning_rate 1e-5 \
            --additional_target wte,lm_head \
            --plot_loss \
            --bf16 true \
            --tf32 true
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


def task_func2():
    print("执行任务2....")
    out = sp.run(
        """
        echo hello word 2
        """,
        shell=True,
        check=True,
    )
    print(out)


if __name__ == "__main__":
    # 为了防止全量和增量并发造成显存溢出，进而训练失败，设置同一时间只能有一个任务运行
    schedule = BlockingScheduler(executors={"default": ThreadPoolExecutor(1)})
    # schedule = BlockingScheduler()
    ##添加任务 依次往后顺延时间就行
    after_trigger1 = (datetime.datetime.now() + datetime.timedelta(hours=4)).strftime("%Y-%m-%d %H:%M:%S")
    # after_trigger2 = (datetime.datetime.now() + datetime.timedelta(hours=10)).strftime("%Y-%m-%d %H:%M:%S")
    trigger1 = DateTrigger(run_date=after_trigger1)
    # trigger2 = DateTrigger(run_date=after_trigger2)
    # 添加定时任务
    schedule.add_job(task_func1, trigger1)
    # schedule.add_job(task_func2, trigger2)
    # 启动调度器
    schedule.print_jobs()

    try:
        schedule.start()
    except (KeyboardInterrupt, SystemExit):
        schedule.shutdown()


# deepspeed --include=localhost:0,1,2,3,4,5,6,7 --master_port=${MASTER_PORT} --hostfile="" src/train_bash.py \
# 	--deepspeed ${DS_CONFIG_STAGE_2} \
# 	--stage rm \
# 	--do_train \
# 	--template qwen \
# 	--dataset comparison_gpt4_zh \
# 	--model_name_or_path /data/SHARE/MODELS/Qwen/Qwen-14B/ \
# 	--resize_vocab true \
# 	--adapter_name_or_path /home/zb/saved_checkpoint/base_qwen_sft \
# 	--create_new_adapter \
# 	--report_to tensorboard \
# 	--output_dir /home/zb/saved_checkpoint/reward_qwen_sn_v12_lora_lr1e6_1epoch \
# 	--overwrite_output_dir \
# 	--overwrite_cache \
# 	--cutoff 2048 \
# 	--num_train_epochs 1 \
# 	--finetuning_type lora \
# 	--lora_target all \
# 	--warmup_ratio 0.1 \
# 	--logging_steps 10 \
# 	--lr_scheduler_type cosine \
# 	--per_device_train_batch_size 8 \
# 	--per_device_eval_batch_size 8 \
# 	--save_steps 1000 \
# 	--save_total_limit 2 \
# 	--learning_rate 1e-6 \
# 	--additional_target wte,lm_head \
# 	--bf16 true \
# 	--tf32 true

# ppo 模型训练
