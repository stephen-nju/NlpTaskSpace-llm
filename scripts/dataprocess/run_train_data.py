# 处理训练脚本所需要的训练数据，并指定文件夹
import json

import numpy as np


tb_data_raw = []
jd_data_raw = []
sn_data_raw = []


# 淘宝ner数据
with open("/data/SHARE/DATA/train/sft/ner/tb_ner.jsonl", "r", encoding="utf-8") as g:
    for line in g:
        tb_data_raw.append(json.loads(line))

with open("/data/SHARE/DATA/train/sft/ner/jd_ner.jsonl", "r", encoding="utf-8") as g:
    for line in g:
        jd_data_raw.append(json.loads(line))

with open("/data/SHARE/DATA/train/sft/ner/sn_query_ner.jsonl", "r", encoding="utf-8") as g:
    for line in g:
        sn_data_raw.append(json.loads(line))


tb_data = []
jd_data = []
sn_data = []


def convert(data_raw):
    data_new = []
    for n in data_raw:
        o = {}
        o["instruction"] = str(n["instruction"])
        o["input"] = str(n["input"])
        output = n["output"]
        n_output = []
        for out in output:
            n = {}
            for k, v in out.items():
                n[k] = str(v)
            n_output.append(n)
        o["output"] = n_output

        data_new.append(o)
    return data_new


tb_data = convert(tb_data_raw)
jd_data = convert(jd_data_raw)
sn_data = convert(sn_data_raw)


print(f"taoba={len(tb_data)},jd_data={len(jd_data)},sn_data={len(sn_data)}")
# taoba=7801,jd_data=166401,sn_data=8410

np.random.shuffle(tb_data)
# 选择900条为验证数据
tb_data_dev = tb_data[:900]
tb_data_train = tb_data[900:]

np.random.shuffle(sn_data)
sn_data_dev = sn_data[:900]
sn_data_train = sn_data[900:]

# 京东数据量较多，先筛选9000条用于微调
np.random.shuffle(jd_data)
jd_fintue_data = jd_data[:9000]
jd_pretrain_data = jd_data[9000:]
jd_fintue_data_dev = jd_fintue_data[:900]
jd_fintue_data_train = jd_fintue_data[900:]

# 多任务的微调训练数据
mutli_task_train = tb_data_train + sn_data_train + jd_fintue_data_train
np.random.shuffle(mutli_task_train)
multi_task_dev = tb_data_dev + sn_data_dev + jd_fintue_data_dev
np.random.shuffle(multi_task_dev)
# 单任务微调数据集

with open("/home/zb/train_data/baichuan_sft/single_task_sn/train.json", "w", encoding="utf-8") as g:
    for d in sn_data_train:
        g.write(json.dumps(d, ensure_ascii=False) + "\n")

with open("/home/zb/train_data/baichuan_sft/single_task_sn/dev.json", "w", encoding="utf-8") as g:
    for d in sn_data_dev:
        g.write(json.dumps(d, ensure_ascii=False) + "\n")

with open("/home/zb/train_data/baichuan_sft/single_task_jd/train.json", "w", encoding="utf-8") as g:
    for d in jd_fintue_data_train:
        g.write(json.dumps(d, ensure_ascii=False) + "\n")

with open("/home/zb/train_data/baichuan_sft/single_task_jd/dev.json", "w", encoding="utf-8") as g:
    for d in jd_fintue_data_dev:
        g.write(json.dumps(d, ensure_ascii=False) + "\n")

with open("/home/zb/train_data/baichuan_sft/single_task_tb/train.json", "w", encoding="utf-8") as g:
    for d in tb_data_train:
        g.write(json.dumps(d, ensure_ascii=False) + "\n")

with open("/home/zb/train_data/baichuan_sft/single_task_tb/dev.json", "w", encoding="utf-8") as g:
    for d in tb_data_dev:
        g.write(json.dumps(d, ensure_ascii=False) + "\n")

with open("/home/zb/train_data/baichuan_sft/multi_task/train.json", "w", encoding="utf-8") as g:
    for d in mutli_task_train:
        g.write(json.dumps(d, ensure_ascii=False) + "\n")

with open("/home/zb/train_data/baichuan_sft/multi_task/dev.json", "w", encoding="utf-8") as g:
    for d in multi_task_dev:
        g.write(json.dumps(d, ensure_ascii=False) + "\n")

with open("/home/zb/train_data/baichuan_pt/train.json", "w", encoding="utf-8") as g:
    for d in jd_pretrain_data:
        g.write(json.dumps(d, ensure_ascii=False) + "\n")

# 针对京东的数据，分割成不同的数据量进行测试
with open("/home/zb/train_data/baichuan_sft/train_size/all/train.json", "w", encoding="utf-8") as g:
    for d in jd_fintue_data_train:
        g.write(json.dumps(d, ensure_ascii=False) + "\n")
    for d in jd_pretrain_data:
        g.write(json.dumps(d, ensure_ascii=False) + "\n")

with open("/home/zb/train_data/baichuan_sft/train_size/medium/train.json", "w", encoding="utf-8") as g:
    for d in jd_fintue_data_train:
        g.write(json.dumps(d, ensure_ascii=False) + "\n")

    medium = int(len(jd_pretrain_data) / 2)
    print(f"medium_num={medium}")
    for d in jd_pretrain_data[:medium]:
        g.write(json.dumps(d, ensure_ascii=False) + "\n")

with open("/home/zb/train_data/baichuan_sft/train_size/dev.json", "w", encoding="utf-8") as g:
    for d in jd_fintue_data_dev:
        g.write(json.dumps(d, ensure_ascii=False) + "\n")
