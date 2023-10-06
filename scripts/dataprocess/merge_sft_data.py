import json
import math

import numpy as np
from sklearn.model_selection import train_test_split


# 淘宝ner数据

# 合并sft数据

data = []

# 淘宝ner数据
with open("/data/SHARE/DATA/train/sft/ner/tb_ner.jsonl", "r", encoding="utf-8") as g:
    for line in g:
        data.append(json.loads(line))


with open("/data/SHARE/DATA/train/sft/ner/jd_ner.jsonl", "r", encoding="utf-8") as g:
    for line in g:
        data.append(json.loads(line))

with open("/data/SHARE/DATA/train/sft/ner/sn_query_ner.jsonl", "r", encoding="utf-8") as g:
    for line in g:
        data.append(json.loads(line))

# 需要保证字典的key顺序一致,字段的类型需要保持一致
data_formated = []
for n in data:
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

    data_formated.append(o)

# performance test dataset
# with open("/home/zb/train_data/performance_test/train.json", "w", encoding="utf-8") as g:
#     for a in data_formated[:10000]:
#         g.write(json.dumps(a, ensure_ascii=False) + "\n")

# with open("/home/zb/train_data/performance_test/dev.json", "w", encoding="utf-8") as g:
#     for a in data_formated[:1000]:
#         g.write(json.dumps(a, ensure_ascii=False) + "\n")


# np.random.shuffle(data_formated)
# rate = 0.8
# index = math.ceil(rate * len(data_formated))

# train = data_formated[:index]
# test = data_formated[index:]

# with open("/home/zb/train_data/baichuan_sft/train.json", "w", encoding="utf-8") as g:
#     for d in train:
#         g.write(json.dumps(d, ensure_ascii=False) + "\n")

# with open("/home/zb/train_data/baichuan_sft/dev.json", "w", encoding="utf-8") as g:
#     for d in test:
#         g.write(json.dumps(d, ensure_ascii=False) + "\n")
