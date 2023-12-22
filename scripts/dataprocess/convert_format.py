import json


def convert_format_v1():
    train_data = []

    dev_data = []
    with open("/home/zb/train_data/baichuan_sft/single_task_sn/train.json", "r", encoding="utf-8") as g:
        for line in g:
            train_data.append(json.loads(line))

    with open("/home/zb/train_data/baichuan_sft/single_task_sn/dev.json", "r", encoding="utf-8") as g:
        for line in g:
            dev_data.append(json.loads(line))

    print(f"train_data=={train_data[0]}")

    # 添加强控制指令
    prompt1 = (
        "搜索词实体识别任务:指从用户搜索词中识别出实体信息。\n\n请仔细阅读下面文本，抽取该文本中所有关于品牌,品类,系列,属性的命名实体，返回json格式\n\n" "注意:\n1.要求识别出的实体需要出现在用户搜索词中\n"
    )

    # 构建新的实体类型，防止和已有的知识形成冲突
    prompt2 = "搜索词实体识别任务:指从用户搜索词中识别出实体信息，并按照json格式返回结果。\n\n请仔细阅读下面文本，抽取该文本中所有关于Sn@PP,Sn@PL,Sn@SX,Sn@XL的命名实体，返回json格式\n"

    print(f"{prompt1}")
    print(f"{prompt2}")

    train_data_v1 = []
    dev_data_v1 = []

    for t in train_data:
        t["instruction"] = prompt1
        train_data_v1.append(t)

    for d in dev_data:
        d["instruction"] = prompt1
        dev_data_v1.append(d)

    with open(
        "/home/zb/train_data/baichuan_sft/single_task_sn/enhance_prompt_v1/train.json", "w", encoding="utf-8"
    ) as g:
        for l in train_data_v1:
            g.write(json.dumps(l, ensure_ascii=False) + "\n")

    with open("/home/zb/train_data/baichuan_sft/single_task_sn/enhance_prompt_v1/dev.json", "w", encoding="utf-8") as g:
        for l in dev_data_v1:
            g.write(json.dumps(l, ensure_ascii=False) + "\n")

    def convert_type(output):
        n = []
        for o in output:
            d = {}
            for k, v in o.items():
                if k == "type":
                    if v == "品牌":
                        v = "Sn@PP"
                    elif v == "属性":
                        v = "Sn@SX"
                    elif v == "品类":
                        v = "Sn@PL"
                    elif v == "系列":
                        v = "Sn@XL"
                d[k] = v
            n.append(d)

        return n

    train_data_v2 = []
    dev_data_v2 = []

    for t in train_data:
        t["instruction"] = prompt2
        out = t["output"]
        new_out = convert_type(out)
        t["output"] = new_out
        train_data_v2.append(t)

    for d in dev_data:
        d["instruction"] = prompt2
        out = d["output"]
        new_out = convert_type(out)
        d["output"] = new_out
        dev_data_v2.append(d)

    with open(
        "/home/zb/train_data/baichuan_sft/single_task_sn/enhance_prompt_v2/train.json", "w", encoding="utf-8"
    ) as g:
        for l in train_data_v2:
            g.write(json.dumps(l, ensure_ascii=False) + "\n")

    with open("/home/zb/train_data/baichuan_sft/single_task_sn/enhance_prompt_v2/dev.json", "w", encoding="utf-8") as g:
        for l in dev_data_v2:
            g.write(json.dumps(l, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    convert_format_v1()
