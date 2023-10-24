import argparse
import json

import torch
from datasets import load_dataset
from metric_utils import report_metric
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True, help="model name or path")
    parser.add_argument("--experiment_name", type=str, help="experiment name")
    parser.add_argument("--lora_ckpt_path", type=str, default=None, help="model name or path")
    parser.add_argument("--dev_data", type=str, help="dev data path")
    return parser.parse_args()


def check_format(data):
    assert isinstance(data, list), print(data)
    for v in data:
        assert isinstance(v, dict)
        if "name" not in v.keys() or "span" not in v.keys():
            return False

    return True


def postprocess_outputdata(output_data):
    res_gt = []
    res_tg = []
    gt = output_data[0]
    # gt ground_truth
    target = output_data[1]
    if isinstance(gt, str):
        gt_list = json.loads(gt)
    elif isinstance(gt, list):
        gt_list = gt
    else:
        return res_gt, res_tg

    for v in gt_list:
        res_gt.append({"name": v["name"], "span": v["span"], "start": None, "end": None})

    # 解析target数据，输入与输出有特殊分割符
    tg = target.split("->")[-1]
    try:
        tg_list = json.loads(tg)
    except:
        return res_gt, res_tg

    if isinstance(tg_list, list):
        for v in tg_list:
            if isinstance(v, dict):
                if "name" in v.keys() and "span" in v.keys():
                    if isinstance(v["name"], str) and isinstance(v["span"], str):
                        res_tg.append({"name": v["name"], "span": v["span"], "start": None, "end": None})
                    # 格式控制会出现错误,TODO,比如值出现list的情况

    return res_gt, res_tg


def main():
    args = parse_argument()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=(torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32),
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        use_fast=True,
        add_bos_token=False,
        add_eos_token=False,
        padding_side="left",
    )

    if args.lora_ckpt_path is not None:
        peft_model = PeftModel.from_pretrained(model, args.lora_ckpt_path)
        model = peft_model.merge_and_unload()

    ################处理数据#########################################

    eval_data = []
    with open(args.dev_data, "r", encoding="utf-8") as g:
        for line in g:
            s = line.strip()
            data = json.loads(s)
            eval_data.append(data)

    output_data = []
    raw_output = []
    eval_output = []
    # eval_data = eval_data[:10]
    for index, data in tqdm(enumerate(eval_data), total=len(eval_data)):
        # if index > 10:
        #     break
        eval_o = {}
        instruct = data["instruction"]
        inp = data["input"]
        inputs = instruct + inp + " ->"
        input_ids = tokenizer.encode(inputs, return_tensors="pt").cuda()
        output = model.generate(
            input_ids,
            max_new_tokens=256,
            return_dict_in_generate=True,
            # repetition_penalty=1.1,
        )
        output_text = tokenizer.decode(
            output.sequences[0], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        if index % 100 == 0:
            print(output_text)
        gt, tg = postprocess_outputdata([data["output"], output_text])

        if check_format(tg):
            output_data.append({"ground_truth": gt, "target": tg})
        else:
            print(f"target error={tg}")

        eval_o["instruction"] = instruct
        eval_o["input"] = inp
        eval_o["output"] = json.loads(data["output"])
        raw_output.append(output_text)
        eval_o["model"] = tg
        eval_output.append(eval_o)

    with open(f"output/{args.experiment_name}.output_raw.txt", "w", encoding="utf-8") as g:
        for s in raw_output:
            g.write(s + "\n")

    with open(f"output/{args.experiment_name}.output_eval.txt", "w", encoding="utf-8") as g:
        for e in eval_output:
            ins = e["instruction"]
            q = e["input"]
            output = e["output"]
            tg = e["model"]
            name_set = set()
            for o in output:
                name_set.add(o["name"])

            for t in tg:
                name_set.add(t["name"])

            name_map = {}
            names = sorted(list(name_set))
            for s in names:
                o_s = []
                for o in output:
                    if o["name"] == s:
                        o_s.append(o["span"])
                t_s = []
                for t in tg:
                    if t["name"] == s:
                        t_s.append(t["span"])
                name_map[s] = [o_s, t_s]
            s = []
            for k, v in name_map.items():
                s.append("\t".join([k, json.dumps(v[0], ensure_ascii=False), json.dumps(v[1], ensure_ascii=False)]))

            g.write(q + "\t" + "\t".join(s) + "\n")

    input_dict = {"data": output_data, "total": len(output_data)}

    with open(f"output/{args.experiment_name}.output_dict.txt", "w", encoding="utf-8") as g:
        json.dump(input_dict, g, ensure_ascii=False, indent=4)

    report_metric(input_dict)


def evaluate():
    data = []
    output = []
    with open(
        "/data/SHARE/tmpt/sft_13b_lora_ner_epochs_20_eval_train_predictions.json",
        "r",
        encoding="utf-8",
    ) as g:
        data = json.load(g)

    for d in data:
        # print(d)
        input = eval(d["Input"])
        groundtruth = input["output"]
        try:
            target = eval(d["Output"])
        except:
            target = []

        tg = []
        for target_dict in target:
            flag = True
            if isinstance(target_dict, dict):
                if "type" not in target_dict.keys() or "span" not in target_dict.keys():
                    flag = False
            if flag:
                tg.append(target_dict)

        output.append({"ground_truth": groundtruth, "target": tg})
    input_dict = {"data": output, "total": len(output)}

    # print(input_dict)
    report_metric(input_dict)


if __name__ == "__main__":
    main()
