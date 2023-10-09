import argparse
import json

import torch
from datasets import load_dataset
from metric_utils import report_metric
from peft import PeftModel
from tqdm import tqdm

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)


def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True, help="model name or path")
    parser.add_argument("--experiment_name", type=str, help="experiment name")
    parser.add_argument("--lora_ckpt_path", type=str, default=None, help="model name or path")
    parser.add_argument("--dev_data", type=str, help="dev data path")
    return parser.parse_args()


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
        res_gt.append({"type": v["type"], "span": v["span"], "start": None, "end": None})

    # 解析target数据，输入与输出有特殊分割符
    tg = target.split("->")[-1]
    try:
        tg_list = eval(tg)
    except:
        return res_gt, res_tg

    if isinstance(tg_list, list):
        for v in tg_list:
            if "type" in v.keys() and "span" in v.keys():
                res_tg.append({"type": v["type"], "span": v["span"], "start": None, "end": None})

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
    # eval_data = eval_data[:10]
    for data in tqdm(eval_data):
        # if index > 10:
        #     break
        instruct = data["instruction"]
        inp = data["input"]
        inputs = instruct + inp + " ->"
        input_ids = tokenizer.encode(inputs, return_tensors="pt").cuda()
        output = model.generate(
            input_ids,
            max_new_tokens=128,
            return_dict_in_generate=True,
            repetition_penalty=1.1,
        )
        output_text = tokenizer.decode(
            output.sequences[0], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        gt, tg = postprocess_outputdata([data["output"], output_text])
        output_data.append({"ground_truth": gt, "baichuan": tg})
        raw_output.append(output_text)

    input_dict = {"data": output_data, "total": len(output_data)}

    with open(f"output/{args.experiment_name}.output.txt", "w", encoding="utf-8") as g:
        for s in raw_output:
            g.write(s + "\n")

    report_metric(input_dict)


if __name__ == "__main__":
    main()
