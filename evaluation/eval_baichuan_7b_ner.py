from tqdm import tqdm
import numpy as np
import torch
import json
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
import argparse
from peft import PeftModel
import json
from metrics.ner import report_metric
from tqdm import tqdm

key_name_map = {"品牌": "HP", "品类": "HC", "系列型号": "XL"}


def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path",
                        type=str,
                        required=True,
                        help="model name or path")
    parser.add_argument("--lora_ckpt_path",
                        type=str,
                        required=True,
                        help="model name or path")
    parser.add_argument("--dev_data", type=str, help="dev data path")
    parser.add_argument("--output_dir",
                        type=str,
                        default="ceval_output",
                        help="output directory")
    return parser.parse_args()


def postprocess_outputdata(output_data):
    res_gt = []
    res_tg = []
    gt = output_data[0]
    # gt ground_truth
    target = output_data[1]
    if isinstance(gt, str):
        gt_dict = json.loads(gt)
    elif isinstance(gt, dict):
        gt_dict = gt
    else:
        return res_gt, res_tg

    for k, v in gt_dict.items():
        if k in key_name_map:
            name = key_name_map[k]
            for tv in v:
                res_gt.append({
                    "type": name,
                    "span": tv,
                    "start": None,
                    "end": None
                })

    # 解析target数据，输入与输出有特殊分割符
    tg = target.split(" ->")[-1]
    try:
        tg_dict = eval(tg)
    except:
        return res_gt, res_tg
    if isinstance(tg_dict, dict):
        for k, v in tg_dict.items():
            if k in key_name_map:
                type_name = key_name_map[k]
                if v is None:
                    continue
                if isinstance(v, list):
                    for tv in v:
                        res_tg.append({
                            "type": type_name,
                            "span": tv,
                            "start": None,
                            "end": None
                        })
                elif isinstance(v, str):
                    res_tg.append({
                        "type": type_name,
                        "span": v,
                        "start": None,
                        "end": None
                    })
                else:
                    continue

    return res_gt, res_tg


def main():
    args = parse_argument()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=(torch.bfloat16
                     if torch.cuda.is_bf16_supported() else torch.float32),
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        use_fast=True,
        add_bos_token=False,
        add_eos_token=False,
        padding_side="left",
    )

    peft_model = PeftModel.from_pretrained(model, args.lora_ckpt_path)
    model = peft_model.merge_and_unload()

    ################处理数据#########################################

    prefix = """命名实体识别：抽取文本中的 品牌，品类，系列型号 这三类命名实体，并按照json格式返回结果。\n\n"""

    eval_data = []
    with open(args.dev_data, "r", encoding="utf-8") as g:
        for line in g:
            s = line.strip()
            data = json.loads(s)
            eval_data.append(data)

    output_data = []

    for data in tqdm(eval_data):
        # if index > 10:
        #     break
        inp = data["context"]
        inputs = prefix + inp + " ->"
        input_ids = tokenizer.encode(inputs, return_tensors="pt").cuda()
        output = model.generate(
            input_ids,
            max_new_tokens=64,
            return_dict_in_generate=True,
            repetition_penalty=1.1,
        )
        output_text = tokenizer.decode(output.sequences[0],
                                       skip_special_tokens=True,
                                       clean_up_tokenization_spaces=False)
        gt, tg = postprocess_outputdata([data["ner"], output_text])
        output_data.append({"ground_truth": gt, "baichuan": tg})

    input_dict = {"data": output_data, "total": len(output_data)}

    with open("output.json", "w", encoding="utf-8") as g:
        json.dump(input_dict, g, ensure_ascii=False, indent=2)

    report_metric(input_dict)


if __name__ == "__main__":
    main()
