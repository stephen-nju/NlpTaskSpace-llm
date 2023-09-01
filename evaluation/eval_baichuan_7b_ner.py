from tqdm import tqdm
import numpy as np
import torch
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

    for data in eval_data:
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

        print(output_text)
        print(f"============\n")


if __name__ == "__main__":
    main()
