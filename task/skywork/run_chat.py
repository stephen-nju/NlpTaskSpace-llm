# coding=utf-8


import argparse
import json
import os

import torch
from generation_utils import chat
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True, help="model name or path")
    parser.add_argument("--experiment_name", type=str, help="experiment name")
    parser.add_argument("--lora_ckpt_path", type=str, default=None, help="model name or path")
    parser.add_argument("--data_path", type=str, help="dev data path")
    return parser.parse_args()


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
        use_fast=False,
        padding_side="left",
    )

    if args.lora_ckpt_path is not None:
        peft_model = PeftModel.from_pretrained(model, args.lora_ckpt_path)
        model = peft_model.merge_and_unload()

    # 加载自定义的generation config

    generation_config = GenerationConfig(
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        user_token_id=195,
        assistant_token_id=196,
        max_new_tokens=2048,
        temperature=0.8,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.1,
        do_sample=True,
        transformers_version="4.29.2",
    )

    # generation_config = GenerationConfig.from_pretrained("/home/zb/NlpTaskSpace-llm/scripts/run_base/")
    # 添加属性
    model.generation_config = generation_config

    output_data = []
    num = 0
    with open(args.data_path, "r", encoding="utf-8") as g:
        for line in tqdm(g):
            example = json.loads(line)
            instruction = example["instruction"]
            inputs = example["input"]
            content = instruction + inputs
            content = instruction + inputs
            messages = []
            messages.append(
                {
                    "role": "user",
                    "content": content,
                }
            )
            response = chat(model, tokenizer, messages, generation_config=generation_config)
            output_data.append({"content": content, "response": response})
            if num % 20 == 0:
                print(f"num=={num},response={response}")
            num += 1

    with open(
        os.path.join("/home/zb/NlpTaskSpace-llm/output/chat/", f"{args.experiment_name}_response.json"),
        "w",
        encoding="utf-8",
    ) as g:
        json.dump(output_data, g, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
