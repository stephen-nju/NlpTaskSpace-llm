# ---coding:utf-8  -----------

import argparse
import json
import os

import torch
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

    # generation_config = GenerationConfig.from_pretrained(args.model_name_or_path)
    generation_config = GenerationConfig(
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        user_token_id=195,
        assistant_token_id=196,
        max_new_tokens=1024,
        temperature=0.2,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.1,
        do_sample=True,
        transformers_version="4.29.2",
    )
    # print(f"generation_config={generation_config}")
    # generation_config["temperature"] = 0.8
    # generation_config["top_k"] = 20
    # generation_config["top_p"] = 0.95

    model.generation_config = generation_config

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        use_fast=False,
        padding_side="left",
    )

    if args.lora_ckpt_path is not None:
        peft_model = PeftModel.from_pretrained(model, args.lora_ckpt_path)
        model = peft_model.merge_and_unload()

    output = []
    num = 0
    with open(args.data_path, "r", encoding="utf-8") as g:
        for line in tqdm(g):
            example = json.loads(line)
            # for line in tqdm(g):
            # if num > 100:
            #     break
            # example = json.loads(line)
            messages = []
            instruction = example["instruction"]
            inputs = example["input"]
            content = instruction + inputs
            messages.append(
                {
                    "role": "user",
                    "content": content,
                }
            )

            response = model.chat(tokenizer, messages)
            output.append({"content": content, "response": response})
            num += 1

            if num % 20 == 0:
                print(response)
    with open(
        os.path.join("/home/zb/NlpTaskSpace-llm/output/chat/", f"{args.experiment_name}_response.json"),
        "w",
        encoding="utf-8",
    ) as g:
        json.dump(output, g, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
