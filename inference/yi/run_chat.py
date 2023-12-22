import argparse
import json
import os
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from fastchat.conversation import (
    Conversation,
    SeparatorStyle,
    get_conv_template,
    register_conv_template,
)
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig


def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True, help="model name or path")
    parser.add_argument("--experiment_name", type=str, help="experiment name")
    parser.add_argument("--lora_ckpt_path", type=str, default=None, help="model name or path")
    parser.add_argument("--data_path", type=str, help="dev data path")
    return parser.parse_args()


def main():
    # args = parse_argument()
    # model = AutoModelForCausalLM.from_pretrained(
    #     args.model_name_or_path,
    #     trust_remote_code=True,
    #     device_map="auto",
    #     torch_dtype=(torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32),
    # )

    # tokenizer = AutoTokenizer.from_pretrained(
    #     args.model_name_or_path,
    #     trust_remote_code=True,
    #     use_fast=False,
    #     padding_side="left",
    # )

    # # if args.lora_ckpt_path is not None:
    # #     peft_model = PeftModel.from_pretrained(model, args.lora_ckpt_path)
    # #     model = peft_model.merge_and_unload()

    # # 加载自定义的generation config

    # generation_config = GenerationConfig(
    #     bos_token_id=6,
    #     do_sample=True,
    #     eos_token_id=7,
    #     pad_token_id=0,
    #     temperature=0.6,
    #     max_length=4096,
    #     top_p=0.8,
    #     transformers_version="4.35.0",
    # )

    chat_template = get_conv_template("qwen-7b-chat")

    chat_template.append_message(chat_template.roles[0], "Hello!")
    chat_template.append_message(chat_template.roles[1], "你好")
    chat_template.append_message(chat_template.roles[0], "你是谁")
    chat_template.append_message(chat_template.roles[1], "我是苏宁大模型")
    print(chat_template.get_prompt())

    print("\n")

    # model.generation_config = generation_config
    # output_data = []
    # num = 0

    # with open(args.data_path, "r", encoding="utf-8") as g:
    #     for line in tqdm(g):
    #         example = json.loads(line)
    #         instruction = example["instruction"]
    #         inputs = example["input"]
    #         content = instruction + inputs
    #         content = instruction + inputs
    #         messages = []
    #         messages.append(
    #             {
    #                 "role": "user",
    #                 "content": content,
    #             }
    #         )

    #
    # input_ids = tokenizer.encode(
    #             chat_template(messages), return_tensors="pt", add_special_tokens=False
    #             ).to("cuda")
    # output_ids = model.generate(input_ids.to("cuda"),**generation_config)
    # response = tokenizer.decode(
    #             output_ids[0][input_ids.shape[1] :], skip_special_tokens=False
    #             )

    # output_data.append({"content": content, "response": response})
    # if num % 20 == 0:
    #     print(f"num=={num},response={response}")
    # num += 1

    # with open(
    #     os.path.join("/home/zb/NlpTaskSpace-llm/output/chat/", f"{args.experiment_name}_response.json"),
    #     "w",
    #     encoding="utf-8",
    # ) as g:
    #     json.dump(output_data, g, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
