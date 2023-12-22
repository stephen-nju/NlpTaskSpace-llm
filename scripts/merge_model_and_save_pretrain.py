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
    parser.add_argument("--chat", action="store_true", help="whether the mode is chat or base")
    parser.add_argument("--lora_ckpt_path", type=str, default=None, help="model name or path")
    parser.add_argument("--output_dir", type=str, help="save pretrained dir")
    return parser.parse_args()


def main():
    args = parse_argument()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=(torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32),
    )

    if args.chat:
        generation_config = GenerationConfig.from_pretrained(args.model_name_or_path)
        model.generation_config = generation_config

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        use_fast=False,
        padding_side="left",
    )

    peft_model = PeftModel.from_pretrained(model, args.lora_ckpt_path)
    model = peft_model.merge_and_unload()

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
