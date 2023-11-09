# ---coding:utf-8  -----------

import argparse
import json
import os
from queue import Queue
from threading import Thread
from typing import List, Optional, Tuple, Union

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


def build_chat_input(model, tokenizer, messages: List[dict], max_new_tokens: int = 0):
    def _parse_messages(messages, split_role="user"):
        system, rounds = "", []
        round = []
        for i, message in enumerate(messages):
            if message["role"] == "system":
                assert i == 0
                system = message["content"]
                continue
            if message["role"] == split_role and round:
                rounds.append(round)
                round = []
            round.append(message)
        if round:
            rounds.append(round)
        return system, rounds

    max_new_tokens = max_new_tokens or model.generation_config.max_new_tokens
    max_input_tokens = model.config.model_max_length - max_new_tokens
    system, rounds = _parse_messages(messages, split_role="user")
    system_tokens = tokenizer.encode(system)
    max_history_tokens = max_input_tokens - len(system_tokens)
    history_tokens = []
    for round in rounds[::-1]:
        round_tokens = []
        for message in round:
            if message["role"] == "user":
                round_tokens.append(model.generation_config.user_token_id)
            else:
                round_tokens.append(model.generation_config.assistant_token_id)
            round_tokens.extend(tokenizer.encode(message["content"]))
        if len(history_tokens) == 0 or len(history_tokens) + len(round_tokens) <= max_history_tokens:
            history_tokens = round_tokens + history_tokens  # concat left
            if len(history_tokens) < max_history_tokens:
                continue

        break
    input_tokens = system_tokens + history_tokens
    if messages[-1]["role"] != "assistant":
        input_tokens.append(model.generation_config.assistant_token_id)
    input_tokens = input_tokens[-max_input_tokens:]  # truncate left
    return torch.LongTensor([input_tokens]).to(model.device)


class TextIterStreamer:
    def __init__(self, tokenizer, skip_prompt=False, skip_special_tokens=False):
        self.tokenizer = tokenizer
        self.skip_prompt = skip_prompt
        self.skip_special_tokens = skip_special_tokens
        self.tokens = []
        self.text_queue = Queue()
        self.next_tokens_are_prompt = True

    def put(self, value):
        if self.skip_prompt and self.next_tokens_are_prompt:
            self.next_tokens_are_prompt = False
        else:
            if len(value.shape) > 1:
                value = value[0]
            self.tokens.extend(value.tolist())
            self.text_queue.put(self.tokenizer.decode(self.tokens, skip_special_tokens=self.skip_special_tokens))

    def end(self):
        self.text_queue.put(None)

    def __iter__(self):
        return self

    def __next__(self):
        value = self.text_queue.get()
        if value is None:
            raise StopIteration()
        else:
            return value


@torch.no_grad()
def chat(model, tokenizer, messages: List[dict], stream=False, generation_config: Optional[GenerationConfig] = None):
    input_ids = build_chat_input(model, tokenizer, messages, generation_config.max_new_tokens)

    if stream:
        streamer = TextIterStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        Thread(
            target=model.generate,
            kwargs=dict(
                inputs=input_ids,
                streamer=streamer,
                generation_config=generation_config,
            ),
        ).start()
        return streamer

    else:
        outputs = model.generate(input_ids, generation_config=generation_config)
        response = tokenizer.decode(outputs[0][len(input_ids[0]) :], skip_special_tokens=True)
        return response


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

    generation_config = GenerationConfig.from_pretrained("/home/zb/NlpTaskSpace-llm/scripts/run_base/")
    # 添加属性
    model.generation_config = generation_config

    output_data = []
    num = 0
    with open(args.data_path, "r", encoding="utf-8") as g:
        data = json.load(g)
        for example in data:
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
