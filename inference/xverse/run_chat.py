import argparse
import json
import os
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig
from transformers.modeling_utils import PreTrainedModel


def _build_chat_input(model, tokenizer, messages: List[dict], max_new_tokens: int = 2048):
    max_new_tokens = max_new_tokens or model.generation_config.max_new_tokens
    max_input_tokens = model.config.max_position_embeddings - max_new_tokens
    max_input_tokens = max(model.config.max_position_embeddings // 2, max_input_tokens)
    max_input_tokens = min(model.config.max_tokenizer_truncation, max_input_tokens)

    total_input, round_input = [], []

    user_prompt_tokens = tokenizer.encode("Human: ", return_token_type_ids=False)
    exec_prompt_tokens = tokenizer.encode("Exec: ", return_token_type_ids=False)
    assist_prompt_tokens = tokenizer.encode("Assistant: ", return_token_type_ids=False)
    assist_prompt_len = len(assist_prompt_tokens)

    for i, message in enumerate(messages[::-1]):
        if message["role"] == "user" or message["role"] == "exec":
            user_content = f"{message['content']}\n\n"

            content_tokens = (
                user_prompt_tokens + tokenizer.encode(user_content, return_token_type_ids=False)
                if message["role"] == "user"
                else exec_prompt_tokens + tokenizer.encode(user_content, return_token_type_ids=False)
            )
            if i == 0:
                content_tokens = content_tokens[: max_input_tokens - assist_prompt_len]
                content_tokens += assist_prompt_tokens

            round_input = content_tokens + round_input
            if i != 0:
                if len(total_input) + len(round_input) > max_input_tokens:
                    break
                else:
                    total_input = round_input + total_input

            else:
                total_input = round_input + total_input

                if len(total_input) >= max_input_tokens:
                    break

            round_input = []

        elif message["role"] == "assistant":
            assist_content = f"{message['content']}"

            content_tokens = assist_prompt_tokens + tokenizer.encode(assist_content, return_token_type_ids=False)

            round_input = content_tokens + [model.generation_config.eos_token_id] + round_input

        elif message["role"] == "system":
            assert i == len(messages) - 1

            user_content = f"{message['content']}\n"

            content_tokens = tokenizer.encode(user_content, return_token_type_ids=False)

            round_input = user_prompt_tokens + content_tokens + round_input

            if len(total_input) + len(round_input) > max_input_tokens:
                break

            else:
                total_input = round_input + total_input

        else:
            raise ValueError(f"message role not supported yet: {message['role']}")

    total_input = torch.LongTensor([total_input]).to(model.device)

    return total_input


@torch.no_grad()
def chat(model, tokenizer, messages: List[dict], stream=False, generation_config: Optional[GenerationConfig] = None):
    generation_config = generation_config or model.generation_config

    input_ids = _build_chat_input(model, tokenizer, messages, generation_config.max_new_tokens)

    if stream:
        from threading import Thread

        from transformers import TextIteratorStreamer

        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True)

        model.__class__.generate = PreTrainedModel.generate

        def stream_generator():
            generation_kwargs = dict(inputs=input_ids, generation_config=generation_config, streamer=streamer)

            thread = Thread(target=model.generate, kwargs=generation_kwargs)

            thread.start()

            for next_text in streamer:
                yield next_text.replace(tokenizer.eos_token, "")

        return stream_generator()

    else:
        model.__class__.generate = PreTrainedModel.generate  # disable stream

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

    generation_config = GenerationConfig(
        pad_token_id=1,
        bos_token_id=2,
        eos_token_id=3,
        max_new_tokens=2048,
        temperature=0.5,
        top_k=30,
        top_p=0.85,
        repetition_penalty=1.1,
        do_sample=True,
        transformers_version="4.29.1",
    )

    model.generation_config = generation_config
    model.config.max_tokenizer_truncation = 6144
    output_data = []
    num = 0
    with open(args.data_path, "r", encoding="utf-8") as g:
        examples = json.load(g)
        for example in tqdm(examples):
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
            response = chat(model, tokenizer, messages)
            output_data.append({"content": content, "response": response})
            if num % 20 == 0:
                print(f"num=={num},response={response}")
            num += 1

    with open(
        os.path.join("/home/zb/NlpTaskSpace-llm/output/ceping/", f"{args.experiment_name}_response.json"),
        "w",
        encoding="utf-8",
    ) as g:
        json.dump(output_data, g, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
