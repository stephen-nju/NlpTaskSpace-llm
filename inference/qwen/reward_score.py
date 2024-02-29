import argparse
import copy
import csv
import json
import os
import warnings
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Tuple,
    Union,
)

import torch
import torch.utils.checkpoint
import transformers
from peft import PeftModel
from qwen_generation_utils import (
    HistoryType,
    StopWordsLogitsProcessor,
    decode_tokens,
    get_stop_words_ids,
    make_context,
)
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedTokenizer,
    StoppingCriteriaList,
)
from transformers.utils import cached_file
from trl import AutoModelForCausalLMWithValueHead

try:
    from einops import rearrange
except ImportError:
    rearrange = None
from torch import nn
from transformers.trainer_pt_utils import LabelSmoother

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

_ERROR_BAD_CHAT_FORMAT = """\
                   We detect you are probably using the pretrained model (rather than chat model) for chatting, since the chat_format in generation_config is not "chatml".
                   If you are directly using the model downloaded from Huggingface, please make sure you are using our "Qwen/Qwen-7B-Chat" Huggingface model (rather than "Qwen/Qwen-7B") when you call model.chat().
                   我们检测到您可能在使用预训练模型（而非chat模型）进行多轮chat，因为您当前在generation_config指定的chat_format，并未设置为我们在对话中所支持的"chatml"格式。
                   如果您在直接使用我们从Huggingface提供的模型，请确保您在调用model.chat()时，使用的是"Qwen/Qwen-7B-Chat"模型（而非"Qwen/Qwen-7B"预训练模型）。
                   """

_SENTINEL = object()
_ERROR_STREAM_IN_CHAT = """\
                   Pass argument `stream` to model.chat() is buggy, deprecated, and marked for removal. Please use model.chat_stream(...) instead of model.chat(..., stream=True).
                   向model.chat()传入参数stream的用法可能存在Bug，该用法已被废弃，将在未来被移除。请使用model.chat_stream(...)代替model.chat(..., stream=True)。
                   """

_ERROR_INPUT_CPU_QUERY_WITH_FLASH_ATTN_ACTIVATED = """\
                   We detect you have activated flash attention support, but running model computation on CPU. Please make sure that your input data has been placed on GPU. If you actually want to run CPU computation, please following the readme and set device_map="cpu" to disable flash attention when loading the model (calling AutoModelForCausalLM.from_pretrained).
                   检测到您的模型已激活了flash attention支持，但正在执行CPU运算任务。如使用flash attention，请您确认模型输入已经传到GPU上。如果您确认要执行CPU运算，请您在载入模型（调用AutoModelForCausalLM.from_pretrained）时，按照readme说法，指定device_map="cpu"以禁用flash attention。
                   """


# def preprocess(
#     sources,
#     tokenizer: transformers.PreTrainedTokenizer,
#     max_len: int,
#     system_message: str = "You are a helpful assistant.",
# ) -> Dict:
#     roles = {"user": "<|im_start|>user", "assistant": "<|im_start|>assistant"}
#     im_start = tokenizer.im_start_id
#     im_end = tokenizer.im_end_id
#     nl_tokens = tokenizer("\n").input_ids
#     _system = tokenizer("system").input_ids + nl_tokens
#     _user = tokenizer("user").input_ids + nl_tokens
#     _assistant = tokenizer("assistant").input_ids + nl_tokens

#     tokenizer.pad_token_id = tokenizer.eod_id
#     # Apply prompt templates
#     input_ids, targets = [], []
#     for i, source in enumerate(sources):
#         if roles[source[0]["from"]] != roles["user"]:
#             source = source[1:]
#         input_id, target = [], []
#         system = [im_start] + _system + tokenizer(system_message).input_ids + [im_end] + nl_tokens
#         input_id += system
#         target += [im_start] + [IGNORE_TOKEN_ID] * (len(system) - 3) + [im_end] + nl_tokens
#         assert len(input_id) == len(target)
#         for j, sentence in enumerate(source):
#             role = roles[sentence["from"]]
#             _input_id = (
#                 tokenizer(role).input_ids + nl_tokens + tokenizer(sentence["value"]).input_ids + [im_end] + nl_tokens
#             )
#             input_id += _input_id
#             if role == "<|im_start|>user":
#                 _target = [im_start] + [IGNORE_TOKEN_ID] * (len(_input_id) - 3) + [im_end] + nl_tokens
#             elif role == "<|im_start|>assistant":
#                 _target = (
#                     [im_start]
#                     + [IGNORE_TOKEN_ID] * len(tokenizer(role).input_ids)
#                     + _input_id[len(tokenizer(role).input_ids) + 1 : -2]
#                     + [im_end]
#                     + nl_tokens
#                 )
#             else:
#                 raise NotImplementedError
#             target += _target
#         assert len(input_id) == len(target)
#         input_id += [tokenizer.pad_token_id] * (max_len - len(input_id))
#         target += [IGNORE_TOKEN_ID] * (max_len - len(target))
#         input_ids.append(input_id[:max_len])
#         targets.append(target[:max_len])

#     input_ids = torch.tensor(input_ids, dtype=torch.int)
#     targets = torch.tensor(targets, dtype=torch.int)

#     return dict(
#         input_ids=input_ids,
#         labels=targets,
#         attention_mask=input_ids.ne(tokenizer.pad_token_id),
#     )


def preprocess(
    query,
    response,
    tokenizer,
    max_len,
):
    im_start = tokenizer.im_start_id
    im_end = tokenizer.im_end_id
    nl_tokens = tokenizer("\n").input_ids
    _system = tokenizer("system").input_ids + nl_tokens
    _user = tokenizer("user").input_ids + nl_tokens
    _assistant = tokenizer("assistant").input_ids + nl_tokens

    system_message: str = "You are a helpful assistant."
    input_id = []
    system = [im_start] + _system + tokenizer(system_message).input_ids + [im_end] + nl_tokens
    input_id += system
    # print(tokenizer("<|im_start|>user").input_ids)

    # print(type(tokenizer("<|im_start|>user").input_ids))
    # print(type(tokenizer(query).input_ids))
    # print(type(nl_tokens))
    # print(type())
    _input_id = (
        tokenizer("<|im_start|>user").input_ids
        + nl_tokens
        + tokenizer(query).input_ids
        + [im_end]
        + nl_tokens
        + tokenizer("<|im_start|>assistant").input_ids
        + nl_tokens
        + tokenizer(response).input_ids
        + [im_end]
    )

    input_id += _input_id
    input_ids = torch.tensor(input_id, dtype=torch.int)
    return input_ids


@torch.no_grad()
def get_reward_score(
    model,
    tokenizer: PreTrainedTokenizer,
    query: str,
    response: str,
) -> Tuple[str, HistoryType]:
    input_ids = preprocess(query, response, tokenizer, 8192)
    input_ids = input_ids.to(model.pretrained_model.device)
    _, _, values = model(input_ids, output_hidden_states=True, return_dict=True)

    # print(values.shape)
    #### values.shape =[batch_size,seqence_length]

    return values[-1].float().detach().cpu()


def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reward_model_name_or_path", type=str, required=True, help="model name or path")
    parser.add_argument("--vhead_path", type=str, help="value head checkpoint path")
    parser.add_argument("--experiment_name", type=str, help="experiment name")
    parser.add_argument("--data_path", type=str, help="dev data path")
    parser.add_argument("--format", type=str, choices=["raw", "product"], help="poduct for chanping_test format")
    return parser.parse_args()


def main():
    """
    reward score计算不支持使用value head lora的方式
    """
    args = parse_argument()
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        args.reward_model_name_or_path,
        torch_dtype=(torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32),
        trust_remote_code=True,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.reward_model_name_or_path,
        trust_remote_code=True,
        use_fast=False,
    )
    value_head_path = args.vhead_path

    try:
        from safetensors import safe_open

        vhead_file = cached_file(
            filename="value_head.safetensors",
            path_or_repo_id=value_head_path,
        )
        with safe_open(vhead_file, framework="pt", device="cpu") as f:
            vhead_params = {key: f.get_tensor(key) for key in f.keys()}
    except Exception as err:
        print("Failed to load {}: {}".format("value_head", str(err)))

    try:
        vhead_file = cached_file(
            filename="value_head.bin",
            path_or_repo_id=value_head_path,
        )
        vhead_params = torch.load(vhead_file, map_location="cpu")
    except Exception as err:
        print("Failed to load {}: {}".format("value_head", str(err)))

    model.load_state_dict(vhead_params, strict=False)
    print(f"im_end token id=={tokenizer.im_end_id}")
    # 加载自定义的generation config
    # # generation_config = GenerationConfig.from_pretrained("/home/zb/NlpTaskSpace-llm/inference/qwen/")
    # generation_config = GenerationConfig(
    #     chat_format="chatml",
    #     eos_token_id=151643,
    #     pad_token_id=151643,
    #     max_window_size=6144,
    #     max_new_tokens=2048,
    #     do_sample=True,
    #     top_k=0,
    #     top_p=0.8,
    #     repetition_penalty=1.1,
    #     transformers_version="4.31.0",
    # )

    # # 添加属性
    # model.generation_config = generation_config

    output_data = []
    num = 0
    with open(args.data_path, "r", encoding="utf-8") as g:
        examples = json.load(g)
        for example in tqdm(examples):
            example_copy = copy.deepcopy(example)
            instruction = example["instruction"]
            inputs = example["input"]
            output = example["output"][0]
            content = instruction + inputs
            scores = get_reward_score(model, tokenizer, content, output)
            example_copy["scores"] = scores
            # print(example_copy)
            output_data.append(example_copy)
            print(f"num=={num},response={scores}")
            # if num % 5 == 0:
            #     print(f"num=={num},response={scores}")
            num += 1
    # if args.format == "product":
    #     with open(
    #         os.path.join("/home/zb/NlpTaskSpace-llm/output/chanping/", f"{args.experiment_name}_response.txt"),
    #         "w",
    #         encoding="utf-8",
    #         newline="",
    #     ) as g:
    #         writer = csv.writer(g, delimiter="\t")
    #         for o in output_data:
    #             writer.writerow(
    #                 [
    #                     o["type"],
    #                     o["product_id"],
    #                     json.dumps(o["instruction"], ensure_ascii=False),
    #                     json.dumps(o["output"], ensure_ascii=False),
    #                 ]
    #             )
    # # 多保留json便于测评
    # with open(
    #     os.path.join("/home/zb/NlpTaskSpace-llm/output/ceping/", f"{args.experiment_name}_response.json"),
    #     "w",
    #     encoding="utf-8",
    # ) as g:
    #     json.dump(output_data, g, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
