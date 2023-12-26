import argparse
import copy
import csv
import json
import os
import warnings
from typing import TYPE_CHECKING, Any, Callable, Generator, List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
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

try:
    from einops import rearrange
except ImportError:
    rearrange = None
from torch import nn

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


@torch.no_grad()
def chat(
    model,
    tokenizer: PreTrainedTokenizer,
    query: str,
    history: Optional[HistoryType],
    system: str = "You are a helpful assistant.",
    stream: Optional[bool] = _SENTINEL,
    stop_words_ids: Optional[List[List[int]]] = None,
    generation_config: Optional[GenerationConfig] = None,
    **kwargs,
) -> Tuple[str, HistoryType]:
    generation_config = generation_config if generation_config is not None else model.generation_config
    assert stream is _SENTINEL, _ERROR_STREAM_IN_CHAT
    assert generation_config.chat_format == "chatml", _ERROR_BAD_CHAT_FORMAT
    if history is None:
        history = []
    else:
        # make a copy of the user's input such that is is left untouched
        history = copy.deepcopy(history)

    if stop_words_ids is None:
        stop_words_ids = []

    max_window_size = kwargs.get("max_window_size", None)
    if max_window_size is None:
        max_window_size = generation_config.max_window_size
    raw_text, context_tokens = make_context(
        tokenizer,
        query,
        history=history,
        system=system,
        max_window_size=max_window_size,
        chat_format=generation_config.chat_format,
    )

    stop_words_ids.extend(get_stop_words_ids(generation_config.chat_format, tokenizer))
    input_ids = torch.tensor([context_tokens]).to(model.device)
    outputs = model.generate(
        input_ids,
        stop_words_ids=stop_words_ids,
        return_dict_in_generate=False,
        generation_config=generation_config,
        **kwargs,
    )
    response = decode_tokens(
        outputs[0],
        tokenizer,
        raw_text_len=len(raw_text),
        context_length=len(context_tokens),
        chat_format=generation_config.chat_format,
        verbose=False,
        errors="replace",
    )

    # as history is a copy of the user inputs,
    # we can always return the new turn to the user.
    # separating input history and output history also enables the user
    # to implement more complex history management
    history.append((query, response))
    return response, history


def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True, help="model name or path")
    parser.add_argument("--experiment_name", type=str, help="experiment name")
    parser.add_argument("--lora_ckpt_path", type=str, default=None, help="model name or path")
    parser.add_argument("--data_path", type=str, help="dev data path")
    parser.add_argument("--format", type=str, choices=["raw", "product"], help="poduct for chanping_test format")
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
    # generation_config = GenerationConfig.from_pretrained("/home/zb/NlpTaskSpace-llm/inference/qwen/")
    generation_config = GenerationConfig(
        chat_format="chatml",
        eos_token_id=151643,
        pad_token_id=151643,
        max_window_size=6144,
        max_new_tokens=2048,
        do_sample=True,
        top_k=0,
        top_p=0.8,
        repetition_penalty=1.1,
        transformers_version="4.31.0",
    )

    # 添加属性
    model.generation_config = generation_config

    output_data = []
    num = 0
    with open(args.data_path, "r", encoding="utf-8") as g:
        examples = json.load(g)
        for example in tqdm(examples):
            output = copy.deepcopy(example)
            instruction = example["instruction"]
            inputs = example["input"]
            content = instruction + inputs
            response, history = chat(model, tokenizer, content, history=None)
            output["output"] = response
            output_data.append(output)
            if num % 20 == 0:
                print(f"num=={num},response={response}")
            num += 1
    if args.format == "product":
        with open(
            os.path.join("/home/zb/NlpTaskSpace-llm/output/chanping/", f"{args.experiment_name}_response.json"),
            "w",
            encoding="utf-8",
            newline="",
        ) as g:
            writer = csv.writer(g, delimiter="\t")
            for o in output_data:
                writer.writerow(
                    [
                        o["type"],
                        o["product_id"],
                        json.dumps(o["instruction"], ensure_ascii=False),
                        json.dumps(o["output"], ensure_ascii=False),
                    ]
                )
    else:
        with open(
            os.path.join("/home/zb/NlpTaskSpace-llm/output/ceping/", f"{args.experiment_name}_response.json"),
            "w",
            encoding="utf-8",
        ) as g:
            json.dump(output_data, g, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
