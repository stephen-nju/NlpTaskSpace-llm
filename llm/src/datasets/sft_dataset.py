# ----coding utf-8--------**

import pickle
from dataclasses import dataclass, field
from functools import partial
from os import cpu_count
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence

import torch
from multiprocess.dummy import Pool
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from llm.src.constant import IGNORE_INDEX

if TYPE_CHECKING:
    from transformers.tokenization_utils import PreTrainedTokenizer


@dataclass
class SftInstructInputExample:
    """
    指令微调时候的数据集
    """

    # 模板名称,自行设置，可以为待训练的模型名称
    instruction: str
    input: str
    output: str


@dataclass
class SftInstructInputFeature:
    input_ids: Optional[torch.tensor]
    labels: Optional[torch.tensor]


@dataclass
class SftInstructTemplate:
    """
    用于保存不同模型构建sft时候的特殊模板
    """

    name: str
    pad_token_id: int
    eos_token_id: int


sft_templates: Dict[str, SftInstructTemplate] = {}


def register_sft_template(template: SftInstructTemplate):
    sft_templates[template.name] = template


def get_sft_template(name: str):
    return sft_templates[name]


def convert_sft_example_to_feature(
    example: "SftInstructInputExample",
    tokenizer: "PreTrainedTokenizer",
    template,
    max_source_length,
    max_target_length,
    stage,
):
    pad_token_id = template.pad_token_id
    eos_token_id = template.eos_token_id
    if stage == "train":
        instruction = example.instruction
        max_seq_length = max_source_length + max_target_length + 1
        input, answer = example.input, example.output
        answer = str(answer)
        prompt = instruction + input + " ->"

        a_ids = tokenizer.encode(
            text=prompt,
            add_special_tokens=True,
            truncation=True,
            max_length=max_source_length,
        )

        b_ids = tokenizer.encode(
            text=answer,
            add_special_tokens=False,
            truncation=True,
            max_length=max_target_length,
        )

        context_length = len(a_ids)
        input_ids = a_ids + b_ids + [eos_token_id]
        # print(f"===={model.generation_config.pad_token_id}")
        labels = [pad_token_id] * context_length + b_ids + [eos_token_id]
        # 构建 batch padding
        pad_len = max_seq_length - len(input_ids)
        input_ids = input_ids + [pad_token_id] * pad_len
        labels = labels + [pad_token_id] * pad_len
        # if args.ignore_pad_token_for_loss:
        labels = [(l if l != pad_token_id else -100) for l in labels]

        return SftInstructInputFeature(input_ids=input_ids, labels=labels)
    elif stage == "eval":
        tokenizer.padding_side = "left"
        tokenizer.pad_token_id = template.pad_token_id

        instruction = example.instruction
        input, answer = example.input, example.output
        answer = str(answer)
        prompt = instruction + input + " ->"

        model_inputs = tokenizer(
            prompt,
            max_length=max_source_length,
            truncation=True,
            padding=True,
        )
        labels = tokenizer(answer, max_length=max_target_length, truncation=True, padding=True)

        labels["input_ids"] = [(l if l != pad_token_id else -100) for l in labels["input_ids"]]
        return SftInstructInputFeature(input_ids=model_inputs["input_ids"], labels=labels["input_ids"])
    else:
        raise ValueError("stage value error")


def convert_sft_examples_to_features(
    examples, tokenizer, max_source_length, max_target_length, template, stage="train", threads=4
):
    threads = min(threads, cpu_count())
    with Pool(threads) as p:
        annotate_ = partial(
            convert_sft_example_to_feature,
            tokenizer=tokenizer,
            template=template,
            max_source_length=max_source_length,
            max_target_length=max_target_length,
            stage=stage,
        )
        features = list(
            tqdm(
                p.imap(annotate_, examples, chunksize=32),
                total=len(examples),
                desc="convert examples to features",
            )
        )
    return features


# @dataclass
# class SftChatInputExample:
#     """
#     模型做sft时候的数据,example
#     """

#     name: str
#     # The system prompt
#     system_prompt: str
#     # All messages. format: list of [question, answer]
#     messages: Optional[List[Sequence[str]]]
#     # The roles of the speakers
#     roles: Optional[Sequence[str]]
#     # Conversation prompt
#     prompt: str
#     # Separator
#     sep: str
#     # Stop token, default is tokenizer.eos_token

#     stop_str: Optional[str] = "</s>"

#     def get_prompt(self, messages: Optional[List[Sequence[str]]] = None, system_prompt: Optional[str] = "") -> str:
#         """

#         Returns a string containing prompt without response.

#         """

#         return "".join(self._format_example(messages, system_prompt))

#     def get_dialog(
#         self, messages: Optional[List[Sequence[str]]] = None, system_prompt: Optional[str] = ""
#     ) -> List[str]:
#         """

#         Returns a list containing 2 * n elements where the 2k-th is a query and the (2k+1)-th is a response.

#         """

#         return self._format_example(messages, system_prompt)

#     def _format_example(
#         self, messages: Optional[List[Sequence[str]]] = None, system_prompt: Optional[str] = ""
#     ) -> List[str]:
#         system_prompt = system_prompt or self.system_prompt

#         system_prompt = system_prompt + self.sep if system_prompt else ""  # add separator for non-empty system prompt

#         messages = messages or self.messages

#         convs = []

#         for turn_idx, [user_query, bot_resp] in enumerate(messages):
#             if turn_idx == 0:
#                 convs.append(system_prompt + self.prompt.format(query=user_query))

#                 convs.append(bot_resp)

#             else:
#                 convs.append(self.sep + self.prompt.format(query=user_query))

#                 convs.append(bot_resp)

#         return convs

#     def append_message(self, query: str, answer: str):
#         """Append a new message."""

#         self.messages.append([query, answer])


# @dataclass
# class SftChatInputFeature:
#     """
#     TODO
#     训练chat类模型
#     """

#     pass


# # 用于构建不同chat模型的数据
# chat_templates: Dict[str, SftChatInputExample] = {}


# def register_chat_template(template: SftChatInputExample):
#     chat_templates[template.name] = template
