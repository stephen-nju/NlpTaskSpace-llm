import json
from copy import deepcopy
from typing import Any, Dict, List

from llm.src.constant import IGNORE_INDEX


def construct_example(examples: Dict[str, List[Any]]):
    for i in range(len(examples["instruction"])):
        instruction, inputs, outputs = examples["instruction"][i], examples["input"][i], examples["output"][i]
        query, response = instruction, outputs
        query = query + "\n" + inputs if inputs else query
        history = examples["history"][i] if "history" in examples else None
        system = examples["system"][i] if "system" in examples else None
        yield query, response, history, system


def preprocess_supervised_dataset_train(
    examples, tokenizer, template, max_source_length, max_target_length
) -> Dict[str, Any]:
    # build inputs with format `<bos> X Y <eos>` and labels with format `<ignore> ... <ignore> Y <eos>`
    # for multiturn examples, we only mask the prompt part in each prompt-response pair.
    # __import__("pdb").set_trace()

    model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}
    for query, response, history, system in construct_example(examples):
        input_ids, labels = [], []
        for turn_idx, (source_ids, target_ids) in enumerate(
            template.encode_multiturn(tokenizer, query, response, history, system)
        ):
            if len(source_ids) > max_source_length:
                source_ids = source_ids[:max_source_length]
            if len(target_ids) > max_target_length:
                target_ids = target_ids[:max_target_length]
            if turn_idx != 0 and template.efficient_eos:
                source_mask = [tokenizer.eos_token_id] + [IGNORE_INDEX] * (len(source_ids) - 1)
            else:
                source_mask = [IGNORE_INDEX] * len(source_ids)

            input_ids += source_ids + target_ids
            labels += source_mask + target_ids

        if template.efficient_eos:
            input_ids += [tokenizer.eos_token_id]
            labels += [tokenizer.eos_token_id]

        model_inputs["input_ids"].append(input_ids)
        model_inputs["attention_mask"].append([1] * len(input_ids))
        model_inputs["labels"].append(labels)

    return model_inputs


def preprocess_supervised_dataset_test(
    examples,
    tokenizer,
    template,
    max_source_length,
    max_target_length,
):
    pad_token_id = tokenizer.pad_token_id
    tokenizer.padding_side = "left"

    model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}
    for query, response, history, system in construct_example(examples):
        input_ids, labels = [], []
        for turn_idx, (source_ids, target_ids) in enumerate(
            template.encode_multiturn(tokenizer, query, response, history, system)
        ):
            if len(source_ids) > max_source_length:
                source_ids = source_ids[:max_source_length]
            if len(target_ids) > max_target_length:
                target_ids = target_ids[:max_target_length]
            if turn_idx != 0 and template.efficient_eos:
                source_mask = [tokenizer.eos_token_id] + [IGNORE_INDEX] * (len(source_ids) - 1)
            else:
                source_mask = [IGNORE_INDEX] * len(source_ids)

            input_ids += source_ids + target_ids
            labels += source_mask + target_ids

        if template.efficient_eos:
            input_ids += [tokenizer.eos_token_id]
            labels += [tokenizer.eos_token_id]

        model_inputs["input_ids"].append(input_ids)
        model_inputs["attention_mask"].append([1] * len(input_ids))
        model_inputs["labels"].append(labels)
    # inputs_ids 进行left padding ,用于生成

    return model_inputs


def preprocess_packed_supervised_dataset_train(examples, tokenizer, template, max_source_length, max_target_length):
    # 将数据打包
    # build inputs with format `<bos> X1 Y1 <eos> <bos> X2 Y2 <eos>`
    # and labels with format `<ignore> ... <ignore> Y1 <eos> <ignore> ... <ignore> Y2 <eos>`
    model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}
    input_ids, labels = [], []

    for query, response, history, system in construct_example(examples):
        if not (isinstance(query, str) and isinstance(response, str) and query != "" and response != ""):
            # 跳过为空的情况
            continue
        for turn_idx, (source_ids, target_ids) in enumerate(
            template.encode_multiturn(tokenizer, query, response, history, system)
        ):
            if turn_idx != 0 and template.efficient_eos:
                # 第一轮对话不需要加上 efficient_eos
                # prompt 不需要计算损失
                source_mask = [tokenizer.eos_token_id] + [IGNORE_INDEX] * (len(source_ids) - 1)
            else:
                source_mask = [IGNORE_INDEX] * len(source_ids)
            input_ids += source_ids + target_ids
            labels += source_mask + target_ids

    if template.efficient_eos:
        input_ids += [tokenizer.eos_token_id]
        labels += [tokenizer.eos_token_id]

    total_length = len(input_ids)
    # 不单独计算source 和 target 的长度，因为包含多个source 和target
    block_size = max_source_length + max_target_length

    if len(total_length) > block_size:
        total_length = (total_length // block_size) * block_size
        for i in range(0, total_length, block_size):
            model_inputs["input_ids"].append(input_ids[i : i + block_size])
            model_inputs["attention_mask"].append([1] * block_size)
            model_inputs["labels"].append(labels[i : i + block_size])
    else:
        # 长度不足的时候可以padding
        model_inputs["input_ids"].append(input_ids)
        model_inputs["attention_mask"].append([1] * len(input_ids))
        model_inputs["labels"].append(labels)

    return model_inputs


def preprocess_reward_function(examples, tokenizer, template, max_source_length, max_target_length):
    # build input pairs with format `<bos> X`, `Y1 <eos>` and `Y2 <eos>`
    model_inputs = {"prompt_ids": [], "chosen_ids": [], "rejected_ids": []}
    for query, response, history, system in construct_example(examples):
        if not (isinstance(query, str) and isinstance(response, list) and query != "" and len(response) > 1):
            continue
        prompt_ids, chosen_ids = template.encode_oneturn(tokenizer, query, response[0], history, system)
        _, rejected_ids = template.encode_oneturn(tokenizer, query, response[1], history, system)

        if template.efficient_eos:
            chosen_ids += [tokenizer.eos_token_id]
            rejected_ids += [tokenizer.eos_token_id]

        source_len, target_len = len(prompt_ids), max(len(chosen_ids), len(rejected_ids))

        if source_len > max_source_length:
            prompt_ids = prompt_ids[:max_source_length]
        if target_len > max_target_length:
            chosen_ids = chosen_ids[:max_target_length]
            rejected_ids = rejected_ids[:max_target_length]

        model_inputs["prompt_ids"].append(prompt_ids)
        model_inputs["chosen_ids"].append(chosen_ids)
        model_inputs["rejected_ids"].append(rejected_ids)

    return model_inputs


def preprocess_ppo_function(example, tokenizer, template, max_source_length, max_target_length):
    pass
