import json
from copy import deepcopy
from typing import Any, Dict, List

from llm.src.constant import IGNORE_INDEX


def construct_sft_example(examples: Dict[str, List[Any]]):
    for i in range(len(examples["instruction"])):
        instruction, inputs, outputs = examples["instruction"][i], examples["input"][i], examples["output"][i]
        query, response = instruction + inputs + "->", str(outputs)
        query = query + "\n" + examples["query"][i] if "query" in examples and examples["query"][i] else query
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
    for query, response, history, system in construct_sft_example(examples):
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
    ignore_pad_token_for_loss=True,
):
    pad_token_id = tokenizer.pad_token_id
    tokenizer.padding_side = "left"
    sources, targets = [], []
    for i in range(len(examples["input"])):
        if examples["input"][i] and examples["output"][i] and examples["instruction"][i]:
            inputs = examples["input"][i]
            instruction = examples["instruction"][i]
            inputs = instruction + inputs
            sources.append(inputs)
            target = examples["output"][i]  # 需要将字典类型转化为字符串类型
            targets.append(target)

    model_inputs = tokenizer(
        sources,
        max_length=max_source_length,
        truncation=True,
        padding=True,
    )

    labels = tokenizer(targets, max_length=max_target_length, truncation=True, padding=True)
    if ignore_pad_token_for_loss:
        labels["input_ids"] = [[(l if l != pad_token_id else -100) for l in label] for label in labels["input_ids"]]

    model_inputs["labels"] = labels["input_ids"]

    return model_inputs
