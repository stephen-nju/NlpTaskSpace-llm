from typing import Any, Dict, List

from llm.src.constant import IGNORE_INDEX


def construct_example(examples: Dict[str, List[Any]]):
    for i in range(len(examples["instruction"])):
        instruction, inputs, outputs = examples["instruction"][i], examples["input"][i], examples["output"][i]
        query, response = instruction + inputs + "->", str(outputs)
        query = query + "\n" + examples["query"][i] if "query" in examples and examples["query"][i] else query
        history = examples["history"][i] if "history" in examples else None
        system = examples["system"][i] if "system" in examples else None
        yield query, response, history, system


def preprocess_supervised_dataset(
    examples, tokenizer, template, max_source_length, max_target_length
) -> Dict[str, Any]:
    # build inputs with format `<bos> X Y <eos>` and labels with format `<ignore> ... <ignore> Y <eos>`
    # for multiturn examples, we only mask the prompt part in each prompt-response pair.
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
