# ---coding:utf-8  -----------
from queue import Queue
from threading import Thread
from typing import List, Optional, Tuple, Union

import torch
from transformers import GenerationConfig


def build_chat_input(model, tokenizer, messages: List[dict], max_new_tokens: int = 0):
    # 构建round =[[round 1],[round2]]
    def _parse_messages(messages, split_role="user"):
        system = (
            "A chat between a curious user and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the user's questions."
        )
        rounds = []
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
    max_input_tokens = model.config.max_position_embeddings - max_new_tokens
    system, rounds = _parse_messages(messages, split_role="user")
    system_tokens = tokenizer.encode(system)
    max_history_tokens = max_input_tokens - len(system_tokens)
    history_tokens = []
    for round in rounds[::-1]:
        round_tokens = []
        for message in round:
            if message["role"] == "user":
                s = "Human: {}\nAssistant:".format(message["content"])
                round_tokens.extend(tokenizer.encode(s))
            else:
                round_tokens.extend(tokenizer.encode(message["content"]))

        if len(history_tokens) == 0 or len(history_tokens) + len(round_tokens) <= max_history_tokens:
            history_tokens = round_tokens + history_tokens  # concat left
            if len(history_tokens) < max_history_tokens:
                continue

        break
    input_tokens = system_tokens + history_tokens
    # if messages[-1]["role"] != "assistant":
    # 不需要该逻辑
    #     input_tokens.append(model.generation_config.assistant_token_id)
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
