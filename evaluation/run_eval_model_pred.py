import json

import torch
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


model_path = "/data/SHARE/MODELS/BAICHUAN/Baichuan2-13B-Chat/"

tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True).cuda()

model.generation_config = GenerationConfig.from_pretrained(model_path)

with open("/data/SHARE/EVAL/BBH/BBH.jsonl", "r", encoding="utf-8") as fr, open(
    "./baichuan2-13b/BBH_prediction.jsonl", "w"
) as fw:
    for example in tqdm(fr.readlines()):
        example = example.strip()
        example = json.loads(example)

        query = example["input"]
        target = example["target"]
        message = []
        message.append({"role": "user", "content": f"{query}"})
        response = model.chat(tokenizer, message)
        response = response.strip()

        fw.write(json.dumps({"prediction": response}) + "\n")


with open("/data/SHARE/EVAL/CEVAL/CEVAL.jsonl", "r", encoding="utf-8") as fr, open(
    "./baichuan2-13b/CEVAL_prediction.jsonl", "w"
) as fw:
    for example in tqdm(fr.readlines()):
        example = example.strip()
        example = json.loads(example)

        query = example["input"]
        target = example["target"]

        message = []
        message.append({"role": "user", "content": f"{query}"})
        response = model.chat(tokenizer, message)
        response = response.strip()
        fw.write(json.dumps({"prediction": response}) + "\n")


with open("/data/SHARE/EVAL/MMLU/MMLU.jsonl", "r", encoding="utf-8") as fr, open(
    "./baichuan2-13b/MMLU_prediction.jsonl", "w"
) as fw:
    for example in tqdm(fr.readlines()):
        example = example.strip()
        example = json.loads(example)

        query = example["input"]
        target = example["target"]

        message = []
        message.append({"role": "user", "content": f"{query}"})
        response = model.chat(tokenizer, message)
        response = response.strip()

        fw.write(json.dumps({"prediction": response}) + "\n")


with open("/data/SHARE/EVAL/ECTE/ECTE.jsonl", "r", encoding="utf-8") as fr, open(
    "./baichuan2-13b/ECTE_prediction.jsonl", "w"
) as fw:
    for example in tqdm(fr.readlines()):
        example = example.strip()
        example = json.loads(example)

        query = example["input"]
        target = example["target"]

        message = []
        message.append({"role": "user", "content": f"{query}"})
        response = model.chat(tokenizer, message)
        response = response.strip()

        fw.write(json.dumps({"prediction": response}) + "\n")

with open("/data/SHARE/EVAL/ECQE/ECQE.jsonl", "r", encoding="utf-8") as fr, open(
    "./baichuan2-13b/ECQE_prediction.jsonl", "w"
) as fw:
    for example in tqdm(fr.readlines()):
        example = example.strip()
        example = json.loads(example)

        query = example["input"]
        target = example["target"]

        message = []
        message.append({"role": "user", "content": f"{query}"})
        response = model.chat(tokenizer, message)
        response = response.strip()

        fw.write(json.dumps({"prediction": response}) + "\n")
