import json
import os

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

lora_path = "/home/zb/saved_checkpoint/baichuan_13b_overfit_100epoch/checkpoint-10000/"
output_dir = "/home/zb/NlpTaskSpace-llm/saved_output/step-10000"

model_name_or_path = "/data/SHARE/MODELS/BAICHUAN/Baichuan2-13B-Base/"
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=(torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32),
)

tokenizer = AutoTokenizer.from_pretrained(
    model_name_or_path,
    trust_remote_code=True,
    use_fast=True,
    add_bos_token=False,
    add_eos_token=False,
    padding_side="left",
)
peft_model = PeftModel.from_pretrained(model, lora_path)
model = peft_model.merge_and_unload()


# with open("/data/SHARE/EVAL/BBH/BBH.jsonl", "r", encoding="utf-8") as fr, open(
#     os.path.join(output_dir, "BBH_prediction.jsonl"), "w"
# ) as fw:
#     for example in tqdm(fr.readlines()):
#         example = example.strip()
#         example = json.loads(example)
#         query = example["input"]
#         query = query + "->"
#         input_ids = tokenizer.encode(query, return_tensors="pt").cuda()
#         output = model.generate(
#             input_ids,
#             max_new_tokens=128,
#             return_dict_in_generate=True,
#             repetition_penalty=1.1,
#         )
#         output_text = tokenizer.decode(
#             output.sequences[0], skip_special_tokens=True, clean_up_tokenization_spaces=False
#         )

#         response = output_text.split("->")[-1]

#         fw.write(json.dumps({"prediction": response}) + "\n")


with open("/data/SHARE/EVAL/CEVAL/CEVAL.jsonl", "r", encoding="utf-8") as fr, open(
    os.path.join(output_dir, "CEVAL_prediction.jsonl"), "w"
) as fw:
    for example in tqdm(fr.readlines()):
        example = example.strip()
        example = json.loads(example)

        query = example["input"]

        query = query + "->"
        input_ids = tokenizer.encode(query, return_tensors="pt").cuda()
        output = model.generate(
            input_ids,
            max_new_tokens=128,
            return_dict_in_generate=True,
            repetition_penalty=1.1,
        )
        output_text = tokenizer.decode(
            output.sequences[0], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        response = output_text.split("->")[-1]
        fw.write(json.dumps({"prediction": response}) + "\n")


with open("/data/SHARE/EVAL/MMLU/MMLU.jsonl", "r", encoding="utf-8") as fr, open(
    os.path.join(output_dir, "MMLU_prediction.jsonl"), "w"
) as fw:
    for example in tqdm(fr.readlines()):
        example = example.strip()
        example = json.loads(example)

        query = example["input"]

        query = query + "->"
        input_ids = tokenizer.encode(query, return_tensors="pt").cuda()
        output = model.generate(
            input_ids,
            max_new_tokens=128,
            return_dict_in_generate=True,
            repetition_penalty=1.1,
        )
        output_text = tokenizer.decode(
            output.sequences[0], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        response = output_text.split("->")[-1]

        fw.write(json.dumps({"prediction": response}) + "\n")


with open("/data/SHARE/EVAL/ECTE/ECTE.jsonl", "r", encoding="utf-8") as fr, open(
    os.path.join(output_dir, "ECTE_prediction.jsonl"), "w"
) as fw:
    for example in tqdm(fr.readlines()):
        example = example.strip()
        example = json.loads(example)

        query = example["input"]

        query = query + "->"
        input_ids = tokenizer.encode(query, return_tensors="pt").cuda()
        output = model.generate(
            input_ids,
            max_new_tokens=128,
            return_dict_in_generate=True,
            repetition_penalty=1.1,
        )
        output_text = tokenizer.decode(
            output.sequences[0], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        response = output_text.split("->")[-1]

        fw.write(json.dumps({"prediction": response}) + "\n")

with open("/data/SHARE/EVAL/ECQE/ECQE.jsonl", "r", encoding="utf-8") as fr, open(
    os.path.join(output_dir, "ECQE_prediction.jsonl"), "w"
) as fw:
    for example in tqdm(fr.readlines()):
        example = example.strip()
        example = json.loads(example)

        query = example["input"]

        query = query + "->"
        input_ids = tokenizer.encode(query, return_tensors="pt").cuda()
        output = model.generate(
            input_ids,
            max_new_tokens=128,
            return_dict_in_generate=True,
            repetition_penalty=1.1,
        )
        output_text = tokenizer.decode(
            output.sequences[0], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        response = output_text.split("->")[-1]

        fw.write(json.dumps({"prediction": response}) + "\n")

with open("/data/SHARE/EVAL/ECTC/ECTC.jsonl", "r", encoding="utf-8") as fr, open(
    os.path.join(output_dir, "ECTC_prediction.jsonl"), "w"
) as fw:
    for example in tqdm(fr.readlines()):
        example = example.strip()
        example = json.loads(example)

        query = example["input"]

        query = query + "->"
        input_ids = tokenizer.encode(query, return_tensors="pt").cuda()
        output = model.generate(
            input_ids,
            max_new_tokens=128,
            return_dict_in_generate=True,
            repetition_penalty=1.1,
        )
        output_text = tokenizer.decode(
            output.sequences[0], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        response = output_text.split("->")[-1]
        fw.write(json.dumps({"prediction": response}) + "\n")
