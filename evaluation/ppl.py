import json
import time
from typing import Dict, List

import torch
import torch.nn as nn
from rich import print
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.trainer_pt_utils import LabelSmoother


def qwen_chat_preprocess(record: Dict, tokenizer, max_seq_length=2048) -> Dict[str, torch.Tensor]:
    """
    千问Chat模型数据预处理,数据拼接格式如下：

    <|im_start|>system
    You are a helpful assistant.<|im_end|>
    <|im_start|>user
    你好呀<|im_end|>
    <|im_start|>assistant
    你好，我是xxx，很高兴为您服务<|im_end|>

    输入示例：
    {
        "system": "这一项可以没有",
        "conversation": [{"human": "中国的首都是哪里？","assistant": "北京"}]
    }

    返回字典：

    input_ids--token的id

    attention_mask--attention mask

    token_type_ids--token_type_ids

    target_mask--需要计算loss的token为1，其余为0
    """
    # 处理system提示词，没有则i设置默认提示词
    if "system" in record.keys():
        system: str = record["system"].strip()
    else:
        system: str = "You are a helpful assistant."

    system_text: str = f"<|im_start|>system\n{system}<|im_end|>\n"
    input_ids: List[int] = tokenizer.encode(system_text, add_special_tokens=False)
    target_mask: List[int] = [0] * len(input_ids)

    # 处理对话内容
    conversations: List[Dict[str, str]] = record["conversation"]

    # 拼接多轮对话
    for conversation in conversations:
        human = conversation["human"].strip()
        assistant = conversation["assistant"].strip()

        input_tokens = tokenizer.encode(f"<|im_start|>user\n{human}<|im_end|>\n", add_special_tokens=False)
        output_tokens = tokenizer.encode(f"<|im_start|>assistant\n{assistant}<|im_end|>\n", add_special_tokens=False)

        input_ids += input_tokens + output_tokens

        # input_tokens部分不计算loss
        target_mask += [0] * len(input_tokens)

        # '<|im_start|>assistant\n'占3个token，结尾的'\n'占1个token，不计算它们的loss
        target_mask += [0] * 3 + [1] * (len(output_tokens) - 4) + [0]

    assert len(input_ids) == len(target_mask)

    # 对长度进行截断
    input_ids = input_ids[:max_seq_length]
    target_mask = target_mask[:max_seq_length]

    attention_mask: List[int] = [1] * len(input_ids)
    token_type_ids: List[int] = [0] * len(input_ids)

    assert len(input_ids) == len(target_mask) == len(attention_mask) == len(token_type_ids)

    # 转成tensor
    input_tensor = torch.tensor(input_ids)
    attention_tensor = torch.tensor(attention_mask)
    target_tensor = torch.tensor(target_mask)
    type_tensor = torch.tensor(token_type_ids)

    return {"input_ids": input_tensor, "attention_mask": attention_tensor, "target_mask": target_tensor, "token_type_ids": type_tensor}


def is_correct_record(record: Dict) -> bool:
    """
    检查记录是否合法

    记录示例：
    {
        "system": "这一项可以没有",
        "conversation": [{"human": "中国的首都是哪里？","assistant": "北京"}]
    }

    system不是必填项，可以没有

    conversation必须有，并且必须包含human和assistant子项
    """
    if "conversation" in record:
        conversations: List[Dict[str, str]] = record["conversation"]
        if isinstance(conversations, list):
            for conversation in conversations:
                if not (isinstance(conversation, dict) and "human" in conversation and "assistant" in conversation):
                    return False
        else:
            return False
    else:
        return False
    return True


class QwenSFTDataset(Dataset):
    def __init__(self, lines: List[str], tokenizer, max_seq_length=2048):
        self.tokenizer = tokenizer
        # 设置特殊token
        self.im_start_id = tokenizer.im_start_id
        self.im_end_id = tokenizer.im_end_id
        self.enter_token_ids = tokenizer.encode("\n")  # 回车键
        # 最大序列长度
        self.max_seq_length = max_seq_length
        # 加载jsonl格式数据文件
        self.data = self._load_jsonl_data(lines)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # 获取原始数据
        return self.data[index]

    def _load_jsonl_data(self, lines: List[str]) -> List[Dict]:
        data: List[Dict] = []

        # 对每一项数据做预处理
        for line in lines:
            record = json.loads(line)

            if not is_correct_record(record):
                continue

            data.append(qwen_chat_preprocess(record, self.tokenizer, self.max_seq_length))

        return data


class TargetLMLoss:
    """
    多轮对话损失函数，只对对话内容中的部分logits计算损失
    """

    def __init__(self, ignore_index):
        super().__init__()
        # 设置ignore index
        self.ignore_index = ignore_index
        # 设置交叉熵损失函数，并对结果取均值
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction="mean")

    def __call__(self, model, inputs, return_outputs=False) -> torch.Tensor:
        # 获取输入的token ids，size = batch size * sequence
        input_ids = inputs["input_ids"]
        # 获取attention mask
        attention_mask = inputs["attention_mask"]
        # 重点！只有target mask上为1才参与后续计算
        target_mask = inputs["target_mask"]
        # 模型前馈预测
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        # logits的size=batch*length*word
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs[0]

        # 将labels中不属于target的部分，设为ignore_index，只计算target部分的loss
        labels = torch.where(target_mask == 1, input_ids, self.ignore_index)
        # 重点！移除序列中【最后一个输出】，将logtis shape从batch size*sequence length*vocabulary size变为batch size*(seqnence length-1)*vocabulary size
        shift_logits = logits[..., :-1, :].contiguous()
        # 重点！移除序列中【第一个输入】，将labels shape从batch size*sequence length变为batch size*（sequence length-1）
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss = self.loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        # 这里修改了原始输出，只返回loss
        # return (loss, outputs) if return_outputs else loss
        return loss


def calculate_target_perplexity(model, tokenizer, texts: List[str], ignore_index, batch_size: int = 32, max_seq_length: int = 2048):
    """
    基于预训练的decoder模型，计算模型在输入语料的指定token上的ppl值

    """
    # 创建损失函数
    loss_fn = TargetLMLoss(ignore_index=ignore_index)

    model.eval()

    total_loss = 0
    total_length = 0

    dataset = QwenSFTDataset(texts, tokenizer, max_seq_length=max_seq_length)

    # TODO 目前实现的是batch size固定为1的版本，未实现批量运行功能
    for index in range(len(dataset)):
        inputs = dataset[index]
        with torch.no_grad():
            # 获取每个token的平均损失
            loss = loss_fn(model, inputs)

            # 计算target的token数
            target_length = inputs["target_mask"].sum().item()

            total_loss += loss.item() * target_length
            total_length += target_length

    # 计算模型在整个输入文本中的平均损失
    average_loss = total_loss / total_length
    print(f"average_loss = {average_loss}")
    # 计算PPL值，返回类型为float
    ppl = torch.exp(input=torch.tensor(data=average_loss)).item()

    return ppl


def calculate_perplexity(model, tokenizer, texts: List[str], batch_size: int = 32, max_seq_length=2048):
    """
    基于预训练的decoder模型，计算模型在对应语料上的ppl值
    return：PPL值，类型为torch Tensor
    """
    model.eval()  # 设置模型为评估模式
    total_loss = 0
    total_length = 0

    for i in range(0, len(texts), batch_size):
        # 从原始输入中截取出一个batch样本
        batch_texts: list[str] = texts[i : i + batch_size]

        # tokenizer输入样本attention_mask=input_ids.ne(tokenizer.pad_token_id)
        # 返回类型为dict，包含三个key:input_ids，token_type_ids，attention_mask
        # 三个key对应value都是tensor，shape = batch size*sequence length
        # TODO 没有实现max nlength
        encoded_inputs: Dict[str, torch.Tensor] = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt").to(model.device)

        with torch.no_grad():  # 在评估时不计算梯度
            # 输出类型为CausalLMOutputWithPast，包含loss、logits、past_key_values、hidden_states和attentions
            outputs: CausalLMOutputWithPast = model(**encoded_inputs, labels=encoded_inputs["input_ids"])

            # logits的size=batch size*sequence length*vocabulary size
            # print(outputs["logits"].size())
            # # past_key_values是一个嵌套tuple结构，tuple结构为：24*2
            # out_tuple_len = len(outputs["past_key_values"])
            # inner_tuple_len = len(outputs["past_key_values"][0])
            # print(f"past_key_value的tuple结构为：{out_tuple_len}*{inner_tuple_len}")
            # # pask_key_value的内层tensor结构为：torch.Size([batch size, sentence length, 16, 128])
            # pkv_tensor_size = outputs["past_key_values"][0][0].size()
            # print(f"pask_key_value的内层tensor结构为：{pkv_tensor_size}")

            # 计算损失，loss为一个数，如果损失值不是tensor，则设为0
            loss = outputs.loss.item() if isinstance(outputs.loss, torch.Tensor) else 0.0
            # 注意！这里计算a得到的loss值是针对这个batch所有step的平均值
            total_loss += loss * encoded_inputs["input_ids"].size(1)
            # 累加序列长度
            total_length += encoded_inputs["input_ids"].size(1)

    # 计算模型在整个输入文本中的平均损失
    average_loss = total_loss / total_length
    # 计算PPL值，返回类型为float
    ppl = torch.exp(input=torch.tensor(data=average_loss)).item()
    return ppl


def use_calculate_perplexity(model, tokenizer) -> None:
    """
    演示如何计算整个文本ppl值
    """
    texts_wrong = ["中国的首都是背景", "我爱南京天安门", "华为手机使用的是iso操作系统"]
    texts_right = ["中国的首都是北京", "我爱北京天安门", "华为手机使用的是鸿蒙操作系统"]

    s_time = time.time()

    # 针对qwen模型，返回的是bf16类型tensor
    ppl_wrong = calculate_perplexity(model, tokenizer, texts_wrong, batch_size=3)
    ppl_right = calculate_perplexity(model, tokenizer, texts_right, batch_size=3)

    e_time = time.time()

    print(f"text wrong PPL = {ppl_wrong}")
    print(f"text right PPL = {ppl_right}")
    print(f"计算ppl耗时 = {e_time-s_time}")


def use_calculate_target_perplexity(model, tokenizer) -> None:
    """
    演示如何计算目标回复的ppl值
    """
    # 加载测试数据
    jsonl_lines = open("eval.jsonl", "r", encoding="utf-8").readlines()
    # 为损失函数设置ignore index
    # ignore index传给交叉熵函数和语言模型没有关系，只要这个index对应的token id不是一个“有用”的id即可
    ignore_index = LabelSmoother.ignore_index  # -100

    s_time = time.time()

    target_ppl = calculate_target_perplexity(model, tokenizer, jsonl_lines, ignore_index)

    e_time = time.time()
    print(f"Target PPL = {target_ppl}")
    print(f"运行耗时：{e_time-s_time}")


def use_loss_fun(model, tokenizer):
    """
    演示如何使用自定义loss计算loss值
    """
    # 构建损失函数
    target_loss = TargetLMLoss(ignore_index=LabelSmoother.ignore_index)
    # 测试文本
    texts_right = ["中国的首都是北京", "我爱北京天安门", "华为手机使用的是鸿蒙操作系统"]
    # 构建模型输入
    inputs = tokenizer(texts_right, return_dict=True, return_tensors="pt", padding=True, truncation=True)
    inputs["target_mask"] = inputs["attention_mask"]
    # print("损失函数输入示例")
    # print(inputs)
    loss = target_loss(model=model, inputs=inputs)
    print(f"验证样本的平均loss={loss}")


def main():
    # 可选的模型包括:  "Qwen-14B-Chat", "Qwen-14B", "Qwen-1_8B"
    model_path = "/data/SHARE/MODELS/Qwen/Qwen-14B-Chat"
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        pad_token="<|extra_0|>",
        eos_token="<|endoftext|>",
        padding_side="left",
        trust_remote_code=True,
    )
    # 打开bf16精度，A100、H100、RTX3060、RTX3070等显卡建议启用以节省显存
    # model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True, bf16=True,use_flash_attn=True,pad_token_id=tokenizer.pad_token_id).eval()
    # 使用CPU进行推理，需要约32GB内存
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cpu", pad_token_id=tokenizer.pad_token_id, trust_remote_code=True).eval()

    # 计算文本PPL值
    use_calculate_perplexity(model=model, tokenizer=tokenizer)
    # 计算目标文本PPL值
    use_calculate_target_perplexity(model=model, tokenizer=tokenizer)


if __name__ == "__main__":
    main()
