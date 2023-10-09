# -----*----coding:utf8-----*----

import argparse
import json
import os
import pickle
from functools import partial
from os import cpu_count

import bitsandbytes as bnb
import torch
import torch.nn as nn
from lightning import (
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from multiprocess.dummy import Pool
from peft import AdaLoraConfig, LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_linear_schedule_with_warmup,
)


class BaichuanInputExample:
    def __init__(self, text, labels) -> None:
        self.text = text
        self.labels = labels


class BaichuanInutFeature:
    def __init__(self, input_ids, labels) -> None:
        self.input_ids = input_ids
        self.labels = labels


def convert_example_to_feature(example, tokenizer, max_source_length, max_target_length, pad_token_id, eos_token_id):
    prefix = """命名实体识别：抽取文本中的 品牌，品类，系列型号 这三类命名实体，并按照json格式返回结果。\n\n"""

    max_seq_length = max_source_length + max_target_length + 1
    query, answer = example.text, example.labels
    answer = str(answer)
    prompt = prefix + query + " ->"

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

    return BaichuanInutFeature(input_ids=input_ids, labels=labels)


def convert_examples_to_features(
    examples, tokenizer, max_source_length, max_target_length, pad_token_id, eos_token_id, threads=4
):
    threads = min(threads, cpu_count())
    with Pool(threads) as p:
        annotate_ = partial(
            convert_example_to_feature,
            tokenizer=tokenizer,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            max_source_length=max_source_length,
            max_target_length=max_target_length,
        )
        features = list(
            tqdm(
                p.imap(annotate_, examples, chunksize=32),
                total=len(examples),
                desc="convert examples to features",
            )
        )
    return features


class BaichuanDataset(Dataset):
    def __init__(self, features) -> None:
        self.features = features
        super().__init__()

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        feature = self.features[index]

        return {
            "input_ids": torch.tensor(feature.input_ids, dtype=torch.long),
            "labels": torch.tensor(feature.labels, dtype=torch.long),
        }


class BaichuanDataModule(LightningDataModule):
    def __init__(self, args) -> None:
        self.args = args
        config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            use_fast=not args.use_slow_tokenizer,
            trust_remote_code=True,
        )
        self.pad_token_id = 0
        self.eos_token_id = 2
        self.cache_path = os.path.join(os.path.dirname(args.train_data), "cache")
        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)
        super().__init__()

    def prepare_data(self):
        train_examples = list(self.read_train_data(self.args.train_data))
        train_features = convert_examples_to_features(
            examples=train_examples,
            tokenizer=self.tokenizer,
            max_source_length=self.args.max_source_length,
            max_target_length=self.args.max_target_length,
            pad_token_id=self.pad_token_id,
            eos_token_id=self.eos_token_id,
        )

        with open(os.path.join(self.cache_path, "train_feature.pkl"), "wb") as g:
            pickle.dump(train_features, g)

        # dev_examples = list(self.read_train_data(self.args.dev_data))
        # dev_features = convert_examples_to_features(examples=dev_examples,
        #                                             tokenizer=self.tokenizer,
        #                                             max_length=self.args.max_length
        # pad_token_id=self.pad_token_id,
        # eos_token_id=self.eos_token_id,
        #                                             )

        # with open(os.path.join(self.cache_path, "dev_feature.pkl"), "wb") as g:
        #     pickle.dump(dev_features, g)

    @staticmethod
    def read_train_data(path):
        with open(path, "r", encoding="utf-8") as g:
            for line in g:
                data = json.loads(line)
                yield BaichuanInputExample(text=data["context"], labels=data["ner"])

    def setup(self, stage: str) -> None:
        with open(os.path.join(self.cache_path, "train_feature.pkl"), "rb") as g:
            self.train_features = pickle.load(g)

        # with open(os.path.join(self.cache_path, "dev_feature.pkl"), "rb") as g:
        #     self.dev_features = pickle.load(g)

    def train_dataloader(self):
        return DataLoader(
            dataset=BaichuanDataset(self.train_features),
            batch_size=self.args.batch_size,
            num_workers=4,
            pin_memory=True,
        )


# def val_dataloader(self):
#      return DataLoader(dataset=Baichuan7bNerDataset(self.dev_features),
#                        batch_size=self.args.batch_size,
#                        num_workers=4,
#                        pin_memory=True
#                        )


class BaichuanModule(LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        # 加载模型
        config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=True)

        self.model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            low_cpu_mem_usage=args.low_cpu_mem_usage,
            trust_remote_code=True,
            # device_map="auto",
        )
        # 模型初始化
        embedding_size = self.model.get_input_embeddings().weight.shape[0]
        # if len(tokenizer) > embedding_size:
        #     self.model.resize_token_embeddings(len(tokenizer))

        # model setting and lora config
        # model.supports_gradient_checkpointing = True  #
        self.model.gradient_checkpointing_enable()
        self.model.enable_input_require_grads()

        self.model.config.use_cache = False  # silence the warnings. Please re-enable for inference!the datasets
        self.model = self.model.half()

        # 配置LoraConfig
        if args.quantization_bit is not None:
            print(f"Quantized to {args.quantization_bit} bit")
            model = model.quantize(args.quantization_bit)
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=["W_pack"],
            inference_mode=False,
            r=16,
            lora_alpha=16,
            lora_dropout=0.05,
        )
        # adalora 会出现 https://github.com/huggingface/peft/issues/479
        self.model = get_peft_model(self.model, peft_config)
        self.model.is_parallelizable = True
        self.model.model_parallel = True
        self.model.print_trainable_parameters()
        self.save_hyperparameters()

    def configure_optimizers(self):
        """Prepare optimizer and learning rate scheduler"""

        no_decay = ["bias", "layer_norm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        if self.args.optimizer == "adam":
            optimizer = torch.optim.Adam(
                optimizer_grouped_parameters,
                lr=self.args.learning_rate,
            )
        else:
            # revisiting few-sample BERT Fine-tuning https://arxiv.org/pdf/2006.05987.pdf
            # https://github.com/asappresearch/revisit-bert-finetuning/blob/master/run_glue.py
            optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters,
                lr=self.args.learning_rate,
                eps=self.args.adam_epsilon,
                weight_decay=self.args.weight_decay,
            )

        num_gpus = self.trainer.num_devices
        # 注：只有在使用pytorch Lightning的LightningDataModule 时候才可以使用该方式回去训练集大小
        t_total = (
            len(self.trainer.datamodule.train_dataloader()) // (self.trainer.accumulate_grad_batches * num_gpus) + 1
        ) * self.args.max_epochs
        warmup_steps = int(self.args.warmup_proportion * t_total)

        if self.args.lr_scheduler == "onecycle":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.args.learning_rate,
                pct_start=float(warmup_steps / t_total),
                final_div_factor=self.args.final_div_factor,
                total_steps=t_total,
                anneal_strategy="linear",
            )

        elif self.args.lr_scheduler == "linear":
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=t_total,
            )
        elif self.args.lr_scheduler == "polydecay":
            scheduler = get_polynomial_decay_schedule_with_warmup(
                optimizer,
                warmup_steps,
                t_total,
                lr_end=self.args.learning_rate / 4.0,
            )
        elif self.args.lr_scheduler == "cawr":
            # TODO
            step = (len(self.trainer.datamodule.train_dataloader())) // (
                self.trainer.accumulate_grad_batches * num_gpus + 1
            )

            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, step * self.args.rewarm_epoch_num, 1
            )
        else:
            raise ValueError("lr_scheduler does not exist.")
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def training_step(self, batch, batch_idx):
        output = self.model(**batch)
        return output.loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train tplinker ner model")
    parser.add_argument("--output_dir", type=str, default="./output_dir/", help="")
    parser.add_argument("--train_data", type=str, default="", help="train data path")
    parser.add_argument("--test_data", type=str, default="", help="test data path")
    parser.add_argument("--dev_data", type=str, default="", help="dev data path")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")

    parser.add_argument(
        "--lr_scheduler",
        choices=["linear", "onecycle", "polydecay", "cawr"],
        default="cawr",
    )
    parser.add_argument(
        "--rewarm_epoch_num",
        type=int,
        help="cawr learning scheduler rewarm epoch num",
        default=2,
    )
    parser.add_argument("--workers", type=int, default=0, help="num workers for dataloader")
    parser.add_argument(
        "--warmup_proportion",
        default=0.1,
        type=int,
        help="warmup steps used for scheduler.",
    )
    parser.add_argument(
        "--adam_epsilon",
        default=1e-8,
        type=float,
        help="Epsilon for Adam optimizer.",
    )
    parser.add_argument(
        "--final_div_factor",
        type=float,
        default=1e4,
        help="final div factor of linear decay scheduler",
    )

    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument("--optimizer", type=str, help=("model optimizer"))
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the �� Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay to use.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="A seed for reproducible training.",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=None,
        help="The max training max epochs.",
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=None,
        help=(
            "Optional input sequence length after tokenization. The training dataset will be truncated in block of"
            " this size for training. Default to the model max input length for single sentence inputs (take into"
            " account special tokens)."
        ),
    )

    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache",
        action="store_true",
        help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument(
        "--no_keep_linebreaks",
        action="store_true",
        help="Do not keep line breaks when using TXT files.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )

    parser.add_argument(
        "--low_cpu_mem_usage",
        action="store_true",
        help=(
            "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded."
            "If passed, LLM loading time and RAM consumption will be benefited."
        ),
    )
    parser.add_argument("--max_source_length", type=int, default=128, help="")
    parser.add_argument(
        "--max_target_length",
        type=int,
        default=32,
        help=(
            "The maximum total sequence length for target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        ),
    )

    parser.add_argument(
        "--ignore_pad_token_for_loss",
        type=bool,
        default=True,
        help="Whether to ignore the tokens corresponding to padded labels in the loss computation or not.",
    )

    parser.add_argument(
        "--quantization_bit",
        type=str,
        default=None,
        help="quantization training",
    )

    arg = parser.parse_args()

    if arg.seed is not None:
        seed_everything(arg.seed)

    datamodule = BaichuanDataModule(arg)
    model = BaichuanModule(arg)

    trainer = Trainer(accelerator="gpu", strategy="ddp", devices=1, max_epoch lightning.strategiestrainer.fit(model, datamodule=datamodule)
