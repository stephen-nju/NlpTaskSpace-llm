# ----*----coding:utf8-----*---

import argparse
import functools
import glob
import os
from itertools import chain
from types import MethodType
from typing import Any, Dict, Tuple, Union

import datasets
import torch
from datasets import load_dataset
from deepspeed.ops.adam import DeepSpeedCPUAdam
from lightning import LightningModule, Trainer, seed_everything
from lightning.pytorch.strategies import DeepSpeedStrategy, FSDPStrategy
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)

from llm.src.callbacks import HFModelCheckpoint
from llm.src.constant import IGNORE_INDEX
from llm.src.utils import find_all_linear_names
from metrics.language_model import LanguageModelMetric


class SupervisedFintuningModule(LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        # 加载模型
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            use_fast=not args.use_slow_tokenizer,
            trust_remote_code=True,
        )
        self.llm_metrics = LanguageModelMetric()
        self.save_hyperparameters()

        # def configure_model(self):
        self.config = AutoConfig.from_pretrained(self.args.model_name_or_path, trust_remote_code=True)
        if self.args.quantization_bit is not None:
            print(f"Quantized to {self.args.quantization_bit}")
            if self.is_deepspeed_zero3_enabled:
                raise ValueError("DeepSpeed ZeRO-3 is incompatible with quantization.")
            if self.args.quantization_bit == "4bit":
                # load_in_4bit = (True,)
                quantization_config = BitsAndBytesConfig(
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
            elif self.args.quantization_bit == "8bit":
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            else:
                raise ValueError("unsupport quantization_bit")

            model = AutoModelForCausalLM.from_pretrained(
                self.args.model_name_or_path,
                from_tf=bool(".ckpt" in self.args.model_name_or_path),
                config=self.config,
                quantization_config=quantization_config,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )

        else:
            # 使用bf16 来加载模型
            print(f"model config===\n{self.config}")
            model = AutoModelForCausalLM.from_pretrained(
                self.args.model_name_or_path,
                from_tf=bool(".ckpt" in self.args.model_name_or_path),
                config=self.config,
                trust_remote_code=True,
                torch_dtype="auto",
            )
        # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
        # on a small vocab and want a smaller embedding size, remove this test.

        # embedding_size = model.get_input_embeddings().weight.shape[0]
        # if len(tokenizer) > embedding_size:
        #     model.resize_token_embeddings(len(tokenizer))

        # model.supports_gradient_checkpointing = True  #
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

        model.config.use_cache = False  # silence the warnings. Please re-enable for inference!the datasets

        # 分布式训练
        model.is_parallelizable = True
        model.model_parallel = True

        if self.args.quantization_bit is not None:
            # 启用模型量化需要开启
            model = prepare_model_for_kbit_training(model)

        # Set NEFTune trick for fine-tuning
        if self.args.neft_alpha > 0:
            input_embed = model.get_input_embeddings()
            if isinstance(input_embed, torch.nn.Embedding):

                def noisy_forward(self: torch.nn.Embedding, x: torch.Tensor) -> torch.Tensor:
                    embeddings = input_embed.__class__.forward(self, x)
                    dims = self.num_embeddings * self.embedding_dim
                    mag_norm = self.args.neft_alpha / (dims**0.5)
                    embeddings += torch.zeros_like(embeddings).uniform_(-mag_norm, mag_norm)
                    return embeddings

                # 将方法绑定到forward上,参考medicalGPT
                input_embed.forward = MethodType(noisy_forward, input_embed)
                self.log("Using noisy embedding with alpha={:.2f}".format(self.args.neft_alpha))
            else:
                self.log("Input embeddings are not normal nn.Embedding, cannot transform into noisy embedding.")

        if self.args.use_lora:
            if isinstance(self.args.lora_target, str):
                if self.args.lora_target == "all":
                    lora_target = find_all_linear_names(model, self.args.quantization_bit)
                else:
                    # support custom target modules/layers of LoRA
                    lora_target = [target.strip() for target in self.args.lora_target.split(",")]
            ### TODO 添加全量lora 的微调
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                target_modules=lora_target,
                inference_mode=False,
                r=self.args.lora_rank,
                lora_alpha=self.args.lora_alpha,
                lora_dropout=self.args.lora_dropout,
            )
            # adalora 会出现 https://github.com/huggingface/peft/issues/479

            model = get_peft_model(model, peft_config)

            # 分布式训练
            model.is_parallelizable = True
            model.model_parallel = True
            model.print_trainable_parameters()

        self.model = model
        self.generation_config = model.generation_config
        self.model.print_trainable_parameters()

    def group_texts(self, examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
        # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
        total_length = (total_length // self.args.block_size) * self.args.block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + self.args.block_size] for i in range(0, total_length, self.args.block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    def print_example(self, example):
        print("input_ids:\n{}".format(example["input_ids"]))
        print("inputs:\n{}".format(self.tokenizer.decode(example["input_ids"], skip_special_tokens=False)))
        print("label_ids:\n{}".format(example["labels"]))
        print(
            "labels:\n{}".format(
                self.tokenizer.decode(
                    [d if d != IGNORE_INDEX else self.tokenizer.pad_token_id for d in example["labels"]],
                    skip_special_tokens=False,
                )
            )
        )

    def tokenize_function(self, examples):
        return self.tokenizer(examples["text"])

    def setup(self, stage):
        if self.args.block_size is None:
            block_size = self.tokenizer.model_max_length
            if block_size > 2048:
                self.logger.warning(
                    "The chosen tokenizer supports a `model_max_length` that is longer than the default `block_size` value"
                    " of 2048. If you would like to use a longer `block_size` up to `tokenizer.model_max_length` you can"
                    " override this default with `--block_size xxx`."
                )

        else:
            if self.args.block_size > self.tokenizer.model_max_length:
                self.logger.warning(
                    f"The block_size passed ({self.args.block_size}) is larger than the maximum length for the model"
                    f"({self.tokenizer.model_max_length}). Using block_size={self.tokenizer.model_max_length}."
                )

            block_size = min(self.args.block_size, self.tokenizer.model_max_length)

        # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
        # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
        # (the dataset will be downloaded automatically from the datasets Hub).
        #
        # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
        # 'text' is found. You can easily tweak this behavior (see below).
        #
        # In distributed training, the load_dataset function guarantee that only one local process can concurrently
        # download the dataset.
        if self.args.dataset_name is not None:
            raw_datasets = load_dataset(
                self.args.dataset_name,
                self.args.dataset_config_name,
                streaming=self.args.streaming,
            )
            if "validation" not in raw_datasets.keys():
                raw_datasets["validation"] = load_dataset(
                    self.args.dataset_name,
                    self.args.dataset_config_name,
                    split=f"train[:{self.args.validation_split_percentage}%]",
                    streaming=self.args.streaming,
                )

                raw_datasets["train"] = load_dataset(
                    self.args.dataset_name,
                    self.args.dataset_config_name,
                    split=f"train[{self.args.validation_split_percentage}%:]",
                    streaming=self.args.streaming,
                )

        else:
            data_files = {}
            dataset_args = {}
            if self.args.train_data is not None:
                train_dof = [s.strip() for s in self.args.train_data.split(",")]
            else:
                raise ValueError("train data should not be none")
            # direcotory_or_file
            train_files = []
            for dof in train_dof:
                if os.path.exists(dof) and os.path.isfile(dof):
                    train_files.append(dof)
                elif os.path.exists(dof) and os.path.isdir(dof):
                    files = (
                        glob(f"{dof}/**/*.txt", recursive=True)
                        + glob(f"{dof}/**/*.json", recursive=True)
                        + glob(f"{dof}/**/*.jsonl", recursive=True)
                    )
                    types = [f.split(".")[-1] for f in files]
                    if len(set(types)) > 1:
                        raise ValueError(f"train files must be same type, e.g. all txt or all jsonl, but got {types}")
                    train_files.extend(files)
                else:
                    raise ValueError("train data should be valid file path or direcotory path split by ','")
            data_files["train"] = train_files

            if self.args.dev_data is not None:
                dev_dof = [s.strip() for s in self.args.dev_data.split(",")]
                dev_files = []
                for dof in dev_dof:
                    if os.path.exists(dof) and os.path.isfile(dof):
                        dev_files.append(dof)
                    elif os.path.exists(dof) and os.path.isdir(dof):
                        files = (
                            glob(f"{dof}/**/*.txt", recursive=True)
                            + glob(f"{dof}/**/*.json", recursive=True)
                            + glob(f"{dof}/**/*.jsonl", recursive=True)
                        )

                        types = [f.split(".")[-1] for f in files]
                        if len(set(types)) > 1:
                            raise ValueError(
                                f"train files must be same type, e.g. all txt or all jsonl, but got {types}"
                            )
                        dev_files.extend(files)
                    else:
                        raise ValueError("train data should be valid file path or direcotory path split by ','")
                self.logger.info(f"eval files: {dev_files}")
                data_files["validation"] = dev_files

            self.print(f"loading data files={data_files}")

            extension = "text" if data_files["train"][0].endswith("txt") else "json"
            if extension == "text":
                dataset_args["keep_linebreaks"] = self.args.keep_linebreaks

            raw_datasets = load_dataset(
                extension,
                data_files=data_files,
                **dataset_args,
            )
            # If no validation data is there, validation_split_percentage will be used to divide the dataset.
            if "validation" not in raw_datasets.keys():
                raw_datasets["validation"] = load_dataset(
                    extension,
                    data_files=data_files,
                    split=f"train[:{self.args.validation_split_percentage}%]",
                    **dataset_args,
                )

                raw_datasets["train"] = load_dataset(
                    extension,
                    data_files=data_files,
                    split=f"train[{self.args.validation_split_percentage}%:]",
                    **dataset_args,
                )
        # Preprocessing the datasets.
        column_names = list(raw_datasets["train"].features)
        if not self.args.streaming:
            tokenized_datasets = raw_datasets.map(
                self.tokenize_function,
                batched=True,
                num_proc=self.args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not self.args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )
            lm_datasets = tokenized_datasets.map(
                self.group_texts,
                batched=True,
                num_proc=self.args.preprocessing_num_workers,
                load_from_cache_file=not self.args.overwrite_cache,
                desc=f"Grouping texts in chunks of {block_size}",
            )
        else:
            tokenized_datasets = raw_datasets.map(
                self.tokenize_function,
                batched=True,
                remove_columns=column_names,
            )

            lm_datasets = tokenized_datasets.map(
                self.group_texts,
                batched=True,
            )

        train_dataset = None
        max_train_samples = 0
        train_dataset = lm_datasets["train"]
        max_train_samples = len(train_dataset)
        if self.args.max_train_samples is not None and self.args.max_train_samples > 0:
            max_train_samples = min(len(train_dataset), self.args.max_train_samples)
            self.train_dataset = train_dataset.select(range(max_train_samples))

        self.logger.debug(f"Num train_samples: {len(train_dataset)}")
        self.logger.debug("Tokenized training example:")
        self.logger.debug(self.tokenizer.decode(train_dataset[0]["input_ids"]))

        eval_dataset = None
        max_eval_samples = 0
        eval_dataset = lm_datasets["validation"]
        max_eval_samples = len(eval_dataset)
        if self.args.max_eval_samples is not None and self.args.max_eval_samples > 0:
            max_eval_samples = min(len(eval_dataset), self.args.max_eval_samples)
            self.dev_dataset = eval_dataset.select(range(max_eval_samples))

        self.print_example(self.train_dataset[0])

    @property
    def deepspeed_offload(self) -> bool:
        strategy = self.trainer.strategy
        if isinstance(strategy, DeepSpeedStrategy):
            config = strategy.config["zero_optimization"]
            return config.get("offload_optimizer") or config.get("offload_param")
        return False

    @property
    def is_deepspeed_zero3_enabled(self):
        strategy = self.trainer.strategy
        if isinstance(strategy, DeepSpeedStrategy):
            config = strategy.config["zero_optimization"]
            return config.get("stage") == 3

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            num_workers=4,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.dev_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            num_workers=4,
            pin_memory=True,
        )

    def training_step(self, batch, batch_idx):
        output = self.model(**batch)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], on_step=True, sync_dist=True)
        self.log("train_loss", output.loss, on_step=True, prog_bar=True, sync_dist=True)
        return output.loss

    def validation_step(self, batch, batch_idx):
        output = self.model(**batch)
        self.log("eval_loss", output.loss, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        params_decay = [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)]
        params_nodecay = [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)]

        optim_groups = [
            {"params": params_decay, "weight_decay": self.args.weight_decay},
            {"params": params_nodecay, "weight_decay": 0.0},
        ]

        if self.deepspeed_offload:
            optimizer = DeepSpeedCPUAdam(optim_groups, lr=self.args.learning_rate, eps=self.arsg.adam_epsilon)

        optimizer = torch.optim.AdamW(optim_groups, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        # num_gpus = self.trainer.num_devices
        # # 注：只有在使用pytorch Lightning的LightningDataModule 时候才可以使用该方式回去训练集大小
        # # print(f"len train dataloader==={len(self.train_dataloader())}")

        # t_total = (
        #     len(self.train_dataloader()) // (self.trainer.accumulate_grad_batches * num_gpus) + 1
        # ) * self.args.max_epochs

        t_total = self.trainer.estimated_stepping_batches
        warmup_steps = int(self.args.warmup_proportion * t_total)
        # print(f"totla={t_total},step_batch={stepping_batches},w={warmup_steps},warm_up=={warmup_steps}")

        if self.args.lr_scheduler_type == "onecycle":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.args.learning_rate,
                pct_start=float(warmup_steps / t_total),
                final_div_factor=self.args.final_div_factor,
                total_steps=t_total,
                anneal_strategy="linear",
            )

        elif self.args.lr_scheduler_type == "linear":
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=t_total,
            )
        elif self.args.lr_scheduler_type == "polydecay":
            scheduler = get_polynomial_decay_schedule_with_warmup(
                optimizer,
                warmup_steps,
                t_total,
                lr_end=self.args.learning_rate / 4.0,
            )
        elif self.args.lr_scheduler_type == "cosine":
            scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, t_total)

        else:
            raise ValueError("lr_scheduler does not exist.")
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        if self.global_rank == 0:
            save_path = os.path.join(self.args.output_dir, "hf_tokenizer")
            self.tokenizer.save_pretrained(save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train llm model")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output_dir/",
        help="path to save model and checkpoint .it is the root dir",
    )
    parser.add_argument("--train_data", type=str, default="", help="the input train file or train data directory")
    parser.add_argument("--test_data", type=str, default="", help="test data path")
    parser.add_argument("--dev_data", type=str, default="", help="dev data path")
    parser.add_argument(
        "--dataset_name", type=str, default=None, help="The name of the dataset to use (via the datasets library)."
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--max_train_samples", type=int, default=None, help="truncate the number of training examples to this"
    )
    parser.add_argument(
        "--max_eval_samples", type=int, default=None, help=" truncate the number of evaluation examples to this"
    )

    parser.add_argument("--streaming", action="store_true", help="Enable streaming mode")
    parser.add_argument(
        "--validation_split_percentage",
        type=int,
        default=1,
        help="The percentage of the train set used as validation set in case there's no validation split",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument("--deepspeed", type=str, default=None, help="deepspeed config file path")
    parser.add_argument(
        "--precision",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--lr_scheduler_type",
        choices=["linear", "onecycle", "polydecay", "cosine"],
        default="cosine",
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
        type=float,
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
        "--neft-alpha",
        type=float,
        default=0,
        help="The alpha parameter to control the noise magnitude in NEFTune. value can be 5.",
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
        help="quantization training like 4bit 8bit",
    )
    parser.add_argument("--use_lora", type=bool, default=True, help="using lora")

    parser.add_argument("--lora_rank", type=int, default=8, help="The intrinsic dimension for LoRA fine-tuning.")
    parser.add_argument(
        "--lora_alpha",
        type=float,
        default=32,
        help="The scale factor for LoRA fine-tuning (similar with the learning rate)",
    )
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="")
    parser.add_argument(
        "--lora_target",
        type=str,
        default=None,
        help="lora target name",
    )
    parser.add_argument("--lora_ckpt_path", type=str, default=None, help="")
    parser.add_argument("--max_steps", type=int, default=-1, help="max train steps")
    parser.add_argument("--fsdp", action="store_true", help="using fsdp strategy for training")
    parser.add_argument(
        "--sft_packing", action="store_true", help="packing multi round qa in supervised  fintuning stage"
    )
    arg = parser.parse_args()
    # 检查会冲突的参数
    if arg.deepspeed and arg.fsdp:
        # 训练策略智能选一个
        raise ValueError("Either --deepspeed or --fsdp has to be provided")

    if arg.seed is not None:
        seed_everything(arg.seed)
    # 先加载模型，再加载数据
    model = SupervisedFintuningModule(arg)

    if arg.deepspeed:
        strategy = DeepSpeedStrategy(config=arg.deepspeed)
    elif arg.fsdp:
        strategy = FSDPStrategy()
    else:
        strategy = "auto"

    # 添加常用的checkpoint
    callbacks = []
    checkpoint = HFModelCheckpoint(
        monitor="train_loss",
        dirpath=arg.output_dir,
        every_n_train_steps=arg.save_steps,
        filename="sn-generate-{epoch:02d}-{train_loss:.2f}",
        save_last=True,
        save_hf=True,
    )
    callbacks.append(checkpoint)

    trainer = Trainer(
        devices="auto",
        max_epochs=arg.max_epochs,
        strategy=strategy,
        max_steps=arg.max_steps,
        callbacks=callbacks,
        log_every_n_steps=1,
        num_sanity_val_steps=0,
        default_root_dir=arg.output_dir,
        accumulate_grad_batches=arg.gradient_accumulation_steps,
    )
    trainer.fit(model)

    trainer.test(model)
