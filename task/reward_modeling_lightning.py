# ----*----coding:utf8-----*---

import argparse
import functools
import glob
import math
import os
from itertools import chain
from types import MethodType
from typing import Any, Dict, List, Mapping, Tuple, Union

import numpy as np
import tiktoken
import torch
from datasets import load_dataset
from deepspeed.ops.adam import DeepSpeedCPUAdam
from lightning import LightningModule, Trainer, seed_everything
from lightning.pytorch.strategies import DeepSpeedStrategy, FSDPStrategy
from loguru import logger
from peft import (
    LoraConfig,
    PeftModel,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from torch.utils.data import DataLoader
from transformers import (
    SAFE_WEIGHTS_NAME,
    WEIGHTS_NAME,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    BloomForCausalLM,
    BloomTokenizerFast,
    LlamaTokenizer,
    PreTrainedModel,
    cached_file,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)
from trl import AutoModelForCausalLMWithValueHead

from llm.src.callbacks import HFModelCheckpoint
from llm.src.constant import IGNORE_INDEX
from llm.src.datasets.preprocessing import preprocess_reward_function
from llm.src.datasets.template import get_template_and_fix_tokenizer, register_template
from llm.src.utils import find_all_linear_names, print_trainable_parameters
from metrics.language_model import AccMetric

MODEL_CLASSES = {
    "bloom": (AutoConfig, BloomForCausalLM, BloomTokenizerFast),
    "llama": (AutoConfig, AutoModelForCausalLM, LlamaTokenizer),
    "auto": (AutoConfig, AutoModelForCausalLM, AutoTokenizer),
}


class RewardModule(LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        # 加载模型
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            use_fast=not args.use_slow_tokenizer,
            trust_remote_code=True,
        )
        self.acc_metrics = AccMetric()
        self.template = get_template_and_fix_tokenizer(args.template_name, self.tokenizer)
        self.save_hyperparameters()

    def print_example(self, example):
        self.trainer.print("input_ids:\n{}".format(example["input_ids"]))
        self.trainer.print("inputs:\n{}".format(self.tokenizer.decode(example["input_ids"], skip_special_tokens=False)))
        self.trainer.print("label_ids:\n{}".format(example["labels"]))
        self.trainer.print(
            "labels:\n{}".format(
                self.tokenizer.decode(
                    [d if d != IGNORE_INDEX else self.tokenizer.pad_token_id for d in example["labels"]],
                    skip_special_tokens=False,
                )
            )
        )

    @staticmethod
    def load_valuehead_params(path_or_repo_id: str, model_args) -> Dict[str, torch.Tensor]:
        r"""
        Loads value head parameters from Hugging Face Hub or local disk.

        Returns: dict with keys `v_head.summary.weight` and `v_head.summary.bias`.
        """
        kwargs = {
            "path_or_repo_id": path_or_repo_id,
            "cache_dir": model_args.cache_dir,
        }

        try:
            from safetensors import safe_open

            vhead_file = cached_file(filename=SAFE_WEIGHTS_NAME, **kwargs)
            with safe_open(vhead_file, framework="pt", device="cpu") as f:
                return {
                    "v_head.summary.weight": f.get_tensor("v_head.summary.weight"),
                    "v_head.summary.bias": f.get_tensor("v_head.summary.bias"),
                }
        except Exception as err:
            logger.info("Failed to load {}: {}".format(SAFE_WEIGHTS_NAME, str(err)))

        try:
            vhead_file = cached_file(filename=WEIGHTS_NAME, **kwargs)
            return torch.load(vhead_file, map_location="cpu")
        except Exception as err:
            logger.info("Failed to load {}: {}".format(WEIGHTS_NAME, str(err)))

        logger.warning("Provided path ({}) does not contain valuehead weights.".format(path_or_repo_id))
        return None

    def setup(self, stage):
        if self.is_deepspeed_zero3_enabled:
            from transformers.integrations import HfDeepSpeedConfig

            self.ds_config = HfDeepSpeedConfig(self.trainer.strategy.config)

        config_class, model_class, tokenizer_class = MODEL_CLASSES[self.args.model_type]

        if self.args.model_name_or_path:
            torch_dtype = (
                self.args.torch_dtype
                if self.args.torch_dtype in ["auto", None]
                else getattr(torch, self.args.torch_dtype)
            )
            config = config_class.from_pretrained(
                self.args.model_name_or_path,
                num_labels=1,
                torch_dtype=torch_dtype,
                trust_remote_code=self.args.trust_remote_code,
                cache_dir=self.args.cache_dir,
            )

            if self.args.model_type in ["bloom", "llama"]:
                model = model_class.from_pretrained(
                    self.args.model_name_or_path,
                    config=config,
                    torch_dtype=torch_dtype,
                    load_in_4bit=self.args.load_in_4bit,
                    load_in_8bit=self.args.load_in_8bit,
                    trust_remote_code=self.args.trust_remote_code,
                )

            else:
                model = model_class.from_pretrained(
                    self.args.model_name_or_path,
                    config=config,
                    cache_dir=self.args.cache_dir,
                )
        else:
            raise ValueError("Error, model_name_or_path is None, RM must be loaded from a pre-trained model")

        model = AutoModelForCausalLMWithValueHead.from_pretrained(model)

        # 参考llama facotry构建valuehead model
        def patch_valuehead_model(model: "AutoModelForCausalLMWithValueHead") -> None:
            def tie_weights(self: "AutoModelForCausalLMWithValueHead") -> None:
                if isinstance(self.pretrained_model, PreTrainedModel):
                    self.pretrained_model.tie_weights()

            def get_input_embeddings(self: "AutoModelForCausalLMWithValueHead") -> torch.nn.Module:
                if isinstance(self.pretrained_model, PreTrainedModel):
                    return self.pretrained_model.get_input_embeddings()

            ignore_modules = [name for name, _ in model.named_parameters() if "pretrained_model" in name]
            setattr(model, "_keys_to_ignore_on_save", ignore_modules)
            setattr(model, "tie_weights", MethodType(tie_weights, model))
            setattr(model, "get_input_embeddings", MethodType(get_input_embeddings, model))

        patch_valuehead_model(model)

        if self.args.vhead_ckpt_path is not None:
            vhead_path = self.args.vhead_ckpt_path
        else:
            vhead_path = self.args.model_name_or_path

        vhead_param = self.load_valuehead_params(vhead_path, self.args)
        if vhead_param is not None:
            model.load_state_dict(vhead_param, strict=True)

        # Load tokenizer
        if getattr(config, "model_type", None) == "bloom":
            self.args.use_fast_tokenizer = True
        tokenizer_kwargs = {
            "cache_dir": self.args.cache_dir,
            "use_fast": self.args.use_fast_tokenizer,
            "trust_remote_code": self.args.trust_remote_code,
        }

        # 如果没有指定
        tokenizer_name_or_path = self.args.tokenizer_name_or_path
        if not tokenizer_name_or_path:
            tokenizer_name_or_path = self.args.model_name_or_path
        tokenizer = tokenizer_class.from_pretrained(tokenizer_name_or_path, **tokenizer_kwargs)

        template = get_template_and_fix_tokenizer(self.args.template_name, tokenizer)

        if self.args.use_peft:
            logger.info("Fine-tuning method: LoRA(PEFT)")
            if self.args.peft_ckpt_path is not None:
                logger.info(f"Peft from pre-trained model: {self.args.peft_ckpt_path}")
                model = PeftModel.from_pretrained(model, self.args.peft_ckpt_path, is_trainable=True)
            else:
                logger.info("Init new peft model")
                if self.args.quantization_bit:
                    model = prepare_model_for_kbit_training(model)
                if self.args.lora_target == "all":
                    lora_target = find_all_linear_names(model, self.args.quantization_bit)
                else:
                    # support custom target modules/layers of LoRA
                    lora_target = [target.strip() for target in self.args.lora_target.split(",")]
                modules_to_save = self.args.modules_to_save
                if modules_to_save is not None:
                    modules_to_save = modules_to_save.split(",")

                logger.info(f"Peft lora_target: {lora_target}")
                logger.info(f"Peft lora_rank: {self.args.lora_rank}")

                peft_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    target_modules=lora_target,
                    inference_mode=False,
                    r=self.args.lora_rank,
                    lora_alpha=self.args.lora_alpha,
                    lora_dropout=self.args.lora_dropout,
                    modules_to_save=modules_to_save,
                )
                model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()

        else:
            logger.info("Fine-tuning method: Full parameters training")
            print_trainable_parameters(model)

        # Get reward dataset for tuning the reward model.
        if self.args.dataset_name is not None:
            # Downloading and loading a dataset from the hub.
            raw_datasets = load_dataset(
                self.args.dataset_name,
                self.args.dataset_config_name,
                cache_dir=self.args.cache_dir,
            )

            if "validation" not in raw_datasets.keys():
                raw_datasets["validation"] = load_dataset(
                    self.args.dataset_name,
                    self.args.dataset_config_name,
                    split=f"train[:{self.args.validation_split_percentage}%]",
                    cache_dir=self.args.cache_dir,
                )
                raw_datasets["train"] = load_dataset(
                    self.args.dataset_name,
                    self.args.dataset_config_name,
                    split=f"train[{self.args.validation_split_percentage}%:]",
                    cache_dir=self.args.cache_dir,
                )

        else:
            data_files = {}
            if self.args.train_data is not None:
                train_dof = [s.strip() for s in self.args.train_data.strip().split(",")]
            else:
                raise ValueError("train data should not be none")
            # direcotory_or_file
            train_files = []
            for dof in train_dof:
                if os.path.exists(dof) and os.path.isfile(dof):
                    train_files.append(dof)
                elif os.path.exists(dof) and os.path.isdir(dof):
                    files = (
                        glob.glob(f"{dof}/**/*.txt", recursive=True)
                        + glob.glob(f"{dof}/**/*.json", recursive=True)
                        + glob.glob(f"{dof}/**/*.jsonl", recursive=True)
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
                            glob.glob(f"{dof}/**/*.txt", recursive=True)
                            + glob.glob(f"{dof}/**/*.json", recursive=True)
                            + glob.glob(f"{dof}/**/*.jsonl", recursive=True)
                        )
                        types = [f.split(".")[-1] for f in files]
                        if len(set(types)) > 1:
                            raise ValueError(
                                f"train files must be same type, e.g. all txt or all jsonl, but got {types}"
                            )
                        dev_files.extend(files)
                    else:
                        raise ValueError("train data should be valid file path or direcotory path split by ','")
                logger.info(f"eval files: {dev_files}")
                data_files["validation"] = dev_files

            raw_datasets = load_dataset(
                "json",
                data_files=data_files,
                cache_dir=self.args.cache_dir,
            )

            # If no validation data is there, validation_split_percentage will be used to divide the dataset.
            if "validation" not in raw_datasets.keys():
                raw_datasets["validation"] = load_dataset(
                    "json",
                    data_files=data_files,
                    split=f"train[:{self.args.validation_split_percentage}%]",
                    cache_dir=self.args.cache_dir,
                )

                raw_datasets["train"] = load_dataset(
                    "json",
                    data_files=data_files,
                    split=f"train[{self.args.validation_split_percentage}%:]",
                    cache_dir=self.args.cache_dir,
                )

        logger.info(f"Raw datasets: {raw_datasets}")
        # Preprocessing the datasets
        preprocessing_function_train = functools.partial(
            preprocess_reward_function,
            tokenizer=tokenizer,
            template=template,
            max_source_length=self.args.max_source_length,
            max_target_length=self.args.max_target_length,
        )

        full_max_length = self.args.max_source_length + self.args.max_target_length
        train_dataset = None
        max_train_samples = 0
        if self.args.do_train:
            if "train" not in raw_datasets:
                raise ValueError("--do_train requires a train dataset")

            train_dataset = raw_datasets["train"]
            max_train_samples = len(train_dataset)
            if self.args.max_train_samples is not None and self.args.max_train_samples > 0:
                max_train_samples = min(len(train_dataset), self.args.max_train_samples)
                train_dataset = train_dataset.select(range(max_train_samples))
            logger.debug(f"Example train_dataset[0]: {train_dataset[0]}")

            with self.args.main_process_first(desc="Train dataset tokenization"):
                tokenized_dataset = train_dataset.shuffle().map(
                    preprocessing_function_train,
                    batched=True,
                    num_proc=self.args.preprocessing_num_workers,
                    remove_columns=train_dataset.column_names,
                    load_from_cache_file=not self.args.overwrite_cache,
                    desc="Running tokenizer on dataset",
                )

                train_dataset = tokenized_dataset.filter(
                    lambda x: 0 < len(x["input_ids_rejected"]) <= full_max_length
                    and 0 < len(x["input_ids_chosen"]) <= full_max_length
                )
                logger.debug(f"Num train_samples: {len(train_dataset)}")
                logger.debug("Tokenized training example:")
                logger.debug(tokenizer.decode(train_dataset[0]["input_ids_chosen"]))

        eval_dataset = None
        max_eval_samples = 0
        if self.args.do_eval:
            with self.args.main_process_first(desc="Eval dataset tokenization"):
                if "validation" not in raw_datasets:
                    raise ValueError("--do_eval requires a validation dataset")
                eval_dataset = raw_datasets["validation"]
                max_eval_samples = len(eval_dataset)
                if self.args.max_eval_samples is not None and self.args.max_eval_samples > 0:
                    max_eval_samples = min(len(eval_dataset), self.args.max_eval_samples)
                    eval_dataset = eval_dataset.select(range(max_eval_samples))
                logger.debug(f"Example eval_dataset[0]: {eval_dataset[0]}")
                tokenized_dataset = eval_dataset.map(
                    preprocessing_function_train,
                    batched=True,
                    num_proc=self.args.preprocessing_num_workers,
                    remove_columns=eval_dataset.column_names,
                    load_from_cache_file=not self.args.overwrite_cache,
                    desc="Running tokenizer on dataset",
                )

                eval_dataset = tokenized_dataset.filter(
                    lambda x: 0 < len(x["input_ids_rejected"]) <= full_max_length
                    and 0 < len(x["input_ids_chosen"]) <= full_max_length
                )
                logger.debug(f"Num eval_samples: {len(eval_dataset)}")
                logger.debug("Tokenized eval example:")
                logger.debug(tokenizer.decode(eval_dataset[0]["input_ids_chosen"]))

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
            collate_fn=self.fault_tolerance_data_collator,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.dev_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            num_workers=4,
            pin_memory=True,
            collate_fn=self.fault_tolerance_data_collator,
        )

    def training_step(self, batch, batch_idx):
        output = self.model(**batch)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], on_step=True, sync_dist=True)
        self.log("train_loss", output.loss, on_step=True, prog_bar=True, sync_dist=True)
        return output.loss

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
            if hasattr(self.tokenizer, "add_eos_token"):  # for LLaMA tokenizer
                setattr(self.tokenizer, "add_eos_token", self.add_eos_token_flag)
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
    parser.add_argument("--train_data", type=str, default=None, help="the input train file or train data directory")
    parser.add_argument("--test_data", type=str, default=None, help="test data path")
    parser.add_argument("--dev_data", type=str, default=None, help="dev data path")
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
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
    )
    parser.add_argument("--save_total_limit", type=int, default=None, help="save total limit")

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
        default=None,
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
    model = RewardModule(arg)

    if arg.deepspeed:
        strategy = DeepSpeedStrategy(config=arg.deepspeed)
    elif arg.fsdp:
        strategy = FSDPStrategy()
    else:
        strategy = "auto"

    # 添加常用的checkpoint
    callbacks = []

    last_checkpoint = HFModelCheckpoint(
        dirpath=arg.output_dir,
        filename="sn-generate-{epoch:02d}-{train_loss:.2f}",
        save_last=True,
        save_top_k=1,
        save_on_train_epoch_end=True,
        every_n_epochs=1,
        save_hf=True,
    )

    callbacks.append(last_checkpoint)

    if arg.checkpointing_steps == "epoch":
        checkpoint = HFModelCheckpoint(
            dirpath=arg.output_dir,
            filename="sn-generate-{epoch:02d}-{train_loss:.2f}",
            save_last=True,
            save_top_k=-1,
            every_n_epochs=1,
        )
        callbacks.append(checkpoint)
    else:
        if arg.save_steps is not None and arg.save_total_limit is None:
            every_n_train_steps = arg.save_steps
            save_top_k = -1
        elif arg.save_steps is not None and arg.save_total_limit is not None:
            every_n_train_steps = arg.save_steps
            save_top_k = arg.save_total_limit
        else:
            every_n_train_steps = None
            save_top_k = None

        checkpoint = HFModelCheckpoint(
            monitor="step",
            mode="max",
            dirpath=arg.output_dir,
            filename="sn-generate-{step:02d}-{train_loss:.2f}",
            every_n_train_steps=every_n_train_steps,
            save_top_k=save_top_k,
            save_on_train_epoch_end=True,
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
