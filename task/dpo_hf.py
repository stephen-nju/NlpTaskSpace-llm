# -*- coding: utf-8 -*-
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text file or a dataset.
part of this code is adapted from https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py
"""

import glob
import math
import os
from copy import deepcopy
from dataclasses import dataclass, field
from itertools import chain
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import torch
from datasets import load_dataset
from loguru import logger
from peft import (
    LoraConfig,
    PeftModel,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    BloomForCausalLM,
    BloomTokenizerFast,
    HfArgumentParser,
    LlamaForCausalLM,
    LlamaTokenizer,
    Seq2SeqTrainingArguments,
    is_torch_tpu_available,
    set_seed,
)
from transformers.utils.versions import require_version
from trl import DPOTrainer

from llm.src.utils import find_all_linear_names, print_trainable_parameters

try:
    from transformers.integrations import is_deepspeed_zero3_enabled

except ImportError:  # https://github.com/huggingface/transformers/releases/tag/v4.33.1
    from transformers.deepspeed import is_deepspeed_zero3_enabled

MODEL_CLASSES = {
    "bloom": (AutoConfig, BloomForCausalLM, BloomTokenizerFast),
    "chatglm": (AutoConfig, AutoModel, AutoTokenizer),
    "llama": (AutoConfig, LlamaForCausalLM, LlamaTokenizer),
    "baichuan": (AutoConfig, AutoModelForCausalLM, AutoTokenizer),
    "auto": (AutoConfig, AutoModelForCausalLM, AutoTokenizer),
}


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_type: str = field(
        default=None, metadata={"help": "Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys())}
    )
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )

    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": ("The tokenizer for weights initialization.Don't set if you want to train a model from scratch.")
        },
    )
    quantization: Optional[str] = field(
        default="bnb", metadata={"help": "qunatization method", "choices": ["bnb", "gptq", "awq"]}
    )

    quantization_bit: Optional[str] = field(
        default=None, metadata={"help": ("quantization bit"), "choices": ["4bit", "8bit"]}
    )

    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )

    use_fast_tokenizer: bool = field(
        default=False,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )

    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )

    device_map: Optional[str] = field(
        default="auto",
        metadata={"help": "Device to map model to. If `auto` is passed, the device will be selected automatically. "},
    )

    trust_remote_code: bool = field(
        default=True,
        metadata={"help": "Whether to trust remote code when loading a model from a remote checkpoint."},
    )

    def __post_init__(self):
        if self.model_type is None:
            raise ValueError(
                "You must specify a valid model_type to run training. Available model types are "
                + ", ".join(MODEL_CLASSES.keys())
            )

        if self.model_name_or_path is None:
            raise ValueError("You must specify a valid model_name_or_path to run training.")


@dataclass
class DataArguments:

    """

    Arguments pertaining to what data we are going to input our model for training and eval.

    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )

    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )

    train_data: Optional[str] = field(default=None, metadata={"help": "The train text data file folder."})

    dev_data: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on text file folder."},
    )

    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )

    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )

    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})

    overwrite_cache: bool = field(default=False, metadata={"help": "Overwrite the cached training and evaluation sets"})

    validation_split_percentage: Optional[int] = field(
        default=1,
        metadata={"help": "The percentage of the train set used as validation set in case there's no validation split"},
    )

    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )

    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )

    def __post_init__(self):
        if self.streaming:
            require_version("datasets>=2.0.0", "The streaming feature requires `datasets>=2.0.0`")


@dataclass
class ScriptArguments:
    use_peft: bool = field(default=True, metadata={"help": "Whether to use peft"})
    target_modules: Optional[str] = field(default="all")
    lora_rank: Optional[int] = field(default=8)
    lora_dropout: Optional[float] = field(default=0.05)
    lora_alpha: Optional[float] = field(default=32.0)
    modules_to_save: Optional[str] = field(default=None)
    peft_ckpt_path: Optional[str] = field(default=None)
    qlora: bool = field(default=False, metadata={"help": "Whether to use qlora"})


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, Seq2SeqTrainingArguments, ScriptArguments))
    model_args, data_args, training_args, script_args = parser.parse_args_into_dataclasses()
    logger.info(f"Model args: {model_args}")
    logger.info(f"Data args: {data_args}")
    logger.info(f"Training args: {training_args}")
    logger.info(f"Script args: {script_args}")
    logger.info(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )

    # Set seed before initializing model.
    set_seed(training_args.seed)
    # Load tokenizer
    config_class, model_class, tokenizer_class = MODEL_CLASSES[model_args.model_type]
    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "trust_remote_code": model_args.trust_remote_code,
    }
    tokenizer_name_or_path = model_args.tokenizer_name_or_path
    if not tokenizer_name_or_path:
        tokenizer_name_or_path = model_args.model_name_or_path
    tokenizer = tokenizer_class.from_pretrained(tokenizer_name_or_path, **tokenizer_kwargs)

    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            streaming=data_args.streaming,
        )
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
                streaming=data_args.streaming,
            )

            raw_datasets["train"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
                streaming=data_args.streaming,
            )

    else:
        data_files = {}
        dataset_args = {}
        if data_args.train_data is not None:
            train_dof = [s.strip() for s in data_args.train_data.strip().split(",")]
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

        if data_args.dev_data is not None:
            dev_dof = [s.strip() for s in data_args.dev_data.split(",")]
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
                        raise ValueError(f"train files must be same type, e.g. all txt or all jsonl, but got {types}")
                    dev_files.extend(files)
                else:
                    raise ValueError("train data should be valid file path or direcotory path split by ','")
            logger.info(f"eval files: {dev_files}")
            data_files["validation"] = dev_files

        logger.info(f"loading data files={data_files}")
        extension = "text" if data_files["train"][0].endswith("txt") else "json"
        if extension == "text":
            dataset_args["keep_linebreaks"] = data_args.keep_linebreaks

        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            **dataset_args,
        )

        # If no validation data is there, validation_split_percentage will be used to divide the dataset.

        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
                **dataset_args,
            )

            raw_datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
                **dataset_args,
            )

    logger.info(f"Raw datasets: {raw_datasets}")

    # Preprocessing the datasets.

    if training_args.do_train:
        column_names = list(raw_datasets["train"].features)

    else:
        column_names = list(raw_datasets["validation"].features)

    train_dataset = None
    max_train_samples = 0
    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        max_train_samples = len(train_dataset)
        if data_args.max_train_samples is not None and data_args.max_train_samples > 0:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

        logger.debug(f"Num train_samples: {len(train_dataset)}")
        logger.debug("Tokenized training example:")
        logger.debug(tokenizer.decode(train_dataset[0]["input_ids"]))
    eval_dataset = None

    max_eval_samples = 0
    if training_args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")

        eval_dataset = raw_datasets["validation"]
        max_eval_samples = len(eval_dataset)
        if data_args.max_eval_samples is not None and data_args.max_eval_samples > 0:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

        logger.debug(f"Num eval_samples: {len(eval_dataset)}")
        logger.debug("Tokenized eval example:")
        logger.debug(tokenizer.decode(eval_dataset[0]["input_ids"]))

    # Load model

    if model_args.model_type and model_args.model_name_or_path:
        torch_dtype = (
            model_args.torch_dtype
            if model_args.torch_dtype in ["auto", None]
            else getattr(torch, model_args.torch_dtype)
        )

        if script_args.qlora and (len(training_args.fsdp) > 0 or is_deepspeed_zero3_enabled()):
            logger.warning("FSDP and ZeRO3 are both currently incompatible with QLoRA.")

        config = config_class.from_pretrained(
            model_args.model_name_or_path,
            torch_dtype=torch_dtype,
            trust_remote_code=model_args.trust_remote_code,
            cache_dir=model_args.cache_dir,
        )

        if model_args.quantization_bit is not None:
            logger.info(f"Quantized to {model_args.quantization_bit}")
            if model_args.quantization == "bnb":
                if model_args.quantization_bit == "4bit":
                    # load_in_4bit = (True,)
                    quantization_config = BitsAndBytesConfig(
                        bnb_4bit_compute_dtype=torch.bfloat16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                    )
                elif model_args.quantization_bit == "8bit":
                    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                else:
                    raise ValueError("unsupport quantization_bit")

                model = model_class.from_pretrained(
                    model_args.model_name_or_path,
                    from_tf=bool(".ckpt" in model_args.model_name_or_path),
                    config=config,
                    quantization_config=quantization_config,
                    low_cpu_mem_usage=True,
                    trust_remote_code=model_args.trust_remote_code,
                )
        else:
            print(f"model config===\n{config}")
            model = model_class.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                trust_remote_code=model_args.trust_remote_code,
                torch_dtype=torch_dtype,
            )

    else:
        raise ValueError("Error, model_name_or_path is None, dpo train must be loaded from a pre-trained model")

    if script_args.use_peft:
        logger.info("Fine-tuning method: LoRA(PEFT)")
        if script_args.peft_ckpt_path is not None:
            logger.info(f"Peft from pre-trained model: {script_args.peft_ckpt_path}")
            model = PeftModel.from_pretrained(model, script_args.peft_ckpt_path, is_trainable=True)
        else:
            logger.info("Init new peft model")
            if model_args.quantization_bit:
                model = prepare_model_for_kbit_training(model, training_args.gradient_checkpointing)

            target_modules = script_args.target_modules.split(",") if script_args.target_modules else None
            if target_modules and "all" in target_modules:
                target_modules = find_all_linear_names(model, quantization_bits=model_args.quantization_bit)

            modules_to_save = script_args.modules_to_save
            if modules_to_save is not None:
                modules_to_save = modules_to_save.split(",")
                # Resize the embedding layer to match the new tokenizer
                embedding_size = model.get_input_embeddings().weight.shape[0]
                if len(tokenizer) > embedding_size:
                    model.resize_token_embeddings(len(tokenizer))

            logger.info(f"Peft target_modules: {target_modules}")
            logger.info(f"Peft lora_rank: {script_args.lora_rank}")

            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                target_modules=target_modules,
                inference_mode=False,
                r=script_args.lora_rank,
                lora_alpha=script_args.lora_alpha,
                lora_dropout=script_args.lora_dropout,
                modules_to_save=modules_to_save,
            )
            model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    else:
        logger.info("Fine-tuning method: Full parameters training")
        model = model.float()
        print_trainable_parameters(model)
    # Initialize our Trainer

    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    else:
        model.config.use_cache = True
    model.enable_input_require_grads()

    model.is_parallelizable = True
    model.model_parallel = True

    trainer = DPOTrainer(
        model=model,
        ref_model=deepcopy(model),
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
    )

    # Training

    if training_args.do_train:
        logger.info("*** Train ***")

        logger.debug(f"Train dataloader example: {next(iter(trainer.get_train_dataloader()))}")

        checkpoint = None

        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint

        train_result = trainer.train(resume_from_checkpoint=checkpoint)

        metrics = train_result.metrics

        metrics["train_samples"] = max_train_samples

        trainer.log_metrics("train", metrics)

        trainer.save_metrics("train", metrics)

        trainer.save_state()

        model.config.use_cache = True  # enable cache after training

        tokenizer.padding_side = "left"  # restore padding side

        tokenizer.init_kwargs["padding_side"] = "left"

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = max_eval_samples
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        if trainer.is_world_process_zero():
            logger.debug(f"Eval metrics: {metrics}")


if __name__ == "__main__":
    main()
