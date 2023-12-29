# -*- coding: utf-8 -*-

import math
import os
from dataclasses import dataclass, field
from glob import glob
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple, Union

import tiktoken
import torch
from datasets import load_dataset
from loguru import logger
from peft import (
    LoraConfig,
    PeftModel,
    TaskType,
    get_peft_model,
    prepare_model_for_int8_training,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import Dataset
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BloomForSequenceClassification,
    BloomTokenizerFast,
    DataCollatorWithPadding,
    HfArgumentParser,
    LlamaForSequenceClassification,
    LlamaTokenizer,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer import TRAINING_ARGS_NAME

from llm.src.datasets.preprocessing import preprocess_reward_function
from llm.src.utils import print_trainable_parameters

if TYPE_CHECKING:
    from transformers import PreTrainedModel


MODEL_CLASSES = {
    "bloom": (AutoConfig, BloomForSequenceClassification, BloomTokenizerFast),
    "llama": (AutoConfig, LlamaForSequenceClassification, LlamaTokenizer),
    "auto": (AutoConfig, AutoModelForSequenceClassification, AutoTokenizer),
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

    load_in_4bit: bool = field(default=False, metadata={"help": "Whether to load the model in 4bit mode or not."})

    load_in_8bit: bool = field(default=False, metadata={"help": "Whether to load the model in 8bit mode or not."})

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

    train_data: Optional[str] = field(default=None, metadata={"help": "The input jsonl data file folder."})

    dev_data: Optional[str] = field(
        default=None,
        metadata={"help": "The evaluation jsonl file folder."},
    )
    max_source_length: Optional[int] = field(default=256, metadata={"help": "Max length of prompt input text"})
    max_target_length: Optional[int] = field(default=256, metadata={"help": "Max length of output text"})
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
    overwrite_cache: bool = field(default=False, metadata={"help": "Overwrite the cached training and evaluation sets"})
    validation_split_percentage: Optional[int] = field(
        default=1,
        metadata={"help": "The percentage of the train set used as validation set in case there's no validation split"},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=4,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )


@dataclass
class ScriptArguments:
    use_peft: bool = field(default=True, metadata={"help": "Whether to use peft"})
    target_modules: Optional[str] = field(default="all")
    lora_rank: Optional[int] = field(default=8)
    lora_dropout: Optional[float] = field(default=0.05)
    lora_alpha: Optional[float] = field(default=32.0)
    modules_to_save: Optional[str] = field(default=None)
    peft_path: Optional[str] = field(default=None)


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    # Here, predictions is rewards_chosen and rewards_rejected.
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    # MSE
    mse = mean_squared_error(labels, preds)
    # MAE
    mae = mean_absolute_error(labels, preds)
    return {"mse": mse, "mae": mae}


class RewardDataCollatorWithPadding(DataCollatorWithPadding):
    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        拼接到一个batch中,前面是chosen_ids,后面是rehected_ids
        """
        features = [
            {
                "input_ids": feature["prompt_ids"] + feature[key],
                "attention_mask": [1] * (len(feature["prompt_ids"]) + len(feature[key])),
            }
            for key in ("chosen_ids", "rejected_ids")
            for feature in features
        ]

        return super().__call__(features)


class RewardTrainer(Trainer):

    """
    Trainer for reward models
        Define how to compute the reward loss. Use the InstructGPT pairwise logloss: https://arxiv.org/abs/2203.02155
    """

    def compute_loss(
        self, model: "PreTrainedModel", inputs: Dict[str, torch.Tensor], return_outputs: Optional[bool] = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        # Compute rewards
        _, _, values = model(**inputs, output_hidden_states=True, return_dict=True)
        unwrapped_model: "PreTrainedModel" = self.accelerator.unwrap_model(self.model)

        if getattr(unwrapped_model.config, "model_type", None) == "chatglm":
            values = torch.transpose(values, 0, 1)

        # Split the inputs and rewards into two parts, chosen and rejected
        batch_size = inputs["input_ids"].size(0) // 2
        chosen_input_ids, rejected_input_ids = inputs["input_ids"][:batch_size], inputs["input_ids"][batch_size:]
        chosen_rewards, rejected_rewards = values[:batch_size], values[batch_size:]
        chosen_scores, rejected_scores = [], []

        # Compute pairwise loss. Only backprop on the different tokens before padding
        # Inspired by: https://github.com/CarperAI/trlx/blob/main/examples/summarize_rlhf/reward_model/reward_model.py

        loss = 0
        for i in range(batch_size):
            chosen_length = (chosen_input_ids[i] != self.tokenizer.pad_token_id).nonzero()[-1] + 1
            rejected_length = (rejected_input_ids[i] != self.tokenizer.pad_token_id).nonzero()[-1] + 1
            check_divergence = (chosen_input_ids[i] != rejected_input_ids[i]).nonzero()

            if len(check_divergence) == 0:
                end_index = chosen_length
                div_index = end_index - 1
            else:
                end_index = max(chosen_length, rejected_length)
                div_index = check_divergence[0]

            assert div_index > 0
            chosen_trunc_rewards = chosen_rewards[i, div_index:end_index]
            rejected_trunc_rewards = rejected_rewards[i, div_index:end_index]
            if return_outputs:  # use the score on the last token except pad token for inference
                chosen_scores.append(chosen_rewards[i, chosen_length - 1])
                rejected_scores.append(rejected_rewards[i, rejected_length - 1])
            loss += -torch.nn.functional.logsigmoid(chosen_trunc_rewards - rejected_trunc_rewards).mean()

        loss = loss / batch_size

        if return_outputs:
            chosen_scores, rejected_scores = torch.stack(chosen_scores), torch.stack(rejected_scores)
            return loss, [loss, chosen_scores, rejected_scores]

        return loss

    # def compute_loss(self, model, inputs, return_outputs=False):
    #     rewards_chosen = model(input_ids=inputs["input_ids_chosen"], attention_mask=inputs["attention_mask_chosen"])[0]
    #     rewards_rejected = model(
    #         input_ids=inputs["input_ids_rejected"], attention_mask=inputs["attention_mask_rejected"]
    #     )[0]
    #     loss = -torch.nn.functional.logsigmoid(rewards_chosen - rewards_rejected).mean()
    #     if return_outputs:
    #         return loss, {"rewards_chosen": rewards_chosen, "rewards_rejected": rewards_rejected}
    #     return loss

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        if eval_dataset is None:
            eval_dataset = self.eval_dataset

        return super().evaluate(eval_dataset=eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        # Prepare inputs for chosen and rejected separately

        device = model.device

        inputs_chosen = {
            "input_ids": inputs["input_ids_chosen"].to(device),
            "attention_mask": inputs["attention_mask_chosen"].to(device),
        }

        outputs_chosen = model(**inputs_chosen)

        rewards_chosen = outputs_chosen.logits.detach()

        inputs_rejected = {
            "input_ids": inputs["input_ids_rejected"].to(device),
            "attention_mask": inputs["attention_mask_rejected"].to(device),
        }

        outputs_rejected = model(**inputs_rejected)

        rewards_rejected = outputs_rejected.logits.detach()

        # Keep the compute_loss method

        loss = -torch.nn.functional.logsigmoid(rewards_chosen - rewards_rejected).mean()

        if prediction_loss_only:
            return (loss, None, None)

        return (loss, rewards_chosen, rewards_rejected)

    def save_model(self, output_dir=None, _internal_call=False):
        """Save the LoRA model."""

        os.makedirs(output_dir, exist_ok=True)

        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

        self.model.save_pretrained(output_dir)


def save_model(model, tokenizer, args):
    """Save the model and the tokenizer."""

    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    # Take care of distributed/parallel training

    model_to_save = model.module if hasattr(model, "module") else model

    model_to_save.save_pretrained(output_dir)

    tokenizer.save_pretrained(output_dir)


class CastOutputToFloat(torch.nn.Sequential):

    """Cast the output of the model to float"""

    def forward(self, x):
        return super().forward(x).to(torch.float32)


def find_all_linear_names(peft_model, int4=False, int8=False):
    cls = torch.nn.Linear
    if int4 or int8:
        import bitsandbytes as bnb

        if int4:
            cls = bnb.nn.Linear4bit
        elif int8:
            cls = bnb.nn.Linear8bitLt
    lora_module_names = set()
    for name, module in peft_model.named_modules():
        if isinstance(module, cls):
            # last layer is not add to lora_module_names
            if "lm_head" in name:
                continue
            if "score" in name:
                continue
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    return sorted(lora_module_names)


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, ScriptArguments))

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

    # Load model
    config_class, model_class, tokenizer_class = MODEL_CLASSES[model_args.model_type]

    if model_args.model_name_or_path:
        torch_dtype = (
            model_args.torch_dtype
            if model_args.torch_dtype in ["auto", None]
            else getattr(torch, model_args.torch_dtype)
        )
        # world_size = int(os.environ.get("WORLD_SIZE", "1"))
        # if world_size > 1:
        #     model_args.device_map = {"": int(os.environ.get("LOCAL_RANK", "0"))}

        config = config_class.from_pretrained(
            model_args.model_name_or_path,
            num_labels=1,
            torch_dtype=torch_dtype,
            trust_remote_code=model_args.trust_remote_code,
            cache_dir=model_args.cache_dir,
        )

        if model_args.model_type in ["bloom", "llama"]:
            model = model_class.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                torch_dtype=torch_dtype,
                load_in_4bit=model_args.load_in_4bit,
                load_in_8bit=model_args.load_in_8bit,
                trust_remote_code=model_args.trust_remote_code,
            )

        else:
            model = model_class.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                cache_dir=model_args.cache_dir,
                ignore_mismatched_sizes=True,
            )
            model.to(training_args.device)

    else:
        raise ValueError("Error, model_name_or_path is None, RM must be loaded from a pre-trained model")

    # Load tokenizer
    if getattr(config, "model_type", None) == "bloom":
        model_args.use_fast_tokenizer = True

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "trust_remote_code": model_args.trust_remote_code,
    }

    # 如果没有指定
    tokenizer_name_or_path = model_args.tokenizer_name_or_path
    if not tokenizer_name_or_path:
        tokenizer_name_or_path = model_args.model_name_or_path
    tokenizer = tokenizer_class.from_pretrained(tokenizer_name_or_path, **tokenizer_kwargs)

    if tokenizer.eos_token_id is None:
        # TODO
        # for Qwen
        tokenizer.eos_token = "<|endoftext|>"
        logger.info("Add eos token: {}".format(tokenizer.eos_token))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        logger.info("Add pad token: {}".format(tokenizer.pad_token))

    if script_args.use_peft:
        logger.info("Fine-tuning method: LoRA(PEFT)")
        if script_args.peft_path is not None:
            logger.info(f"Peft from pre-trained model: {script_args.peft_path}")
            model = PeftModel.from_pretrained(model, script_args.peft_path, is_trainable=True)
        else:
            logger.info("Init new peft model")
            if model_args.load_in_8bit:
                model = prepare_model_for_int8_training(model)

            target_modules = script_args.target_modules.split(",") if script_args.target_modules else None

            if target_modules and "all" in target_modules:
                target_modules = find_all_linear_names(model, int4=False, int8=model_args.load_in_8bit)
            modules_to_save = script_args.modules_to_save

            if modules_to_save is not None:
                modules_to_save = modules_to_save.split(",")

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
        print_trainable_parameters(model)
    # Get reward dataset for tuning the reward model.
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
        )

        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
            )

            raw_datasets["train"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
            )

    else:
        data_files = {}
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

        raw_datasets = load_dataset(
            "json",
            data_files=data_files,
            cache_dir=model_args.cache_dir,
        )

        # If no validation data is there, validation_split_percentage will be used to divide the dataset.
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                "json",
                data_files=data_files,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
            )

            raw_datasets["train"] = load_dataset(
                "json",
                data_files=data_files,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
            )

    logger.info(f"Raw datasets: {raw_datasets}")
    # Preprocessing the datasets

    full_max_length = data_args.max_source_length + data_args.max_target_length

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
        logger.debug(f"Example train_dataset[0]: {train_dataset[0]}")
        with training_args.main_process_first(desc="Train dataset tokenization"):
            tokenized_dataset = train_dataset.shuffle().map(
                preprocess_reward_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=train_dataset.column_names,
                load_from_cache_file=not data_args.overwrite_cache,
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
    if training_args.do_eval:
        with training_args.main_process_first(desc="Eval dataset tokenization"):
            if "validation" not in raw_datasets:
                raise ValueError("--do_eval requires a validation dataset")
            eval_dataset = raw_datasets["validation"]
            max_eval_samples = len(eval_dataset)
            if data_args.max_eval_samples is not None and data_args.max_eval_samples > 0:
                max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
                eval_dataset = eval_dataset.select(range(max_eval_samples))

            logger.debug(f"Example eval_dataset[0]: {eval_dataset[0]}")

            tokenized_dataset = eval_dataset.map(
                preprocess_reward_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=eval_dataset.column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )

            eval_dataset = tokenized_dataset.filter(
                lambda x: 0 < len(x["input_ids_rejected"]) <= full_max_length
                and 0 < len(x["input_ids_chosen"]) <= full_max_length
            )
            logger.debug(f"Num eval_samples: {len(eval_dataset)}")
            logger.debug("Tokenized eval example:")
            logger.debug(tokenizer.decode(eval_dataset[0]["input_ids_chosen"]))
    # Initialize our Trainer
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    else:
        model.config.use_cache = True
    model.enable_input_require_grads()
    if torch.cuda.device_count() > 1:
        # Keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        data_collator=RewardDataCollatorWithPadding(
            tokenizer=tokenizer, max_length=full_max_length, padding="max_length"
        ),
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
        # trainer.save_state()

        model.config.use_cache = True  # enable cache after training

        if trainer.is_world_process_zero():
            logger.debug(f"Training metrics: {metrics}")
            logger.info(f"Saving model checkpoint to {training_args.output_dir}")
            save_model(model, tokenizer, training_args)

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
