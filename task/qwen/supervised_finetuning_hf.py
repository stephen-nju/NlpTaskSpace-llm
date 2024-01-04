import functools
import glob
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple, Union

import jieba
import numpy as np
import torch
import torch.nn as nn
import transformers
from datasets import load_dataset
from loguru import logger
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from peft import (
    LoraConfig,
    PeftModel,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from rouge_chinese import Rouge
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)
from transformers.deepspeed import is_deepspeed_zero3_enabled
from transformers.trainer_utils import get_last_checkpoint

from llm.src.constant import IGNORE_INDEX
from llm.src.datasets.preprocessing import preprocess_supervised_dataset_train
from llm.src.datasets.template import get_template_and_fix_tokenizer, register_template
from llm.src.utils import find_all_linear_names

if TYPE_CHECKING:
    from transformers.tokenization_utils import PreTrainedTokenizer
    from transformers.trainer import PredictionOutput


register_template(
    name="qwen",
    prefix=[{"token": "<|im_start|>"}, "system\n{{system}}"],
    prompt=[
        {"token": "<|im_start|>"},
        "user\n{{query}}",
        {"token": "<|im_end|>"},
        "\n",
        {"token": "<|im_start|>"},
        "assistant\n",
    ],
    system="You are a helpful assistant.",
    sep=[{"token": "<|im_end|>"}, "\n"],
    stop_words=["<|im_end|>"],
    efficient_eos=True,
)


class SupervisedFinetuningTrainer(Seq2SeqTrainer):
    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        r"""

        Removes the prompt part in the generated tokens.
        Subclass and override to inject custom behavior.

        """

        labels = inputs["labels"].detach().clone() if "labels" in inputs else None  # backup labels

        if self.args.predict_with_generate:
            assert self.tokenizer.padding_side == "left", "This method only accepts left-padded tensor."
            prompt_len, label_len = inputs["input_ids"].size(-1), inputs["labels"].size(-1)
            if prompt_len > label_len:
                inputs["labels"] = self._pad_tensors_to_target_len(inputs["labels"], inputs["input_ids"])
            if label_len > prompt_len:  # truncate the labels instead of padding the inputs (llama2 fp16 compatibility)
                inputs["labels"] = inputs["labels"][:, :prompt_len]
        loss, generated_tokens, _ = super().prediction_step(  # ignore the returned labels (may be truncated)
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
        )
        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, :prompt_len] = self.tokenizer.pad_token_id
            generated_tokens = generated_tokens.contiguous()
        return loss, generated_tokens, labels

    def _pad_tensors_to_target_len(self, src_tensor: torch.Tensor, tgt_tensor: torch.Tensor) -> torch.Tensor:
        r"""

        Pads the tensor to the same length as the target tensor.

        """

        assert self.tokenizer.pad_token_id is not None, "Pad token is required."
        padded_tensor = self.tokenizer.pad_token_id * torch.ones_like(tgt_tensor)
        padded_tensor[:, -src_tensor.shape[-1] :] = src_tensor  # adopt left-padding
        return padded_tensor.contiguous()  # in contiguous memory

    def save_predictions(self, predict_results: "PredictionOutput") -> None:
        r"""
        Saves model predictions to `output_dir`.
        A custom behavior that not contained in Seq2SeqTrainer.
        """

        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info(f"Saving prediction results to {output_prediction_file}")
        preds = np.where(
            predict_results.predictions != IGNORE_INDEX, predict_results.predictions, self.tokenizer.pad_token_id
        )
        labels = np.where(
            predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.tokenizer.pad_token_id
        )
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        decoded_labels = self.tokenizer.batch_decode(
            labels, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        with open(output_prediction_file, "w", encoding="utf-8") as writer:
            res: List[str] = []
            for pred, label in zip(decoded_preds, decoded_labels):
                res.append(json.dumps({"label": label, "predict": pred}, ensure_ascii=False))
            writer.write("\n".join(res))


@dataclass
class FinetuneArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    template_name: str = field(metadata={"help": "template name for construct prompt for model training or inference"})
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )

    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=False,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )

    resize_position_embeddings: Optional[bool] = field(
        default=None,
        metadata={
            "help": (
                "Whether to automatically resize the position embeddings if `max_source_length` exceeds "
                "the model's position embeddings."
            )
        },
    )

    quantization_bit: Optional[str] = field(default=None, metadata={"help": "quantization bit when loading model"})
    # data arguement
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    train_data: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a jsonlines or csv file)."}
    )
    dev_data: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "An optional input evaluation data file to evaluate the metrics (rouge) on (a jsonlines or csv file)."
            )
        },
    )
    test_data: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to evaluate the metrics (rouge) on (a jsonlines or csv file)."
        },
    )

    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={"help": "The percentage of the train set used as validation set in case there's no validation split"},
    )

    overwrite_cache: bool = field(default=False, metadata={"help": "Overwrite the cached training and evaluation sets"})

    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": (
                "The maximum total sequence length for target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total sequence length for validation target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
                "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
                "during ``evaluate`` and ``predict``."
            )
        },
    )

    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to model maximum sentence length. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                "efficient on GPU but very bad for TPU."
            )
        },
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
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                "which is used during ``evaluate`` and ``predict``."
            )
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )

    forced_bos_token: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The token to force as the first generated token after the decoder_start_token_id."
                "Useful for multilingual models like mBART where the first generated token"
                "needs to be the target language token (Usually it is the target language token)"
            )
        },
    )

    use_peft: bool = field(default=True, metadata={"help": "Whether to using peft for tuning"})
    lora_rank: Optional[int] = field(default=8, metadata={"help": "The intrinsic dimension for LoRA fine-tuning."})
    lora_alpha: Optional[float] = field(
        default=32.0, metadata={"help": "The scale factor for LoRA fine-tuning (similar with the learning rate)."}
    )
    lora_dropout: Optional[float] = field(default=0.1, metadata={"help": "Dropout rate for the LoRA fine-tuning."})
    lora_target: Optional[str] = field(
        default=None,
        metadata={"help": "Name(s) of target modules to apply LoRA. Use commas to separate multiple modules."},
    )
    peft_ckpt_path: Optional[str] = field(
        default=None,
        metadata={"help": "lora checkpoint path for evaluate"},
    )

    def __post_init__(self):
        if isinstance(self.lora_target, str):  # support custom target modules/layers of LoRAk
            self.lora_target = [target.strip() for target in self.lora_target.split(",")]

        if self.dataset_name is None and self.train_data is None and self.dev_data is None and self.test_data is None:
            raise ValueError("Need either a dataset name or a training/validation/test file.")
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length


def save_model(model, tokenizer, args):
    """Save the model and the tokenizer."""
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Take care of distributed/parallel training
    model_to_save = model.module if hasattr(model, "module") else model
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def save_model_zero3(model, tokenizer, args, trainer):
    """Save the model for deepspeed zero3.
    refer https://github.com/lm-sys/FastChat/blob/main/fastchat/train/train_lora.py#L209
    """
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    state_dict_zero3 = trainer.model_wrapped._zero3_consolidated_16bit_state_dict()
    model_to_save = model.module if hasattr(model, "module") else model
    model_to_save.save_pretrained(args.output_dir, state_dict=state_dict_zero3)
    tokenizer.save_pretrained(output_dir)


def main():
    finetune_args, training_args = HfArgumentParser(
        (FinetuneArguments, Seq2SeqTrainingArguments)
    ).parse_args_into_dataclasses()

    #############logging prepare ########################

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()

    # logger.setLevel(log_level)
    # datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")
    # 参数合规性检查,参考LLaMA-Efficient-Tuning
    if training_args.do_train and training_args.predict_with_generate:
        raise ValueError("`predict_with_generate` cannot be set as True while training.")

    #######检测checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    set_seed(training_args.seed)

    ####### prepare model ############

    config = AutoConfig.from_pretrained(finetune_args.model_name_or_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(
        finetune_args.model_name_or_path,
        use_fast=finetune_args.use_fast_tokenizer,
        trust_remote_code=True,
    )
    if finetune_args.quantization_bit is not None:
        print(f"Quantized to {finetune_args.quantization_bit}")
        if finetune_args.quantization_bit == "4bit":
            # load_in_4bit = (True,)
            quantization_config = BitsAndBytesConfig(
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif finetune_args.quantization_bit == "8bit":
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        else:
            raise ValueError("unsupport quantization_bit")

        model = AutoModelForCausalLM.from_pretrained(
            finetune_args.model_name_or_path,
            from_tf=bool(".ckpt" in finetune_args.model_name_or_path),
            config=config,
            quantization_config=quantization_config,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
    else:
        print(f"model config===\n{config}")
        model = AutoModelForCausalLM.from_pretrained(
            finetune_args.model_name_or_path,
            from_tf=bool(".ckpt" in finetune_args.model_name_or_path),
            config=config,
            trust_remote_code=True,
            torch_dtype="auto",
        )
        # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
        # on a small vocab and want a smaller embedding size, remove this test.

    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

        # model.supports_gradient_checkpointing = True  #

    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!the datasets

    # 分布式训练
    if torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    if finetune_args.quantization_bit is not None:
        # 启用模型量化需要开启
        model = prepare_model_for_kbit_training(model)
    # 获取数据处理的模板
    template = get_template_and_fix_tokenizer(finetune_args.template_name, tokenizer)

    if finetune_args.use_peft:
        if finetune_args.peft_ckpt_path:
            peft_model = PeftModel.from_pretrained(model, finetune_args.peft_ckpt_path, is_trainable=True)
            model = peft_model.merge_and_unload()
        else:
            logger.info(">>>init new lora model")
            if len(finetune_args.lora_target) == 1 and finetune_args.lora_target[0] == "all":
                lora_target = find_all_linear_names(model, finetune_args.quantization_bit)
            else:
                lora_target = finetune_args.lora_target

            if len(lora_target) > 0:
                peft_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    target_modules=lora_target,
                    inference_mode=False,
                    r=finetune_args.lora_rank,
                    lora_alpha=finetune_args.lora_alpha,
                    lora_dropout=finetune_args.lora_dropout,
                )
            else:
                raise ValueError("lora target should not be none")
            # adalora 会出现 https://github.com/huggingface/peft/issues/479
            model = get_peft_model(model, peft_config)

        # 分布式训练
        model.is_parallelizable = True
        model.model_parallel = True
        model.print_trainable_parameters()

        # tokenizer.pad_token_id = model.generation_config.pad_token_id
        # if training_args.predict_with_generate:
        #     tokenizer.padding_side = "left"

    ############# prepare data ###########

    data_files = {}
    dataset_args = {}
    if finetune_args.train_data is not None:
        train_dof = [s.strip() for s in finetune_args.train_data.strip().split(",")]
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
            raise ValueError(f"train data should be valid file path or direcotory {dof}")
    data_files["train"] = train_files

    if finetune_args.dev_data is not None:
        dev_dof = [s.strip() for s in finetune_args.dev_data.split(",")]
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
            dataset_args["keep_linebreaks"] = finetune_args.keep_linebreaks

    raw_datasets = load_dataset(extension, data_files=data_files, **dataset_args)

    if "validation" not in raw_datasets.keys():
        raw_datasets["validation"] = load_dataset(
            extension,
            data_files=data_files,
            split=f"train[:{finetune_args.validation_split_percentage}%]",
            **dataset_args,
        )

        raw_datasets["train"] = load_dataset(
            extension,
            data_files=data_files,
            split=f"train[{finetune_args.validation_split_percentage}%:]",
            **dataset_args,
        )

    if training_args.do_train:
        column_names = list(raw_datasets["train"].features)

    else:
        column_names = list(raw_datasets["validation"].features)
    preprocessing_function_train = functools.partial(
        preprocess_supervised_dataset_train,
        tokenizer=tokenizer,
        template=template,
        max_source_length=finetune_args.max_source_length,
        max_target_length=finetune_args.max_target_length,
    )
    with training_args.main_process_first(desc="Dataset tokenization and grouping"):
        if not finetune_args.streaming:
            lm_datasets = raw_datasets.map(
                preprocess_supervised_dataset_train,
                batched=True,
                num_proc=finetune_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not finetune_args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )
        else:
            lm_datasets = raw_datasets.map(
                preprocessing_function_train,
                batched=True,
                remove_columns=column_names,
            )
    train_dataset = None
    max_train_samples = 0
    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = lm_datasets["train"]
        max_train_samples = len(train_dataset)
        if finetune_args.max_train_samples is not None and finetune_args.max_train_samples > 0:
            max_train_samples = min(len(train_dataset), finetune_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

        logger.debug(f"Num train_samples: {len(train_dataset)}")
        logger.debug("Tokenized training example:")
        logger.debug(tokenizer.decode(train_dataset[0]["input_ids"]))
    eval_dataset = None
    max_eval_samples = 0
    if training_args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = lm_datasets["validation"]
        max_eval_samples = len(eval_dataset)
        if finetune_args.max_eval_samples is not None and finetune_args.max_eval_samples > 0:
            max_eval_samples = min(len(eval_dataset), finetune_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        logger.debug(f"Num eval_samples: {len(eval_dataset)}")
        logger.debug("Tokenized eval example:")
        logger.debug(tokenizer.decode(eval_dataset[0]["input_ids"]))

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        if finetune_args.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        score_dict = {"rouge-1": [], "rouge-2": [], "rouge-l": [], "bleu-4": []}
        for pred, label in zip(decoded_preds, decoded_labels):
            hypothesis = list(jieba.cut(pred))
            reference = list(jieba.cut(label))
            rouge = Rouge()
            scores = rouge.get_scores(" ".join(hypothesis), " ".join(reference))
            result = scores[0]

            for k, v in result.items():
                score_dict[k].append(round(v["f"] * 100, 4))
            bleu_score = sentence_bleu([list(label)], list(pred), smoothing_function=SmoothingFunction().method3)
            score_dict["bleu-4"].append(round(bleu_score * 100, 4))

        for k, v in score_dict.items():
            score_dict[k] = float(np.mean(v))
        return score_dict

    # 重新更新生成的相关参数
    # training_args.generation_max_length = (
    #     training_args.generation_max_length
    #     if training_args.generation_max_length is not None
    #     else finetune_args.val_max_target_length
    # )
    # training_args.generation_num_beams = (
    #     finetune_args.num_beams if finetune_args.num_beams is not None else training_args.generation_num_beams
    # )

    trainer = SupervisedFinetuningTrainer(
        model=model,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        args=training_args,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer,
            label_pad_token_id=IGNORE_INDEX if finetune_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
            pad_to_multiple_of=8,
            return_tensors="pt",
            padding=True,
        ),
        compute_metrics=compute_metrics if training_args.do_eval and training_args.predict_with_generate else None,
    )

    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint

        train_output = trainer.train(resume_from_checkpoint=checkpoint)

        metrics = train_output.metrics
        max_train_samples = (
            finetune_args.max_train_samples if finetune_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        try:
            tokenizer.save_pretrained(training_args.output_dir)
        except:
            logger.warning("Cannot save tokenizer, please copy the files manually.")

        if trainer.is_world_process_zero():
            logger.info(f"Saving model checkpoint to {training_args.output_dir}")
            if is_deepspeed_zero3_enabled():
                save_model_zero3(model, tokenizer, training_args, trainer)
            else:
                save_model(model, tokenizer, training_args)

    # Evaluation
    if training_args.do_eval:
        gen_kwarg = model.generation_config.to_dict()
        logger.info("*** Evaluate ***")
        # 可以自定义解码策略
        # print(gen_kwarg)
        metrics = trainer.evaluate(metric_key_prefix="eval", **gen_kwarg)
        if training_args.predict_with_generate:  # eval_loss will be wrong if predict_with_generate is enabled
            metrics.pop("eval_loss", None)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()
