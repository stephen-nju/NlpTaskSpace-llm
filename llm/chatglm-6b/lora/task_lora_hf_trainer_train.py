# -----*---coding=utf-8---**---
"""
采用huggingface官方代码
"""
import sys
import torch
import torch.nn as nn
import datasets
import os
import logging
from dataclasses import dataclass, field
from typing import Optional
import transformers

from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    HfArgumentParser,
    DataCollatorForSeq2Seq,
    AutoTokenizer,
    AutoConfig,
    AutoModel,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset

logger = logging.getLogger(__name__)

# 模型参数


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    ptuning_checkpoint: str = field(
        default=None, metadata={"help": "Path to p-tuning v2 checkpoints"}
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where to store the pretrained models downloaded from huggingface.co"
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
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
    quantization_bit: Optional[int] = field(default=None)
    pre_seq_len: Optional[int] = field(default=None)
    prefix_projection: bool = field(default=False)


# 数据参数


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    lang: Optional[str] = field(
        default=None, metadata={"help": "Language id for summarization."}
    )

    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        },
    )
    prompt_column: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of the column in the datasets containing the full texts (for summarization)."
        },
    )
    response_column: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of the column in the datasets containing the summaries (for summarization)."
        },
    )
    history_column: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of the column in the datasets containing the history of chat."
        },
    )
    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "The input training data file (a jsonlines or csv file)."},
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "An optional input evaluation data file to evaluate the metrics (rouge) on (a jsonlines or csv file)."
            )
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to evaluate the metrics (rouge) on (a jsonlines or csv file)."
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
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
    source_prefix: Optional[str] = field(
        default="",
        metadata={
            "help": "A prefix to add before every source text (useful for T5 models)."
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

    def __post_init__(self):
        if (
            self.dataset_name is None
            and self.train_file is None
            and self.validation_file is None
            and self.test_file is None
        ):
            raise ValueError(
                "Need either a dataset name or a training/validation/test file."
            )
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in [
                    "csv",
                    "json",
                ], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in [
                    "csv",
                    "json",
                ], "`validation_file` should be a csv or a json file."
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.abs
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    last_checkpoint = None

    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
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

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # 加载数据集
    data_files = {}
    if data_args.train_file is not None:
        data_files["train"] = data_args.train_file

    if data_args.validation_file is not None:
        data_files["validation"] = data_args.validation_file

    # __import__('pdb').set_trace()
    raw_datasets = load_dataset(
        "json",
        data_files=data_files,
        cache_dir=model_args.cache_dir,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    elif training_args.do_eval:
        column_names = raw_datasets["validation"].column_names
    elif training_args.do_predict:
        column_names = raw_datasets["test"].column_names
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return

    # Get the column names for input/target.
    prompt_column = data_args.prompt_column
    response_column = data_args.response_column
    history_column = data_args.history_column


    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=True
    )
    ##### 自定义配置
    config.pre_seq_len = model_args.pre_seq_len
    config.prefix_projection = model_args.prefix_projection
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=True
    )
    # 加载base的基础模型
    model = AutoModel.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=True
    )

    # 加载lora配置
    lora_config = LoraConfig(
        r=8,  # lora矩阵的秩
        lora_alpha=16,  # 归一化超参,具体涉及的计算位置 #TODO
        lora_dropout=0.05,
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    
    # 采用半精度训练
    model = model.half()
    
    if model_args.quantization_bit is not None:
        print(f"Quantized to {model_args.quantization_bit} bit")
        model = model.quantize(model_args.quantization_bit)

    
    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""
    # 数据处理
    def preprocessing_function_train(examples):
        max_seq_length = data_args.max_source_length + data_args.max_target_length
        model_inputs = {
            "input_ids": [],
            "labels": [],
        }
        # __import__('pdb').set_trace()
        for i in range(len(examples[prompt_column])):
            if examples[prompt_column][i] and examples[response_column][i]:
                query, answer = examples[prompt_column][i], examples[response_column][i]

                if history_column is None:
                    prompt = query
                else:
                    prompt = ""
                    history = examples[history_column][i]
                    for turn_idx, (old_query, response) in enumerate(history):
                        prompt += "[Round {}]\n问：{}\n答：{}\n".format(
                            turn_idx, old_query, response
                        )
                    prompt += "[Round {}]\n问：{}\n答：".format(len(history), query)

                prompt = prefix + prompt
                a_ids = tokenizer.encode(text=prompt, add_special_tokens=False)
                b_ids = tokenizer.encode(text=answer, add_special_tokens=False)

                if len(a_ids) > data_args.max_source_length - 1:
                    a_ids = a_ids[: data_args.max_source_length - 1]

                if len(b_ids) > data_args.max_target_length - 2:
                    b_ids = b_ids[: data_args.max_target_length - 2]

                input_ids = tokenizer.build_inputs_with_special_tokens(a_ids, b_ids)

                context_length = input_ids.index(tokenizer.bos_token_id)
                mask_position = context_length - 1
                labels = [-100] * context_length + input_ids[mask_position + 1 :]

                pad_len = max_seq_length - len(input_ids)
                input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
                labels = labels + [tokenizer.pad_token_id] * pad_len
                if data_args.ignore_pad_token_for_loss:
                    labels = [
                        (l if l != tokenizer.pad_token_id else -100) for l in labels
                    ]

                model_inputs["input_ids"].append(input_ids)
                model_inputs["labels"].append(labels)

        return model_inputs

    def preprocess_function_eval(examples):
        inputs, targets = [], []
        for i in range(len(examples[prompt_column])):
            if examples[prompt_column][i] and examples[response_column][i]:
                query = examples[prompt_column][i]
                if history_column is None or len(examples[history_column][i]) == 0:
                    prompt = query
                else:
                    prompt = ""
                    history = examples[history_column][i]
                    for turn_idx, (old_query, response) in enumerate(history):
                        prompt += "[Round {}]\n问：{}\n答：{}\n".format(
                            turn_idx, old_query, response
                        )
                    prompt += "[Round {}]\n问：{}\n答：".format(len(history), query)
                inputs.append(prompt)
                targets.append(examples[response_column][i])

        inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(
            inputs,
            max_length=data_args.max_source_length,
            truncation=True,
            padding=True,
        )
        labels = tokenizer(
            text_target=targets, max_length=max_target_length, truncation=True
        )

        if data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label]
                for label in labels["input_ids"]
            ]
        model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    # __import__("pdb").set_trace()
    # 模型训练
    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                preprocessing_function_train,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )

    if training_args.do_eval:
        max_target_length = data_args.val_max_target_length
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        with training_args.main_process_first(
            desc="validation dataset map pre-processing"
        ):
            eval_dataset = eval_dataset.map(
                preprocess_function_eval,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )
    # Data collator
    label_pad_token_id = (
        -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    )
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=None,
        padding=False,
    )

    # Metric
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        if data_args.ignore_pad_token_for_loss:
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
            bleu_score = sentence_bleu(
                [list(label)],
                list(pred),
                smoothing_function=SmoothingFunction().method3,
            )
            score_dict["bleu-4"].append(round(bleu_score * 100, 4))

        for k, v in score_dict.items():
            score_dict[k] = float(np.mean(v))
        return score_dict

    # 重新配置Seq2SeqTrainer的参数
    training_args.generation_max_length = (
        training_args.generation_max_length
        if training_args.generation_max_length is not None
        else data_args.val_max_target_length
    )
    training_args.generation_num_beams = (
        data_args.num_beams
        if data_args.num_beams is not None
        else training_args.generation_num_beams
    )

    # Initialize our Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint

        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        # trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples
            if data_args.max_train_samples is not None
            else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(
            metric_key_prefix="eval",
            do_sample=True,
            top_p=0.7,
            max_length=512,
            temperature=0.95,
        )
        max_eval_samples = (
            data_args.max_eval_samples
            if data_args.max_eval_samples is not None
            else len(eval_dataset)
        )
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")

        predict_results = trainer.predict(
            predict_dataset,
            metric_key_prefix="predict",
            max_length=512,
            do_sample=True,
            top_p=0.7,
            temperature=0.95,
        )
        metrics = predict_results.metrics
        max_predict_samples = (
            data_args.max_predict_samples
            if data_args.max_predict_samples is not None
            else len(predict_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        if trainer.is_world_process_zero():
            if training_args.predict_with_generate:
                predictions = tokenizer.batch_decode(
                    predict_results.predictions,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )
                predictions = [pred.strip() for pred in predictions]
                labels = tokenizer.batch_decode(
                    predict_results.label_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )
                labels = [label.strip() for label in labels]
                output_prediction_file = os.path.join(
                    training_args.output_dir, "generated_predictions.txt"
                )
                with open(output_prediction_file, "w", encoding="utf-8") as writer:
                    for p, l in zip(predictions, labels):
                        res = json.dumps(
                            {"labels": l, "predict": p}, ensure_ascii=False
                        )
                        writer.write(f"{res}\n")
    return results

    def postprocessing_function_train():
        pass


if __name__ == "__main__":
    main()
