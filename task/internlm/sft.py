# ----*----coding:utf8-----*----

import argparse
import functools
import os
from typing import Any, Dict

import jieba
import numpy as np
import torch
from datasets import load_dataset
from deepspeed.ops.adam import DeepSpeedCPUAdam
from lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.strategies import DeepSpeedStrategy
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from rouge_chinese import Rouge
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    GenerationConfig,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)

from llm.src.constant import IGNORE_INDEX
from llm.src.datasets.preprocessing import preprocess_supervised_dataset
from llm.src.datasets.template import get_template_and_fix_tokenizer


class SupervisedFintuningModule(LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        # 加载模型
        self.config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            use_fast=not args.use_slow_tokenizer,
            trust_remote_code=True,
        )
        self.template = get_template_and_fix_tokenizer(self.args.template_name, self.tokenizer)
        if self.args.quantization_bit is not None:
            print(f"Quantized to {self.args.quantization_bit}")
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
            model = AutoModelForCausalLM.from_pretrained(
                self.args.model_name_or_path,
                from_tf=bool(".ckpt" in self.args.model_name_or_path),
                config=self.config,
                trust_remote_code=True,
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

        if self.args.use_lora:
            if isinstance(self.args.lora_target, str):  # support custom target modules/layers of LoRA
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
        self.model.print_trainable_parameters()
        self.save_hyperparameters()

    def setup(self, stage):
        raw_datasets = load_dataset("json", data_files={"train": self.args.train_data, "dev": self.args.dev_data})
        preprocessing_function_train = functools.partial(
            preprocess_supervised_dataset,
            tokenizer=self.tokenizer,
            template=self.template,
            max_source_length=self.args.max_source_length,
            max_target_length=self.args.max_target_length,
        )
        column_names = raw_datasets["train"].column_names
        self.train_dataset = raw_datasets["train"].map(
            preprocessing_function_train,
            batched=True,
            num_proc=self.args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not self.args.overwrite_cache,
            desc="Running tokenizer on train dataset",
        )
        # self.dev_dataset = raw_datasets["dev"].map(
        #         preprocessing_function_train,
        #         batched=True,
        #         num_proc=finetune_args.preprocessing_num_workers,
        #         remove_columns=column_names,
        #         load_from_cache_file=not finetune_args.overwrite_cache,
        #         desc="Running tokenizer on train dataset",
        #     )

    @property
    def deepspeed_offload(self) -> bool:
        strategy = self.trainer.strategy
        if isinstance(strategy, DeepSpeedStrategy):
            config = strategy.config["zero_optimization"]
            return config.get("offload_optimizer") or config.get("offload_param")
        return False

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
        num_gpus = self.trainer.num_devices
        # 注：只有在使用pytorch Lightning的LightningDataModule 时候才可以使用该方式回去训练集大小
        print(f"len train dataloader==={len(self.train_dataloader())}")
        t_total = (
            len(self.train_dataloader()) // (self.trainer.accumulate_grad_batches * num_gpus) + 1
        ) * self.args.max_epochs
        warmup_steps = int(self.args.warmup_proportion * t_total)

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
            save_path = os.path.join(self.args.output_dir, "hf_model")
            self.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)

    # def validation_step(self, batch, batch_idx):
    #     output = self.model(**batch)
    #     self.log("eval_loss", output.loss, on_step=True, on_epoch=True)

    # def test_step(self, batch, batch_idx):
    #     pass
    # output = self.model.generate(batch["input_ids"], max_new_tokens=128)
    # output_text = self.tokenizer.decode(
    #     output.sequence[0], skip_special_tokens=True, clean_up_tokenization_spaces=False
    # )
    # 计算其它指标

    def compute_metrics(self, eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        if self.args.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

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

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            num_workers=4,
            pin_memory=True,
            collate_fn=DataCollatorForSeq2Seq(
                self.tokenizer,
                label_pad_token_id=IGNORE_INDEX if self.args.ignore_pad_token_for_loss else self.tokenizer.pad_token_id,
                pad_to_multiple_of=8,
                return_tensors="pt",
                padding=True,
            ),
        )

    def training_step(self, batch, batch_idx):
        output = self.model(**batch)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], on_step=True, sync_dist=True)
        self.log("train_loss", output.loss, on_step=True, prog_bar=True, sync_dist=True)
        return output.loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train llm model")
    parser.add_argument("--output_dir", type=str, default="./output_dir/", help="")
    parser.add_argument("--train_data", type=str, default="", help="train data path")
    parser.add_argument("--test_data", type=str, default="", help="test data path")
    parser.add_argument("--dev_data", type=str, default="", help="dev data path")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--template_name",
        type=str,
        help="template name",
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
        help='Baichuan choices: ["W_pack", "o_proj", "gate_proj", "up_proj", "down_proj"]',
    )
    parser.add_argument("--lora_ckpt_path", type=str, default=None, help="")
    arg = parser.parse_args()

    if arg.seed is not None:
        seed_everything(arg.seed)

    # 先加载模型，再加载数据
    model = SupervisedFintuningModule(arg)

    if arg.deepspeed:
        strategy = DeepSpeedStrategy(config=arg.deepspeed)
    else:
        strategy = "auto"

    # 添加常用的checkpoint
    callbacks = []
    checkpoint = ModelCheckpoint(dirpath=arg.output_dir, every_n_train_steps=arg.save_steps, save_last=True)
    callbacks.append(checkpoint)

    trainer = Trainer(
        devices="auto",
        max_epochs=arg.max_epochs,
        strategy=strategy,
        callbacks=callbacks,
        log_every_n_steps=1,
        default_root_dir=arg.output_dir,
    )

    trainer.fit(model)
