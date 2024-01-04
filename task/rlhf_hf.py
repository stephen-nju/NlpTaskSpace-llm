import math
import os
import sys
from dataclasses import asdict, dataclass, field
from glob import glob
from types import MethodType
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple

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
from tqdm import tqdm
from transformers import (
    PREFIX_CHECKPOINT_DIR,
    SAFE_WEIGHTS_NAME,
    WEIGHTS_NAME,
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BloomForCausalLM,
    BloomTokenizerFast,
    GenerationConfig,
    HfArgumentParser,
    InfNanRemoveLogitsProcessor,
    LlamaForCausalLM,
    LlamaTokenizer,
    LogitsProcessorList,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
    cached_file,
)
from trl import (
    AutoModelForCausalLMWithValueHead,
    PPOConfig,
    PPODecorators,
    PPOTrainer,
    logprobs_from_logits,
    set_seed,
)

from llm.src.datasets.template import get_template_and_fix_tokenizer
from llm.src.utils import find_all_linear_names, print_trainable_parameters

if TYPE_CHECKING:
    from transformers import PreTrainedModel


os.environ["TOKENIZERS_PARALLELISM"] = "FALSE"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

MODEL_CLASSES = {
    "bloom": (AutoConfig, BloomForCausalLM, BloomTokenizerFast),
    "chatglm": (AutoConfig, AutoModel, AutoTokenizer),
    "llama": (AutoConfig, LlamaForCausalLM, LlamaTokenizer),
    "baichuan": (AutoConfig, AutoModelForCausalLM, AutoTokenizer),
    "auto": (AutoConfig, AutoModelForCausalLM, AutoTokenizer),
}


def get_logits_processor() -> "LogitsProcessorList":
    r"""
    Gets logits processor that removes NaN and Inf logits.
    """
    logits_processor = LogitsProcessorList()
    logits_processor.append(InfNanRemoveLogitsProcessor())
    return logits_processor


def load_valuehead_params(path_or_repo_id: str, model_args) -> Dict[str, torch.Tensor]:
    r"""
    Loads value head parameters from Hugging Face Hub or local disk.

    Returns: dict with keys `v_head.summary.weight` and `v_head.summary.bias`.
    """
    kwargs = {"path_or_repo_id": path_or_repo_id, "cache_dir": model_args.cache_dir, "token": model_args.hf_hub_token}

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


def replace_model(model: "AutoModelForCausalLMWithValueHead", target: Literal["default", "reward"]) -> None:
    if target == "reward":  # save default head temporarily
        valuehead_state_dict: Dict[str, torch.Tensor] = model.v_head.state_dict()
        setattr(model, "default_head_weight", valuehead_state_dict["summary.weight"].detach().clone())
        setattr(model, "default_head_bias", valuehead_state_dict["summary.bias"].detach().clone())

    model.pretrained_model.set_adapter(target)  # set the LoRA adapter to be active
    model.v_head.load_state_dict(
        {
            "summary.weight": model.get_buffer("{}_head_weight".format(target)).detach().clone(),
            "summary.bias": model.get_buffer("{}_head_bias".format(target)).detach().clone(),
        }
    )


def dump_layernorm(model: "PreTrainedModel") -> Dict[str, torch.Tensor]:
    layer_norm_params = {}
    for name, param in model.named_parameters():
        if param.data.dtype == torch.float32:
            layer_norm_params[name] = param.data.detach().clone()
            param.data = param.data.to(model.config.torch_dtype)

        return layer_norm_params


def restore_layernorm(model: "PreTrainedModel", layernorm_params: Optional[Dict[str, torch.Tensor]] = None) -> None:
    for name, param in model.named_parameters():
        if name in layernorm_params:
            param.data = layernorm_params[name]


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


@dataclass
class ScriptArguments:

    """
    The name of the Casual LM model we wish to fine with PPO
    """

    # Model arguments
    model_type: str = field(
        default=None, metadata={"help": "Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys())}
    )
    model_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "The model checkpoint for weights initialization."}
    )
    reward_model_name_or_path: Optional[str] = field(default=None, metadata={"help": "The reward model name"})
    reward_model_device: Optional[str] = field(default="cuda:0", metadata={"help": "The reward model device"})
    tokenizer_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "The tokenizer for weights initialization."}
    )

    quantization_bit: Optional[str] = field(
        default=None, metadata={"help": "quantization bit when loading model", "choices": ["4bit", "8bit"]}
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
    # Dataset arguments
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
    template_name: Optional[str] = field(default="vicuna", metadata={"help": "The template name."})
    max_source_length: Optional[int] = field(default=256, metadata={"help": "Max length of prompt input text"})
    max_target_length: Optional[int] = field(default=256, metadata={"help": "Max length of output text"})
    min_target_length: Optional[int] = field(default=4, metadata={"help": "Min length of output text"})
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
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )

    # Training arguments
    use_peft: bool = field(default=True, metadata={"help": "Whether to use peft"})
    target_modules: Optional[str] = field(default=None)
    lora_rank: Optional[int] = field(default=8)
    lora_dropout: Optional[float] = field(default=0.05)
    lora_alpha: Optional[float] = field(default=32.0)
    modules_to_save: Optional[str] = field(default=None)
    peft_ckpt_path: Optional[str] = field(default=None)

    def __post_init__(self):
        if self.model_type is None:
            raise ValueError("You must specify a valid model_type to run training.")

        if self.model_name_or_path is None:
            raise ValueError("You must specify a valid model_name_or_path to run training.")

        if self.reward_model_name_or_path is None:
            raise ValueError("You must specify a valid reward_model_name_or_path to run training.")

        if self.max_source_length < 60:
            raise ValueError("You must specify a valid max_source_length >= 60 to run training")


class AverageMeter:
    r"""
    Computes and stores the average and current value.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


@dataclass
class RLHFTraingArguemnts(TrainingArguments):
    """
    需要添加ppo参数和模型生成模型的解码参数
    """

    ppo_epochs: Optional[int] = field(default=4, metadata={"help": "the number of ppo epochs"})
    batch_size: Optional[int] = field(default=8, metadata={"help": "Batch size"})
    mini_batch_size: Optional[int] = field(default=1, metadata={"help": "PPO minibatch size"})
    target_kl: Optional[float] = field(default=0.1, metadata={"help": "The kl target for early stopping"})
    reward_baseline: Optional[float] = field(
        default=0.0,
        metadata={"help": "Baseline value that is subtracted from the reward"},
    )
    init_kl_coef: Optional[float] = field(
        default=0.2,
        metadata={"help": "Initial KL penalty coefficient (used for adaptive and linear control)"},
    )
    adap_kl_ctrl: Optional[bool] = field(default=True, metadata={"help": "Use adaptive KL control, otherwise linear"})
    learning_rate: Optional[float] = field(default=1.5e-5, metadata={"help": "Learning rate"})
    gradient_accumulation_steps: Optional[int] = field(
        default=1, metadata={"help": "the number of gradient accumulation steps"}
    )
    save_steps: Optional[int] = field(default=50, metadata={"help": "X steps to save the model"})
    output_dir: Optional[str] = field(default="outputs-rl", metadata={"help": "The output directory"})
    seed: Optional[int] = field(default=0, metadata={"help": "Seed"})
    max_steps: Optional[int] = field(default=200, metadata={"help": "Number of steps to train"})
    report_to: Optional[str] = field(default="tensorboard", metadata={"help": "Report to wandb or tensorboard"})

    do_sample: Optional[bool] = field(
        default=True, metadata={"help": "Whether or not to use sampling, use greedy decoding otherwise."}
    )
    temperature: Optional[float] = field(
        default=0.95, metadata={"help": "The value used to modulate the next token probabilities."}
    )
    top_p: Optional[float] = field(
        default=0.7,
        metadata={
            "help": "The smallest set of most probable tokens with probabilities that add up to top_p or higher are kept."
        },
    )
    top_k: Optional[int] = field(
        default=50,
        metadata={"help": "The number of highest probability vocabulary tokens to keep for top-k filtering."},
    )
    num_beams: Optional[int] = field(
        default=1, metadata={"help": "Number of beams for beam search. 1 means no beam search."}
    )
    max_length: Optional[int] = field(
        default=512,
        metadata={"help": "The maximum length the generated tokens can have. It can be overridden by max_new_tokens."},
    )
    max_new_tokens: Optional[int] = field(
        default=512,
        metadata={"help": "The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt."},
    )
    repetition_penalty: Optional[float] = field(
        default=1.0, metadata={"help": "The parameter for repetition penalty. 1.0 means no penalty."}
    )
    length_penalty: Optional[float] = field(
        default=1.0, metadata={"help": "Exponential penalty to the length that is used with beam-based generation."}
    )

    def to_dict(self) -> Dict[str, Any]:
        args = asdict(self)
        if args.get("max_new_tokens", -1) > 0:
            args.pop("max_length", None)
        else:
            args.pop("max_new_tokens", None)
        return args


class RLHFTrianer(PPOTrainer, Trainer):
    """
    继承PPOTrainer的属性和Trainer的方法,注意可能会出现异常
    copy from https://github.com/hiyouga/LLaMA-Factory/blob/main/src/llmtuner/train/ppo/trainer.py
    """

    def __init__(
        self,
        args: RLHFTraingArguemnts = None,
        reward_model: AutoModelForCausalLMWithValueHead = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        **kwargs,
    ):
        # 不需要初始化trainer的属性，如果有需要，按需自己初始化
        PPOTrainer.__init__(**kwargs)
        if args is None:
            output_dir = "tmp_trainer"
            logger.info(f"No `TrainingArguments` passed, using `output_dir={output_dir}`.")
            args = TrainingArguments(output_dir=output_dir)

        self.args = args
        self.tokenizer = tokenizer
        self.reward_model = reward_model
        self.state = TrainerState(
            is_local_process_zero=self.is_local_process_zero(),
            is_world_process_zero=self.is_world_process_zero(),
        )
        self.control = TrainerControl()

    def ppo_train(self, resume_from_checkpoint: Optional[str] = None) -> None:
        r"""
        Implements training loop for the PPO stage, like _inner_training_loop() in Huggingface's Trainer.
        """
        if resume_from_checkpoint is not None:
            raise ValueError("`resume_from_checkpoint` will be supported in the future version.")

        total_train_batch_size = (
            self.args.per_device_train_batch_size
            * self.args.gradient_accumulation_steps
            * self.srcipt_args.ppo_buffer_size
            * self.args.world_size
        )
        if self.args.max_steps > 0:
            num_examples = total_train_batch_size * self.args.max_steps
            num_train_epochs = sys.maxsize
            max_steps = self.args.max_steps
            steps_in_epoch = self.args.max_steps

        else:
            len_dataloader = len(self.dataloader)
            num_examples = len(self.dataset)
            num_train_epochs = self.args.num_train_epochs
            max_steps = math.ceil(num_train_epochs * len_dataloader)
            steps_in_epoch = len_dataloader

        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        if self.is_world_process_zero():
            logger.info("***** Running training *****")
            logger.info("  Num examples = {}".format(num_examples))
            logger.info("  Num Epochs = {}".format(num_train_epochs))
            logger.info("  Instantaneous batch size per device = {}".format(self.args.per_device_train_batch_size))
            logger.info(
                "  Total train batch size (w. parallel, buffer, distributed & accumulation) = {}".format(
                    total_train_batch_size
                )
            )
            logger.info("  Gradient Accumulation steps = {}".format(self.args.gradient_accumulation_steps))
            logger.info("  Num optimization epochs per batch = {}".format(self.srcipt_args.ppo_epochs))
            logger.info("  Total training steps = {}".format(max_steps))

        unwrapped_model: "AutoModelForCausalLMWithValueHead" = self.accelerator.unwrap_model(self.model)

        dataiter = iter(self.dataloader)
        loss_meter = AverageMeter()
        reward_meter = AverageMeter()
        self.log_callback.on_train_begin(self.args, self.state, self.control)
        for step in tqdm(range(max_steps), disable=not self.is_local_process_zero()):
            try:
                batch = next(dataiter)
            except StopIteration:
                dataiter = iter(self.dataloader)
                batch = next(dataiter)
            # Cast to inference mode
            unwrapped_model.gradient_checkpointing_disable()
            unwrapped_model.config.use_cache = True
            self.model.eval()
            # Get inputs
            self.tokenizer.padding_side = "right"  # change padding side
            queries, responses, rewards = [], [], []
            for idx in range(0, self.config.batch_size, self.config.mini_batch_size):
                mini_batch_queries, mini_batch_responses = self.get_inputs(
                    batch[idx : idx + self.config.mini_batch_size]
                )
                mini_batch_rewards = self.get_rewards(mini_batch_queries, mini_batch_responses, unwrapped_model)
                queries.extend(mini_batch_queries)
                responses.extend(mini_batch_responses)
                rewards.extend(mini_batch_rewards)

            # Cast to training mode
            unwrapped_model.gradient_checkpointing_enable()
            unwrapped_model.config.use_cache = False
            self.model.train()
            # Run PPO step

            stats = self.step(queries, responses, rewards)
            self.tokenizer.padding_side = "left"  # restore padding side
            loss_meter.update(float(stats["ppo/loss/total"]), n=len(rewards))
            reward_meter.update(torch.stack(rewards).mean().item(), n=len(rewards))
            if self.config.log_with is not None:
                try:
                    batch["query"] = self.tokenizer.batch_decode(queries, skip_special_tokens=True)
                    batch["response"] = self.tokenizer.batch_decode(responses, skip_special_tokens=True)
                    self.log_stats(stats, batch, rewards)
                except:
                    logger.warning("Failed to save stats due to unknown errors.")

            self.state.global_step += 1
            self.log_callback.on_step_end(self.args, self.state, self.control)
            if self.is_local_process_zero() and (step + 1) % self.args.logging_steps == 0:
                logs = {
                    "loss": round(loss_meter.avg, 4),
                    "reward": round(reward_meter.avg, 4),
                    "Learning_rate": stats["ppo/learning_rate"],
                    "epoch": round(step / steps_in_epoch, 2),
                }
                tqdm.write(str(logs))
                logs["step"] = step
                self.state.log_history.append(logs)
                self.log_callback.on_log(self.args, self.state, self.control)
                loss_meter.reset()
                reward_meter.reset()
            if (step + 1) % self.args.save_steps == 0:  # save checkpoint
                self.save_model(
                    os.path.join(self.args.output_dir, "{}-{}".format(PREFIX_CHECKPOINT_DIR, self.state.global_step))
                )

                self.save_callback.on_save(
                    self.args, self.state, self.control, model=self.accelerator.unwrap_model(self.model)
                )

            if self.control.should_epoch_stop or self.control.should_training_stop:
                break

        self.log_callback.on_train_end(self.args, self.state, self.control)
        self.save_callback.on_train_end(
            self.args, self.state, self.control, model=self.accelerator.unwrap_model(self.model)
        )

    @torch.no_grad()
    def get_inputs(self, batch: Dict[str, torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        r"""
        Generates model's responses given queries.
        """
        if self.script_args.upcast_layernorm:
            layernorm_params = dump_layernorm(self.model)
        if batch["input_ids"].size(0) == 1:  # handle llama2 ppo with gradient accumulation > 1
            start_index = (batch["input_ids"][0] != self.tokenizer.pad_token_id).nonzero()[0].item()
            for k, v in batch.items():
                batch[k] = v[:, start_index:]

        unwrapped_model: "AutoModelForCausalLMWithValueHead" = self.accelerator.unwrap_model(self.model)
        generate_output: torch.Tensor = unwrapped_model.generate(
            generation_config=self.generation_config, logits_processor=get_logits_processor(), **batch
        )

        if self.script_args.upcast_layernorm:
            restore_layernorm(self.model, layernorm_params)
        query = batch["input_ids"].detach().cpu()
        response = generate_output[:, batch["input_ids"].size(-1) :].detach().cpu()
        queries, responses = [], []
        for i in range(len(query)):
            query_start_index = (query[i] != self.tokenizer.pad_token_id).nonzero()[0].item()
            response_index = (response[i] != self.tokenizer.pad_token_id).nonzero()
            if len(response_index) == 0:
                response_length = 1  # allow empty response
            else:
                response_length = response_index[-1].item() + 1
            queries.append(query[i, query_start_index:])  # remove padding from left
            responses.append(response[i, :response_length])  # remove padding from right

        return queries, responses

    @torch.no_grad()
    def get_rewards(
        self,
        queries: List[torch.Tensor],
        responses: List[torch.Tensor],
        unwrapped_model: "AutoModelForCausalLMWithValueHead",
    ) -> List[torch.Tensor]:
        r"""
        Computes scores using given reward model.
        Both inputs and outputs are put on CPU.
        """

        if self.srcipt_args.reward_model_type == "lora":
            replace_model(unwrapped_model, target="reward")
            reward_model = self.model
        else:
            reward_model = self.reward_model
        batch = self.prepare_model_inputs(queries, responses)
        with torch.cuda.amp.autocast(dtype=self.script_args.compute_dtype):  # support bf16
            _, _, values = reward_model(**batch, output_hidden_states=True, return_dict=True)
        if getattr(unwrapped_model.config, "model_type", None) == "chatglm":  # assume same architecture
            values = torch.transpose(values, 0, 1)
        rewards = []
        for i in range(values.size(0)):
            end_indexes = (batch["input_ids"][i] != self.tokenizer.pad_token_id).nonzero()
            end_index = end_indexes[-1].item() if len(end_indexes) else 0
            rewards.append(values[i, end_index].float().detach().cpu())  # use fp32 type
        if self.srcipt_args.reward_model_type == "lora":
            replace_model(unwrapped_model, target="default")

        return rewards

    @PPODecorators.empty_device_cache()
    def batched_forward_pass(
        self,
        model: "AutoModelForCausalLMWithValueHead",
        queries: torch.Tensor,
        responses: torch.Tensor,
        model_inputs: dict,
        return_logits: Optional[bool] = False,
        response_masks: Optional[torch.Tensor] = None,
    ):
        r"""

        Calculates model outputs in multiple batches.
        Subclass and override to inject custom behavior.

        """
        bs = len(queries)
        fbs = self.config.mini_batch_size
        all_logprobs = []
        all_logits = []
        all_masks = []
        all_values = []
        for i in range(math.ceil(bs / fbs)):
            input_kwargs = {key: value[i * fbs : (i + 1) * fbs] for key, value in model_inputs.items()}
            query_batch = queries[i * fbs : (i + 1) * fbs]
            response_batch = responses[i * fbs : (i + 1) * fbs]
            if response_masks is not None:
                response_masks_batch = response_masks[i * fbs : (i + 1) * fbs]

            input_ids = input_kwargs["input_ids"]
            attention_mask = input_kwargs["attention_mask"]
            with torch.cuda.amp.autocast(dtype=self.script_args.compute_dtype):  # support bf16
                logits, _, values = model(**input_kwargs)
            unwrapped_model: "AutoModelForCausalLMWithValueHead" = self.accelerator.unwrap_model(self.model)
            if getattr(unwrapped_model.config, "model_type", None) == "chatglm":
                values = torch.transpose(values, 0, 1)

            logprobs = logprobs_from_logits(logits[:, :-1, :], input_ids[:, 1:])
            masks = torch.zeros_like(attention_mask)
            masks[:, :-1] = attention_mask[:, 1:]
            for j in range(len(query_batch)):
                start = len(query_batch[j]) - 1
                if attention_mask[j, 0] == 0:  # offset left padding
                    start += attention_mask[j, :].nonzero()[0].item()

                end = start + len(response_batch[j])

                if response_masks is not None:
                    response_masks_batch = torch.cat((torch.zeros_like(query_batch[j]), response_masks_batch[j]))[1:]

                masks[j, :start] = 0
                masks[j, end:] = 0
                if response_masks is not None:
                    masks[j, start:end] = masks[j, start:end] * response_masks_batch[j][start:end]

            if return_logits:
                all_logits.append(logits)

            else:
                del logits

            all_values.append(values)
            all_logprobs.append(logprobs)
            all_masks.append(masks)

        return (
            torch.cat(all_logprobs),
            torch.cat(all_logits)[:, :-1] if return_logits else None,
            torch.cat(all_values)[:, :-1],
            torch.cat(all_masks)[:, :-1],
        )

    def save_model(self, output_dir: Optional[str] = None) -> None:
        r"""
        Saves model checkpoint.
        Subclass and override to inject custom behavior.
        """

        if self.args.should_save:
            try:
                self._save(output_dir, state_dict=self.accelerator.get_state_dict(self.model))
            except ValueError:
                logger.warning(
                    " stage3_gather_16bit_weights_on_model_save=false. Saving the full checkpoint instead,"
                    " use zero_to_fp32.py to recover weights"
                )
                self._save(output_dir, state_dict={})
                for filename in [WEIGHTS_NAME, SAFE_WEIGHTS_NAME]:  # remove dummy checkpoint
                    file = os.path.join(output_dir, filename)
                    if os.path.isfile(file):
                        os.remove(file)
                self.model.save_checkpoint(output_dir)  # wrapped model


def main():
    parser = HfArgumentParser((ScriptArguments, RLHFTraingArguemnts))
    script_args, rlhf_training_args = parser.parse_args_into_dataclasses()
    logger.info(f"Parse script args: {script_args}")
    config_class, model_class, tokenizer_class = MODEL_CLASSES[script_args.model_type]

    if script_args.model_type == "bloom":
        script_args.use_fast_tokenizer = True
    # Load tokenizer
    tokenizer_kwargs = {
        "cache_dir": script_args.cache_dir,
        "use_fast": script_args.use_fast_tokenizer,
        "trust_remote_code": script_args.trust_remote_code,
    }
    # load model
    if script_args.model_name_or_path is not None:
        tokenizer_name_or_path = script_args.tokenizer_name_or_path
        if not tokenizer_name_or_path:
            tokenizer_name_or_path = script_args.model_name_or_path
        tokenizer = tokenizer_class.from_pretrained(tokenizer_name_or_path, **tokenizer_kwargs)
        template = get_template_and_fix_tokenizer(script_args.template_name, tokenizer)
        torch_dtype = (
            script_args.torch_dtype
            if script_args.torch_dtype in ["auto", None]
            else getattr(torch, script_args.torch_dtype)
        )
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if world_size > 1:
            script_args.device_map = {"": int(os.environ.get("LOCAL_RANK", "0"))}

        config = config_class.from_pretrained(
            script_args.model_name_or_path,
            torch_dtype=torch_dtype,
            trust_remote_code=script_args.trust_remote_code,
            cache_dir=script_args.cache_dir,
        )
        if script_args.model_type in ["bloom", "llama"]:
            model = model_class.from_pretrained(
                script_args.model_name_or_path,
                config=config,
                torch_dtype=torch_dtype,
                load_in_4bit=script_args.load_in_4bit,
                load_in_8bit=script_args.load_in_8bit,
                trust_remote_code=script_args.trust_remote_code,
            )

        else:
            model = model_class.from_pretrained(
                script_args.model_name_or_path,
                config=config,
                cache_dir=script_args.cache_dir,
            )

    model = AutoModelForCausalLMWithValueHead.from_pretrained(model)
    patch_valuehead_model(model)

    if script_args.vhead_ckpt_path is not None:
        vhead_path = script_args.vhead_ckpt_path
    else:
        vhead_path = script_args.model_name_or_path

    vhead_param = load_valuehead_params(vhead_path, script_args)
    if vhead_param is not None:
        model.load_state_dict(vhead_param, strict=True)

        if script_args.use_peft:
            logger.info("Fine-tuning method: LoRA(PEFT)")
            if script_args.peft_ckpt_path is not None:
                logger.info(f"Peft from pre-trained model: {script_args.peft_ckpt_path}")
                model = PeftModel.from_pretrained(model, script_args.peft_ckpt_path, is_trainable=True)
            else:
                logger.info("Init new peft model")
                if script_args.quantization_bit:
                    model = prepare_model_for_kbit_training(model)
                if script_args.lora_target == "all":
                    lora_target = find_all_linear_names(model, script_args.quantization_bit)
                else:
                    # support custom target modules/layers of LoRA
                    lora_target = [target.strip() for target in script_args.lora_target.split(",")]
                modules_to_save = script_args.modules_to_save
                if modules_to_save is not None:
                    modules_to_save = modules_to_save.split(",")

                logger.info(f"Peft lora_target: {lora_target}")
                logger.info(f"Peft lora_rank: {script_args.lora_rank}")

                peft_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    target_modules=lora_target,
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

        peft_config = None
        if script_args.use_peft:
            logger.info("Fine-tuning method: LoRA(PEFT)")
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                target_modules=script_args.target_modules,
                inference_mode=False,
                r=script_args.lora_rank,
                lora_alpha=script_args.lora_alpha,
                lora_dropout=script_args.lora_dropout,
            )
        else:
            logger.info("Fine-tuning method: Full parameters training")

        model = AutoModelForCausalLMWithValueHead.from_pretrained(
            script_args.model_name_or_path,
            config=config,
            torch_dtype=torch_dtype,
            device_map=script_args.device_map,
            trust_remote_code=script_args.trust_remote_code,
            peft_config=peft_config if script_args.use_peft else None,
        )

        print_trainable_parameters(model)

    else:
        raise ValueError("Error, model_name_or_path is None, RM must be loaded from a pre-trained model")
    if script_args.reward_model_name_or_path is not None:
        # Load reward model
        default_device = "cuda" if torch.cuda.is_available() else "cpu"
        device = script_args.reward_model_device if script_args.reward_model_device is not None else default_device
        reward_config = config_class.from_pretrained(
            script_args.reward_model_name_or_path,
            torch_dtype=torch_dtype,
            trust_remote_code=script_args.trust_remote_code,
            cache_dir=script_args.cache_dir,
        )
        reward_model = AutoModelForSequenceClassification.from_pretrained(
            script_args.reward_model_name_or_path,
            config=reward_config,
            load_in_8bit=script_args.load_in_8bit,
            trust_remote_code=script_args.trust_remote_code,
        )
        reward_model.to(device)
        reward_tokenizer = AutoTokenizer.from_pretrained(script_args.reward_model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError("Error,reward_model_name_or_path is None,RLHF show load a reward model")

    config = PPOConfig(
        steps=script_args.max_steps,
        model_name=script_args.model_name_or_path,
        learning_rate=script_args.learning_rate,
        log_with=script_args.report_to,
        batch_size=script_args.batch_size,
        mini_batch_size=script_args.mini_batch_size,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        optimize_cuda_cache=True,
        early_stopping=script_args.early_stopping,
        target_kl=script_args.target_kl,
        seed=script_args.seed,
        init_kl_coef=script_args.init_kl_coef,
        adap_kl_ctrl=script_args.adap_kl_ctrl,
        project_kwargs={"logging_dir": rlhf_training_args.output_dir},
    )

    set_seed(config.seed)


if __name__ == "__main__":
    main()
