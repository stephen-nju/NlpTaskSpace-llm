import bitsandbytes as bnb
import torch


def find_all_linear_names(model, quantization_bits: str):
    cls = (
        bnb.nn.Linear4bit
        if quantization_bits == "4bit"
        else (bnb.nn.Linear8bitLt if quantization_bits == "8bit" else torch.nn.Linear)
    )
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
            if "lm_head" in lora_module_names:  # needed for 16-bit
                lora_module_names.remove("lm_head")

    return list(lora_module_names)


from typing import Any

import pytorch_lightning as pl
import transformers.deepspeed


class ZeRO3Config:
    def __init__(self, pl_module):
        self.config = pl_module.trainer.strategy.config

        def __call__(self, *args, **kwargs) -> Any:
            return self

        def is_zero3(self) -> bool:
            return self.config.get("zero_optimization") and self.config.get("zero_optimization").get("stage") == 3


def enable_transformers_pretrained_deepspeed_sharding(pl_module: "pl.LightningModule") -> None:
    transformers.deepspeed._hf_deepspeed_config_weak_ref = ZeRO3Config(pl_module)
