import bitsandbytes as bnb
import torch


def find_all_linear_names(model, quantization_bits):
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
