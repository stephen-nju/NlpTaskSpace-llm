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
            # last layer is not add to lora_module_names
            if "lm_head" in name:
                continue
            if "output_layer" in name:
                continue
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    return list(lora_module_names)
