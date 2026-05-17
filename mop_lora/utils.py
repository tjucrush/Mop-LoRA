import torch
import torch.nn as nn
from model import MoPLinear

def inject_mop_lora(model: nn.Module, target_modules: list = ["q_proj", "v_proj"], r: int = 8, num_paths: int = 4, lora_alpha: int = 16):
    """
    Recursively replaces target linear layers with MoPLinear layers.
    """
    replacement_count = 0
    for name, module in model.named_children():
        if isinstance(module, nn.Linear) and any(t in name for t in target_modules):
            # Replace nn.Linear with MoPLinear
            mop_layer = MoPLinear(
                base_layer=module,
                r=r,
                num_paths=num_paths,
                lora_alpha=lora_alpha
            )
            setattr(model, name, mop_layer)
            replacement_count += 1
        else:
            # Recursion for nested layers
            replacement_count += inject_mop_lora(module, target_modules, r, num_paths, lora_alpha)
            
    return replacement_count

def print_trainable_parameters(model: nn.Module):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.4f}"
    )
