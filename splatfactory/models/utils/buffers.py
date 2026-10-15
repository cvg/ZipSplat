"""Turn a module's parameters into buffers (frozen, optionally out of the state_dict)."""

import torch.nn as nn


def convert_to_buffer(module: nn.Module, persistent: bool = True):
    """Re-register every parameter and buffer of `module` (recursively) as a buffer.

    With persistent=False the tensors also leave the state_dict, so a frozen pretrained
    network inside a loss (LPIPS) adds no checkpoint keys and no optimizer state.
    """
    for _, child in list(module.named_children()):
        convert_to_buffer(child, persistent)
    own = (*module.named_parameters(recurse=False), *module.named_buffers(recurse=False))
    for name, tensor in own:
        value = tensor.detach().clone()
        delattr(module, name)
        module.register_buffer(name, value, persistent=persistent)
