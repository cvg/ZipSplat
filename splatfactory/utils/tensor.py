"""TensorWrapper base for packed per-tensor types, with numpy<->torch autocast.

Adapted from gluefactory (https://github.com/cvg/glue-factory), Apache-2.0.
"""

import functools
from typing import Any, Callable, Dict

import numpy as np
import torch
from tensordict import TensorClass

string_classes = (str, bytes)


def autocast(func):
    """Cast the inputs of a TensorWrapper method to PyTorch tensors
    if they are numpy arrays. Use the device and dtype of the wrapper.
    """

    @functools.wraps(func)
    def wrap(self, *args):
        device = torch.device("cpu")
        dtype = None
        if isinstance(self, torch.Tensor):
            device = self.device
            dtype = self.dtype
        # elif not inspect.isclass(self) or not issubclass(self, torch.Tensor):
        #     raise ValueError(self)

        cast_args = []
        for arg in args:
            if isinstance(arg, np.ndarray):
                arg = torch.from_numpy(arg)
                arg = arg.to(device=device, dtype=dtype)
            cast_args.append(arg)
        return func(self, *cast_args)

    return wrap


def autovmap(func):
    """Batch and vectorize a simple TensorWrapper method."""

    @functools.wraps(func)
    def wrap(self, arg):
        # assert isinstance(self, TensorWrapper), self
        cls = self.__class__
        _wrap = lambda d, x: wrap(cls(d), x)
        if arg.ndim == self.data_.ndim and arg.ndim == 1:
            return func(self, arg)
        elif arg.ndim == 3 and self.data_.ndim == 2 and arg.shape[0] == self.data_.shape[0]:
            # Handle the lazy scenario: wrapper BxP, arg BxNxD
            return torch.vmap(_wrap)(self.data_, arg)
        elif arg.ndim > self.data_.ndim:
            return torch.vmap(_wrap, in_dims=(None, 0))(self.data_, arg)
        else:
            arg = arg.broadcast_to(self.shape + arg.shape[-1:])
            if arg.ndim == self.data_.ndim:
                return torch.vmap(_wrap)(self.data_, arg)
            else:
                raise ValueError(
                    f"Broadcast failed: self.data_={self.data_.shape}, arg={arg.shape}."
                )

    return wrap


class TensorWrapper(TensorClass, tensor_only=True):
    """Wrapper for PyTorch tensors."""

    data_: torch.Tensor

    def __post_init__(self):
        self.batch_size = self.data_.shape[:-1]

    @property
    def device(self) -> torch.device:
        return self.data_.device

    def __deepcopy__(self, memo: Dict[int, Any]) -> "TensorWrapper":
        # The tensorclass __deepcopy__ will be deprecated, so we clone instead.
        return self.clone()

    @classmethod
    def where(cls, condition, input, other, *, out=None):
        if not (isinstance(input, cls) and isinstance(other, cls)):
            raise ValueError(f"Incorrect inputs: {input}, {other}.")
        if out is not None and isinstance(out, cls):
            out = out.data_
        ret = torch.where(condition.unsqueeze(-1), input.data_, other.data_, out=out)
        return cls(ret)

    @classmethod
    def __torch_function__(
        cls,
        func: Callable,
        types: tuple[type, ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ):
        if func == torch.concat:
            func = torch.cat
        if func == torch.where:
            func = cls.where

        # Handle torch.cat for tensordict compatibility
        if func == torch.cat and args and isinstance(args[0], list):
            if all(isinstance(x, cls) for x in args[0]):
                data_list = [x.data_ for x in args[0]]
                result_data = torch.cat(data_list, *args[1:], **(kwargs or {}))
                return cls(result_data)

        # Handle torch.stack for tensordict 0.9.0 compatibility
        if func.__name__ == "stack" and args and isinstance(args[0], list):
            if hasattr(args[0][0], "_tensordict"):
                from tensordict import stack as td_stack

                return td_stack(args[0], *args[1:], **(kwargs or {}))

        return getattr(cls, func.__name__)(*args, **(kwargs or {}))
