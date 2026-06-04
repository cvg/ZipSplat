"""Utilities for DataLoader workers: collation, distributed info.

Author: Alexander Veicht
"""

import collections
import dataclasses
import os

import torch
import torch.distributed as dist
from tensordict import TensorClass
from torch.utils.data import get_worker_info
from torch.utils.data._utils.collate import default_collate_err_msg_format, np_str_obj_array_pattern

from splatfactory.utils import types

# --- Distributed helpers ------------------------------------------------------


def get_rank() -> int:
    """Get the rank of the current process, falling back to RANK env var for forkserver workers."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return int(os.environ.get("RANK", 0))


def get_world_size() -> int:
    """Get the world size, falling back to WORLD_SIZE env var for forkserver workers."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return int(os.environ.get("WORLD_SIZE", 1))


def get_dataloader_worker_info() -> tuple[int, int]:
    """Return (worker_id, num_workers) for the current DataLoader worker."""
    info = get_worker_info()
    if info is not None:
        return info.id, info.num_workers
    return 0, 1


# --- Collation ----------------------------------------------------------------


def collate(batch):
    """Difference with PyTorch default_collate: it can stack of other objects."""
    if not isinstance(batch, list):  # no batching
        return batch
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        return torch.stack(batch, dim=0)
    elif isinstance(elem, TensorClass):
        return torch.stack(batch, dim=0)
    elif (
        elem_type.__module__ == "numpy"
        and elem_type.__name__ != "str_"
        and elem_type.__name__ != "string_"
    ):
        if elem_type.__name__ == "ndarray" or elem_type.__name__ == "memmap":
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))
            return collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, types.STRING_CLASSES):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        return {key: collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, "_fields"):  # namedtuple
        return elem_type(*(collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError("each element in list of batch should be of equal size")
        transposed = zip(*batch)
        return [collate(samples) for samples in transposed]
    elif elem is None:
        return elem
    elif dataclasses.is_dataclass(elem):
        # do not convert dataclass until we move to tensordict
        return batch
    else:
        # try to stack anyway in case the object implements stacking.
        return torch.stack(batch, 0)
