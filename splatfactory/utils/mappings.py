"""Nested-structure (dict/list/TensorWrapper) mapping and device-transfer helpers.

Adapted from gluefactory (https://github.com/cvg/glue-factory), Apache-2.0.
"""

import pprint
from collections.abc import MutableMapping
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import torch

from splatfactory.utils import tensor, types
from splatfactory.utils.tensor import TensorWrapper


def map_tensor(input_, func):
    scalar_classes = (str, bytes, bool, float, int)
    if isinstance(input_, scalar_classes):
        return input_
    elif isinstance(input_, Mapping):
        return {k: map_tensor(sample, func) for k, sample in input_.items()}
    elif isinstance(input_, Sequence):
        return input_.__class__([map_tensor(sample, func) for sample in input_])
    elif isinstance(input_, np.ndarray):
        return func(torch.from_numpy(input_))
    elif input_ is None:
        return None
    else:
        return func(input_)


def batch_to_numpy(batch):
    """Recursively convert a batch of tensors to numpy arrays."""
    return map_tensor(
        batch,
        lambda tensor: (
            tensor.cpu().numpy()
            if tensor.dtype != torch.bfloat16
            else tensor.cpu().to(torch.float32).numpy()
        ),
    )


def batch_to_tensor(batch, dtype: torch.dtype = None):
    """Recursively convert a batch of arrays to tensors."""

    def _func(ndarray):
        if isinstance(ndarray, TensorWrapper):
            return ndarray

        tensor_ = torch.as_tensor(ndarray)
        if dtype is not None:
            tensor_ = tensor_.to(dtype=dtype)
        return tensor_

    return map_tensor(batch, _func)


def batch_to_device(batch, device, detach=False, non_blocking=False):
    """Recursively move a batch of tensors/arrays to a device."""
    if device == "numpy":
        return batch_to_numpy(batch)

    def _func(tensor):
        if detach:
            tensor = tensor.detach()
        return tensor.to(device=device, non_blocking=non_blocking)

    return map_tensor(batch, _func)


def remove_batch_dim(data: dict) -> dict:
    """Remove batch dimension from elements in data"""
    return tree_map(data, lambda t: t[0])


def add_batch_dim(data: dict) -> dict:
    """Add batch dimension to elements in data"""
    return tree_map(data, lambda t: t[None])


def unsqueeze_n(tensor: torch.Tensor, dim: int, n: int) -> torch.Tensor:
    for _ in range(n):
        tensor = tensor.unsqueeze(dim)
    return tensor


def bunsqueeze_like(tensor: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    assert tensor.shape[0] == ref.shape[0], "Batch size must match"
    for _ in range(ref.dim() - tensor.dim()):
        tensor = tensor.unsqueeze(-1)
    return tensor


def add_prefix(d: dict, prefix: str) -> dict:
    return {prefix + k: v for k, v in d.items()}


def index_batch(tensor_dict):
    batch_size = len(next(iter(tensor_dict.values())))
    for i in range(batch_size):
        yield map_tensor(tensor_dict, lambda t: t[i])


def to_view(data, i):
    return {k + i: v for k, v in data.items()}


def get_view(data, i):
    data_g = {k: v for k, v in data.items() if not k[-1].isnumeric()}
    data_i = {k[:-1]: v for k, v in data.items() if k[-1] == i}
    return {**data_g, **data_i}


def get_twoview(data, idx):
    li = idx[0]
    ri = idx[-1]
    assert idx == f"{li}to{ri}"
    data_lr = {k[:-4] + "0to1": v for k, v in data.items() if k[-4:] == f"{li}to{ri}"}
    data_rl = {k[:-4] + "1to0": v for k, v in data.items() if k[-4:] == f"{ri}to{li}"}
    data_l = {k[:-1] + "0": v for k, v in data.items() if k[-1:] == li and k[-3:-1] != "to"}
    data_r = {k[:-1] + "1": v for k, v in data.items() if k[-1:] == ri and k[-3:-1] != "to"}
    return {**data_lr, **data_rl, **data_l, **data_r}


def stack_twoviews(data, indices=["0to1", "0to2", "1to2"]):
    idx0 = indices[0]
    m_data = data[idx0] if idx0 in data else get_twoview(data, idx0)
    # stack on dim=0
    for idx in indices[1:]:
        data_i = data[idx] if idx in data else get_twoview(data, idx)
        for k, v in data_i.items():
            m_data[k] = torch.cat([m_data[k], v], dim=0)
    return m_data


def unstack_twoviews(data, B, indices=["0to1", "0to2", "1to2"]):
    out = {}
    for i, idx in enumerate(indices):
        out[idx] = {k: v[i * B : (i + 1) * B] for k, v in data.items()}
    return out


def iterelements(data: dict, pattern="view") -> Iterable[Any]:
    i = 0
    while True:
        view = data.get(f"{pattern}{i}", None)
        if view is None:
            break
        yield view
        i += 1


def pack_elements(data, pattern="view"):
    return pack_tree(iterelements(data, pattern=pattern))


def concat_elements(data, pattern="view"):
    return concat_tree(iterelements(data, pattern=pattern))


def pack_tree(
    trees: Iterable[types.Tree],
    check: bool = False,
    fn: Callable[[Sequence[Any]], Any] = lambda x: x,
    sep: str | None = ".",
) -> types.Tree:
    """Concatenate a list of trees into a list per entry"""
    if not trees:
        return {}

    trees = list(trees)
    flat_trees = [flatten_dict(batch, sep=sep) for batch in trees]
    keys = set(flat_trees[0].keys())
    if check:
        for batch in trees[1:]:
            if keys != set(batch.keys()):
                raise ValueError("All trees must have the same keys.")
    joined_tree = {k: fn([batch[k] for batch in flat_trees]) for k in keys}
    return unflatten_dict(joined_tree, sep=sep)


def concat_tree(trees: Iterable[types.Tree], check: bool = False, dim: int = 0) -> types.Tree:
    """Concatenate a list of trees into a single batch"""

    def combine(val_list: Sequence[Any]) -> Any:
        if isinstance(val_list[0], (torch.Tensor, tensor.TensorWrapper)):
            return torch.cat(val_list, dim=dim)
        elif isinstance(val_list[0], tuple):
            return tuple([combine([v[i] for v in val_list]) for i in range(len(val_list[0]))])
        elif isinstance(val_list[0], Sequence):
            return sum(val_list, start=[])
        elif isinstance(val_list[0], (int, float)):
            return val_list
        else:
            raise TypeError(f"Cannot combine values of type {type(val_list[0])}")

    return pack_tree(trees, check=check, fn=combine)


def stack_tree(trees: Iterable[types.Tree], dim: int = 0, check: bool = False) -> types.Tree:
    """Stack a list of trees into a single batch"""

    def combine(val_list: Sequence[Any]) -> Any:
        if isinstance(val_list[0], (torch.Tensor, tensor.TensorWrapper)):
            return torch.stack(val_list, dim=dim)
        elif isinstance(val_list[0], tuple):
            return tuple([combine([v[i] for v in val_list]) for i in range(len(val_list[0]))])
        elif isinstance(val_list[0], Sequence):
            return [combine([v[i] for v in val_list]) for i in range(len(val_list[0]))]
        elif isinstance(val_list[0], (int, float)):
            return val_list
        else:
            raise TypeError(f"Cannot combine values of type {type(val_list[0])}")

    return pack_tree(trees, check=check, fn=combine)


def split_tree(tree, num_splits: int) -> list[types.Tree]:
    """Split a tree into a list of trees along the first dimension of tensors."""
    flat_tree = flatten_dict(tree)

    split_trees = [{}] * num_splits
    for k, v in flat_tree.items():
        if isinstance(v, (torch.Tensor, tensor.TensorWrapper)):
            splits = torch.split(v, num_splits, dim=0)
            for i in range(num_splits):
                split_trees[i][k] = splits[i]
        else:
            raise NotImplementedError(f"Cannot split values of type {type(v)}")
    return [unflatten_dict(t) for t in split_trees]


def compare_tree(
    tree_i: types.Tree,
    tree_j: types.Tree,
    compare_fn: Callable[[Any, Any], bool | None] | None = None,
) -> types.Tree:
    if compare_fn is None:

        def compare_fn(el1, el2):
            if isinstance(el1, torch.Tensor):
                if el1.dtype in [torch.float16, torch.float32, torch.float64]:
                    return torch.all(torch.abs(el1 - el2) < 1e-2).item()
                return torch.all(el1 == el2).item()
            if isinstance(el1, np.ndarray):
                if np.issubdtype(el1.dtype, np.floating):
                    return np.all(np.abs(el1 - el2) < 1e-2)
                return np.array_equal(el1, el2)
            elif isinstance(el1, (int, float, str, bool)):
                return el1 == el2
            elif isinstance(el1, Iterable):
                return all(compare_fn(e1, e2) for e1, e2 in zip(el1, el2))
            else:
                return None

    is_equal = pack_tree([tree_i, tree_j], fn=lambda x: compare_fn(x[0], x[1]))
    flat_is_equal = flatten_dict(is_equal)
    flat_is_equal = {k: v for k, v in flat_is_equal.items() if v is not None}
    return flat_is_equal


def flatten_dict(
    dictionary: Mapping[str, Any],
    parent_keys: tuple[str, ...] = (),
    sep: str | None = ".",
    cast_to_str: bool = False,
) -> dict[str | tuple[str, ...], Any]:
    items = []
    for key, value in dictionary.items():
        new_key = parent_keys + (key,)
        if isinstance(value, MutableMapping):
            items.extend(flatten_dict(value, new_key, sep=sep).items())
        else:
            items.append((new_key, value))
    flat_dict = dict(items)
    if len(parent_keys) == 0 and sep is not None:
        # Top-level
        return {sep.join(map(str, k) if cast_to_str else k): v for k, v in flat_dict.items()}
    else:
        return flat_dict


def unflatten_dict(
    flat_dict: Mapping[str | tuple[str, ...], Any],
    sep: str | None = ".",
) -> dict[str, Any]:
    unflattened = {}
    for key, value in flat_dict.items():
        if isinstance(key, tuple):
            parts = key
        elif sep is not None:
            parts = key.split(sep)
        else:
            parts = (key,)
        current = unflattened
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value
    return unflattened


def flat_map(
    input_: types.Tree,
    func: Callable[[types.Key, types.Value], types.Value],
    sep: str | None = ".",
    unflatten: bool = False,
) -> types.Tree:
    """Apply a function to each item in a flattened dictionary."""
    flat_dict = flatten_dict(input_, sep=sep)
    out = {}
    for k in sorted(flat_dict.keys()):
        out[k] = func(k, flat_dict[k])
    if unflatten:
        out = unflatten_dict(out, sep=sep)
    return out


def filter_tree(
    input_: types.Tree | Any,
    valid_fn: Callable[[types.Key, types.Value], bool],
    sep: str | None = None,
) -> types.Tree:
    """Filter a tree structure based on a predicate function."""
    flat_dict = flatten_dict(input_, sep=sep)
    filtered = {k: v for k, v in flat_dict.items() if valid_fn(k, v)}
    return unflatten_dict(filtered, sep=sep)


def tree_map(
    input_: types.Tree,
    func: Callable[[types.Value], types.Value],
    sep: str | None = None,
    unflatten: bool = True,
) -> types.Tree:
    """Apply a function to each item in a flattened dictionary."""
    return flat_map(input_, func=lambda k, v: func(v), sep=sep, unflatten=unflatten)


def tree_tensormap(
    input_: types.Tree,
    func: Callable[[torch.Tensor], torch.Tensor],
    sep: str | None = None,
    unflatten: bool = True,
) -> types.Tree:
    """Apply a function to each tensor item in a flattened dictionary."""
    return flat_map(
        input_,
        func=lambda k, v: func(v) if isinstance(v, torch.Tensor) else v,
        sep=sep,
        unflatten=unflatten,
    )


def all_gather(
    item: torch.Tensor | Any, num_devices: int | None = None, dim: int | None = 0
) -> torch.Tensor | Sequence[torch.Tensor]:
    """Gather item from all devices."""
    if num_devices is None:
        num_devices = torch.distributed.get_world_size()
    if isinstance(item, torch.Tensor):
        item_list = [torch.zeros_like(item) for _ in range(num_devices)]
        torch.distributed.all_gather(item_list, item)
    else:
        item_list = [None for _ in range(num_devices)]
        torch.distributed.all_gather_object(item_list, item)
    if dim is None:
        return item_list
    else:
        return torch.cat(item_list, dim=dim)


def tree_all_gather(tree: types.Tree) -> types.Tree:
    """Gather all tensors from all devices."""
    trees = all_gather(tree, dim=None)
    return concat_tree(trees)


def tree_summary(tree: types.Tree, flatten: bool = False) -> str:
    """Summarize a tree structure."""

    def _summarize(t):
        if isinstance(t, torch.Tensor):
            return f"{type(t).__name__}{tuple(t.shape)} {t.dtype}"
        elif isinstance(t, tensor.TensorWrapper):
            return f"{type(t).__name__}{tuple(t.shape)} {t.dtype}"
        elif isinstance(t, np.ndarray):
            return f"ndarray{t.shape} {t.dtype}"
        elif isinstance(t, (list, tuple)):
            return f"{type(t).__name__}[{len(t)}]"
        elif isinstance(t, (int, str, bytes, float, bool)):
            return str(t)
        elif t is None:
            return "None"
        else:
            return type(t).__name__

    return pprint.pformat(
        tree_map(tree, _summarize, unflatten=not flatten, sep=("." if flatten else None)),
        indent=2,
    )


def print_summary(tree: types.Tree, flatten: bool = False):
    print(tree_summary(tree, flatten=flatten))
