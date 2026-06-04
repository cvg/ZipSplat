"""Distributed-aware running metric accumulators (mean / median / recall).

Adapted from gluefactory (https://github.com/cvg/glue-factory), Apache-2.0.
"""

from collections.abc import Iterable
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist


def _allgather_list(elements: list) -> list:
    """Gather a list from all ranks into a single flat list."""
    all_elements = [None] * dist.get_world_size()
    dist.all_gather_object(all_elements, elements)
    return [e for sublist in all_elements for e in sublist]


class AverageMetric:
    def __init__(self, elements=None, distributed=False):
        self.distributed = distributed
        if elements is None:
            elements = []
            self._sum = 0
            self._num_examples = 0
        else:
            mask = ~np.isnan(elements)
            self._sum = sum(elements[mask])
            self._num_examples = len(elements[mask])

    def update(self, tensor):
        assert tensor.dim() == 1, tensor.shape
        tensor = tensor[~torch.isnan(tensor)]
        self._sum += tensor.sum().item()
        self._num_examples += len(tensor)

    def compute(self):
        s, n = self._sum, self._num_examples
        if self.distributed and dist.is_initialized():
            s_t = torch.tensor(s, dtype=torch.float64, device="cuda")
            n_t = torch.tensor(n, dtype=torch.float64, device="cuda")
            dist.all_reduce(s_t)
            dist.all_reduce(n_t)
            s, n = s_t.item(), int(n_t.item())
        return np.nan if n == 0 else s / n

    def reset(self):
        self._sum = 0
        self._num_examples = 0


# same as AverageMetric, but tracks all elements
class FAverageMetric:
    def __init__(self, distributed=False):
        self.distributed = distributed
        self._sum = 0
        self._num_examples = 0
        self._elements = []

    def update(self, tensor):
        self._elements += tensor.cpu().numpy().tolist()
        assert tensor.dim() == 1, tensor.shape
        tensor = tensor[~torch.isnan(tensor)]
        self._sum += tensor.sum().item()
        self._num_examples += len(tensor)

    def compute(self):
        s, n = self._sum, self._num_examples
        if self.distributed and dist.is_initialized():
            s_t = torch.tensor(s, dtype=torch.float64, device="cuda")
            n_t = torch.tensor(n, dtype=torch.float64, device="cuda")
            dist.all_reduce(s_t)
            dist.all_reduce(n_t)
            s, n = s_t.item(), int(n_t.item())
        return np.nan if n == 0 else s / n

    def reset(self):
        self._sum = 0
        self._num_examples = 0
        self._elements = []


class MedianMetric:
    def __init__(self, elements=None, distributed=False):
        self.distributed = distributed
        if elements is None:
            elements = []
        self._elements = elements

    def update(self, tensor):
        assert tensor.dim() == 1, tensor.shape
        self._elements += tensor.cpu().numpy().tolist()

    def compute(self):
        elements = self._elements
        if self.distributed and dist.is_initialized():
            elements = _allgather_list(elements)

        if len(elements) == 0:
            return np.nan

        elements = np.array(elements)
        elements[np.isnan(elements)] = np.inf
        return np.nanmedian(elements)

    def reset(self):
        self._elements = []


class PRMetric:
    def __init__(self, distributed=False):
        self.distributed = distributed
        self.labels = []
        self.predictions = []

    @torch.no_grad()
    def update(self, labels, predictions, mask=None):
        assert labels.shape == predictions.shape
        self.labels += (labels[mask] if mask is not None else labels).cpu().numpy().tolist()
        self.predictions += (
            (predictions[mask] if mask is not None else predictions).cpu().numpy().tolist()
        )

    @torch.no_grad()
    def compute(self):
        labels, predictions = self.labels, self.predictions
        if self.distributed and dist.is_initialized():
            labels = _allgather_list(labels)
            predictions = _allgather_list(predictions)
        return np.array(labels), np.array(predictions)

    def reset(self):
        self.labels = []
        self.predictions = []


class QuantileMetric:
    def __init__(self, q=0.05, distributed=False):
        self.distributed = distributed
        self._elements = []
        self.q = q

    def update(self, tensor):
        assert tensor.dim() == 1
        self._elements += tensor.cpu().numpy().tolist()

    def compute(self):
        elements = self._elements
        if self.distributed and dist.is_initialized():
            elements = _allgather_list(elements)

        if len(elements) == 0:
            return np.nan
        else:
            return np.nanquantile(elements, self.q)

    def reset(self):
        self._elements = []


class RecallMetric:
    def __init__(self, ths, elements=None, distributed=False):
        self.distributed = distributed
        if elements is None:
            elements = []
        self._elements = elements
        self.ths = ths

    def update(self, tensor):
        assert tensor.dim() == 1, tensor.shape
        self._elements += tensor.cpu().numpy().tolist()

    def compute(self):
        elements = self._elements
        if self.distributed and dist.is_initialized():
            elements = _allgather_list(elements)

        elements = np.array(elements)
        elements[np.isnan(elements)] = np.inf

        if isinstance(self.ths, Iterable):
            return [self._compute(elements, th) for th in self.ths]
        else:
            return self._compute(elements, self.ths[0])

    def _compute(self, elements, th):
        if len(elements) == 0:
            return np.nan
        s = (elements < th).sum()
        return s / len(elements)

    def reset(self):
        self._elements = []


def compute_recall(errors):
    num_elements = len(errors)
    sort_idx = np.argsort(errors)
    errors = np.array(errors.copy())[sort_idx]
    recall = (np.arange(num_elements) + 1) / num_elements
    return errors, recall


def compute_auc(errors, thresholds, min_error: Optional[float] = None):
    errors, recall = compute_recall(errors)

    if min_error is not None:
        min_index = np.searchsorted(errors, min_error, side="right")
        min_score = min_index / len(errors)
        recall = np.r_[min_score, min_score, recall[min_index:]]
        errors = np.r_[0, min_error, errors[min_index:]]
    else:
        recall = np.r_[0, recall]
        errors = np.r_[0, errors]

    aucs = []
    for t in thresholds:
        last_index = np.searchsorted(errors, t, side="right")
        r = np.r_[recall[:last_index], recall[last_index - 1]]
        e = np.r_[errors[:last_index], t]
        auc = np.trapz(r, x=e) / t
        aucs.append(np.round(auc, 4))
    return aucs


class AUCMetric:
    def __init__(
        self, thresholds, elements=None, min_error: Optional[float] = None, distributed=False
    ):
        self.distributed = distributed
        self._elements = elements if elements is not None else []
        self.thresholds = thresholds
        self.min_error = min_error
        if not isinstance(thresholds, list):
            self.thresholds = [thresholds]

    def update(self, tensor):
        assert tensor.dim() == 1, tensor.shape
        self._elements += tensor.cpu().numpy().tolist()

    def compute(self):
        elements = self._elements
        if self.distributed and dist.is_initialized():
            elements = _allgather_list(elements)

        if len(elements) == 0:
            return np.nan

        elements = np.array(elements)
        elements[np.isnan(elements)] = np.inf
        return compute_auc(elements, self.thresholds, self.min_error)

    def reset(self):
        self._elements = []
