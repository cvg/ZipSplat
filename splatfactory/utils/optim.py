"""Optimization helpers: parameter filtering, LR scheduling/scaling, and gradient/param norms.

Author: Paul-Edouard Sarlin (skydes) and contributors.
Partly adapted from gluefactory (https://github.com/cvg/glue-factory), Apache-2.0.
"""

import collections
import re

import torch
from omegaconf import OmegaConf
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler

from splatfactory import get_logger

logger = get_logger(__name__)


def filter_parameters(params, regexp):
    """Filter trainable parameters based on regular expressions."""

    # Examples of regexp:
    #     '.*(weight|bias)$'
    #     'cnn\.(enc0|enc1).*bias'
    def filter_fn(x):
        n, p = x
        match = re.search(regexp, n)
        if not match:
            p.requires_grad = False
        return match

    params = list(filter(filter_fn, params))
    assert len(params) > 0, regexp
    logger.info("Selected parameters:\n" + "\n".join(n for n, p in params))
    return params


def get_lr_scheduler(optimizer: Optimizer, conf: OmegaConf) -> LRScheduler:
    """Get lr scheduler specified by conf.

    Example usage:
    conf = {
        "type": "SequentialLR",
        "options": {
            "milestones": [1_000],
            "schedulers": [
                {"type": "LinearLR", "options": {"total_iters": 10, "start_factor": 0.001}},
                {"type": "MultiStepLR", "options": {"milestones": [40, 60], "gamma": 0.1}},
            ],
        }
    }
    """
    # logger.info(f"Using lr scheduler with conf: {conf}")
    if hasattr(conf.options, "schedulers"):
        # Add option to chain multiple schedulers together
        # This is useful for e.g. warmup, then cosine decay
        schedulers = []
        for scheduler_conf in conf.options.schedulers:
            scheduler = get_lr_scheduler(optimizer, scheduler_conf)
            schedulers.append(scheduler)

        options = {k: v for k, v in conf.options.items() if k != "schedulers"}
        return getattr(torch.optim.lr_scheduler, conf.type)(optimizer, schedulers, **options)

    return getattr(torch.optim.lr_scheduler, conf.type)(optimizer, **conf.options)


def pack_lr_parameters(params, base_lr, lr_scaling, weight_decay_skip=None):
    """Pack each group of parameters with the respective scaled learning rate.

    If weight_decay_skip is set (a regexp), matched params are split into a group with
    weight_decay=0.0, overriding the optimizer default for those params only.
    """
    if lr_scaling:
        filters, scales = tuple(
            zip(*[(n, s) for pattern, s in lr_scaling.items() for n in pattern.split("+")])
        )
    else:
        filters, scales = [], []
    scale2params = collections.defaultdict(list)
    for n, p in params:
        scale = 1
        # TODO: use proper regexp rather than just this inclusion check
        is_match = [f in n for f in filters]
        if any(is_match):
            scale = scales[is_match.index(True)]
        scale2params[scale].append((n, p))

    n_scaled_lr_params = sum(len(ps) for s, ps in scale2params.items() if s != 1)
    logger.info(f"Number of parameters with scaled learning rate: {n_scaled_lr_params}")

    for s, ps in scale2params.items():
        logger.debug(f"LR scale {s}: {len(ps)} parameter groups")
        for n, _ in ps:
            if "bias" in n or "norm" in n.lower():
                continue
            logger.debug(f"  - {n}")

    lr_params = []
    for scale, ps in scale2params.items():
        if weight_decay_skip:
            decay = [p for n, p in ps if not re.search(weight_decay_skip, n)]
            no_decay = [p for n, p in ps if re.search(weight_decay_skip, n)]
            if decay:
                lr_params.append({"lr": scale * base_lr, "params": decay})
            if no_decay:
                lr_params.append({"lr": scale * base_lr, "params": no_decay, "weight_decay": 0.0})
        else:
            lr_params.append({"lr": scale * base_lr, "params": [p for _, p in ps]})
    return lr_params


def grad_norm(params):
    """Compute the norm of gradients of parameters."""
    return torch.nn.utils.get_total_norm([p.grad for p in params if p.grad is not None])


def param_norm(params):
    """Compute the norm of parameters."""
    return torch.nn.utils.get_total_norm([p for p in params if p.requires_grad])
