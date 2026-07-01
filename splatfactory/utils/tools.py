"""
Various handy Python and PyTorch utils.

Author: Paul-Edouard Sarlin (skydes)

Adapted from gluefactory (https://github.com/cvg/glue-factory), Apache-2.0.
"""

import collections
import functools
import os
import random
import re
import time
from collections.abc import Callable, Iterable, Sequence
from contextlib import contextmanager
from typing import Any, List

import matplotlib.pyplot as plt
import numpy as np
import plotly
import torch
from omegaconf import OmegaConf
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler

from splatfactory import get_logger

logger = get_logger(__name__)


def get_device() -> str:
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    return device


class Timer(object):
    """A simpler timer context object.
    Usage:
    ```
    > with Timer('mytimer'):
    >   # some computations
    [mytimer] Elapsed: X
    ```
    """

    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()
        return self

    def __exit__(self, type, value, traceback):
        self.duration = time.time() - self.tstart
        if self.name is not None:
            print("[%s] Elapsed: %s" % (self.name, self.duration))


def timeit(func):
    """Simple wrapper to time a function."""

    def wrapper(*args, **kwargs):
        with Timer(func.__name__):
            return func(*args, **kwargs)

    return wrapper


class RunningStats:
    """
    A numerically stable running statistics tracker using Welford's algorithm.
    Avoids overflow and maintains precision for large datasets.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all statistics to initial state."""
        self.count = 0
        self.mean = 0.0
        self.M2 = 0.0  # Sum of squares of deviations from mean

    def update(self, val: float) -> None:
        # Welford's online algorithm
        self.count += 1
        delta = val - self.mean
        self.mean += delta / self.count
        delta2 = val - self.mean
        self.M2 += delta * delta2

    def compute(self) -> tuple[float, float]:
        """Compute the mean and standard deviation."""
        if self.count == 0:
            return 0.0, 0.0
        elif self.count == 1:
            return self.mean, 0.0
        else:
            variance = self.M2 / self.count  # Population variance
            std = np.sqrt(variance)
            return self.mean, std


class StepTimer:
    def __init__(self):
        self.stats: dict[str, RunningStats] = {}
        self.start = None

    def reset(self):
        """Reset the timer for a specific name."""
        self.start = time.time()

    def hard_reset(self):
        self.stats = {}
        self.start = time.time()

    def measure(self, name: str):
        """Measure the time taken for a specific operation."""
        # torch.cuda.synchronize()
        elapsed = time.time() - self.start
        if name not in self.stats:
            self.stats[name] = RunningStats()
        self.stats[name].update(elapsed)
        self.start = time.time()

    def compute(self) -> tuple[float, dict[str, float]]:
        """Compute the average time for each operation (in seconds)."""
        avg_step_times = {k: v.compute()[0] for k, v in self.stats.items()}
        total_time = sum(avg_step_times.values())
        return total_time, avg_step_times

    def log_stats(self):
        from tabulate import tabulate

        total_time = 0
        for name, stats in self.stats.items():
            section_time, section_var = stats.compute()
            total_time += section_time

        table = [
            [name, f"{stats.compute()[0]:.4f}s", f"{(stats.compute()[0]/total_time*100):.2f}%"]
            for name, stats in self.stats.items()
        ]
        # add total row
        table.append(["Total", f"{total_time:.4f}s", "100.00%"])

        tbl_str = tabulate(table, headers=["Name", "Avg Time", "% of Step"], tablefmt="github")
        logger.info(f"Step time breakdown:\n\n{tbl_str}\n")

    def plot(self):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        total_time = 0
        for name, stats in self.stats.items():
            section_time, section_var = stats.compute()
            ax.bar(name, section_time * 1000, label=name, yerr=section_var * 1000)
            total_time += section_time
        ax.set_ylabel("Duration (ms)")
        ax.set_title(
            f"Step Time Composition " f"(total: {total_time:.2f}s = {1 / total_time:.2f} steps/s)"
        )

        return fig

    def num_steps(self):
        """Return the number of steps measured."""
        return list(self.stats.values())[0].count if self.stats else 0

    def __getitem__(self, name: str) -> float:
        """Get the average time for a specific operation."""
        return self.stats[name].compute()[0]


def _format_count(n: int) -> str:
    """Format count as 12.35k or 1.23M."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    elif n >= 1_000:
        return f"{n / 1_000:.2f}k"
    return str(n)


def _format_eta(seconds: float) -> str:
    """Format ETA as Xs, X.Xm, X.Xh, or X.Xd."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    elif seconds < 86400:
        return f"{seconds / 3600:.1f}h"
    else:
        return f"{seconds / 86400:.1f}d"


class EpochTimer:
    """Track epoch timing for progress/ETA estimation."""

    def __init__(self):
        # Running averages (survives checkpoint resume via self-correction)
        self.train_time_avg = 0.0
        self.total_time_avg = 0.0
        self.num_epochs = 0

        # Current epoch tracking
        self.epoch_start = None
        self.section_start = None
        self.current_sections = {}

    def start_epoch(self):
        """Call at start of each epoch."""
        self.epoch_start = time.time()
        self.section_start = time.time()
        self.current_sections = {}

    def measure(self, name: str):
        """Record time for a section (train, eval, ckpt, bench)."""
        if self.section_start is None:
            return
        elapsed = time.time() - self.section_start
        self.current_sections[name] = elapsed
        self.section_start = time.time()

    def end_epoch(self) -> dict:
        """Finalize epoch and update running averages. Returns section times."""
        if self.epoch_start is None:
            return {}

        total_time = time.time() - self.epoch_start
        train_time = self.current_sections.get("train", total_time)

        # Update running averages: avg = avg + (new - avg) / count
        self.num_epochs += 1
        self.train_time_avg += (train_time - self.train_time_avg) / self.num_epochs
        self.total_time_avg += (total_time - self.total_time_avg) / self.num_epochs

        result = dict(self.current_sections)
        self.epoch_start = None
        self.section_start = None
        self.current_sections = {}
        return result

    def train_fraction(self) -> float:
        """Fraction of epoch spent training."""
        if self.total_time_avg > 0:
            return self.train_time_avg / self.total_time_avg
        return 1.0

    def format_eta(self, remaining_its: int, its_per_sec: float) -> str:
        """ETA adjusted for non-training overhead."""
        if its_per_sec <= 0 or remaining_its <= 0:
            return "-"
        raw_eta = remaining_its / its_per_sec
        adjusted_eta = raw_eta / self.train_fraction()
        return _format_eta(adjusted_eta)

    def log_epoch_summary(self, epoch: int):
        """Log epoch time breakdown table."""
        from tabulate import tabulate

        if not self.current_sections and self.num_epochs == 0:
            return

        # Use last epoch's data (already stored in current_sections before end_epoch clears it)
        # This should be called before end_epoch, or we use the averages
        total_time = (
            sum(self.current_sections.values()) if self.current_sections else self.total_time_avg
        )
        if total_time == 0:
            return

        sections = (
            self.current_sections if self.current_sections else {"train": self.train_time_avg}
        )

        table = [
            [name, f"{t:.2f}s", f"{(t / total_time * 100):.2f}%"] for name, t in sections.items()
        ]
        table.append(["Total", f"{total_time:.2f}s", "100.00%"])

        tbl_str = tabulate(table, headers=["Name", "Time", "% of Epoch"], tablefmt="github")
        logger.info(f"Epoch {epoch} time breakdown:\n\n{tbl_str}\n")


def collect_device_stats() -> dict[str, float]:
    """Collect device usage statistics."""

    def _per_device_stats(device: torch.device | None = None) -> dict[str, float]:
        free, total = torch.cuda.mem_get_info(device)
        used = total - free
        bytes_stats = {
            # "z_allocated": torch.cuda.memory_allocated(device),
            # "z_reserved": torch.cuda.memory_reserved(device),
            "z_allocated_peak": torch.cuda.max_memory_allocated(device),
            "z_reserved_peak": torch.cuda.max_memory_reserved(device),
            "used": used,
            "total": total,
        }

        device_stats = {k: v / 10**9 for k, v in bytes_stats.items()}
        device_stats["utilization"] = device_stats["used"] / device_stats["total"]
        # Reset peak memory stats for next cycle
        torch.cuda.reset_peak_memory_stats(device)
        return device_stats

    num_devices = torch.cuda.device_count()
    all_devices = [torch.cuda.device(i) for i in range(num_devices)]
    all_device_stats = [_per_device_stats(d) for d in all_devices]
    all_device_stats = {k: [pds[k] for pds in all_device_stats] for k in all_device_stats[0]}
    device_stats = {k: np.mean(v).item() for k, v in all_device_stats.items()}
    for i in range(num_devices):
        device_stats[f"utilization_{i}"] = all_device_stats["utilization"][i]
    # Assumes all devices have the same memory stats.
    device_stats["global_total"] = sum(all_device_stats["total"])
    device_stats["global_used"] = sum(all_device_stats["used"])
    return device_stats


def get_class(mod_path, BaseClass):
    """Get the class object which inherits from BaseClass and is defined in
    the module named mod_name, child of base_path.
    """
    import inspect

    mod = __import__(mod_path, fromlist=[""])
    classes = inspect.getmembers(mod, inspect.isclass)
    # Filter classes defined in the module
    classes = [c for c in classes if c[1].__module__ == mod_path]
    # Filter classes inherited from BaseModel
    classes = [c for c in classes if issubclass(c[1], BaseClass)]
    assert len(classes) == 1, classes
    return classes[0][1]


def set_num_threads(nt):
    """Force numpy and other libraries to use a limited number of threads."""
    try:
        import mkl
    except ImportError:
        pass
    else:
        mkl.set_num_threads(nt)
    torch.set_num_threads(1)
    os.environ["IPC_ENABLE"] = "1"
    for o in [
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
    ]:
        os.environ[o] = str(nt)


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_random_state(with_cuda):
    pth_state = torch.get_rng_state()
    np_state = np.random.get_state()
    py_state = random.getstate()
    if torch.cuda.is_available() and with_cuda:
        cuda_state = torch.cuda.get_rng_state_all()
    else:
        cuda_state = None
    return pth_state, np_state, py_state, cuda_state


def set_random_state(state):
    pth_state, np_state, py_state, cuda_state = state
    torch.set_rng_state(pth_state)
    np.random.set_state(np_state)
    random.setstate(py_state)
    if (
        cuda_state is not None
        and torch.cuda.is_available()
        and len(cuda_state) == torch.cuda.device_count()
    ):
        torch.cuda.set_rng_state_all(cuda_state)


@contextmanager
def fork_rng(seed=None, with_cuda=True):
    state = get_random_state(with_cuda)
    if seed is not None:
        set_seed(seed)
    try:
        yield
    finally:
        set_random_state(state)


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


def pack_lr_parameters(params, base_lr, lr_scaling):
    """Pack each group of parameters with the respective scaled learning rate."""
    if lr_scaling:
        filters, scales = tuple(
            zip(*[(n, s) for pattern, s in lr_scaling.items() for n in pattern.split("+")])
        )
    else:
        filters, scales = [], []
    scale2params = collections.defaultdict(list)
    filter_patterns = [re.compile(r'(?:^|[._])' + re.escape(f) + r'(?=[._]|$)') for f in filters]
    
    for n, p in params:
        scale = 1
        is_match = [bool(pat.search(n)) for pat in filter_patterns]
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

    lr_params = [
        {"lr": scale * base_lr, "params": [p for _, p in ps]} for scale, ps in scale2params.items()
    ]
    return lr_params


def write_dict_summaries(writer, name: str, items: dict, step: int):
    for k, v in items.items():
        key = f"{name}/{k}"
        if isinstance(v, dict):
            writer.add_scalars(key, v, step=step)
        elif isinstance(v, tuple):
            writer.add_pr_curve(f"pr/{key}", *v, step=step)
        else:
            writer.add_scalar(key, v, step=step)


def write_image_summaries(writer, name: str, figures: List[Any], step: int):
    # Stacked grayscale is not supported, convert to RGB!
    def _add_plot(tag, fig_or_image, step):
        if isinstance(fig_or_image, (np.ndarray, torch.Tensor)):
            if fig_or_image.ndim in (2, 3):
                writer.add_image(tag, fig_or_image, step)
            else:
                assert fig_or_image.ndim == 4
                writer.add_images(tag, fig_or_image, step)
        elif isinstance(fig_or_image, plt.Figure):
            # Figure or list[Figure]
            writer.add_figure(tag, fig_or_image, step)
            plt.close(fig_or_image)
        elif isinstance(fig_or_image, plotly.graph_objs.Figure):
            writer.add_plotly(tag, fig_or_image, step)
        else:
            raise ValueError(f"Unsupported figure or image type: {type(fig_or_image)}")

    if isinstance(figures, list):
        for i, figs in enumerate(figures):
            for k, fig in figs.items():
                _add_plot(f"{name}/{i}_{k}", fig, step)
    else:
        for k, fig in figures.items():
            _add_plot(f"{name}/{k}", fig, step)


def grad_norm(params):
    """Compute the norm of gradients of parameters."""
    return torch.nn.utils.get_total_norm([p.grad for p in params if p.grad is not None])


def param_norm(params):
    """Compute the norm of parameters."""
    return torch.nn.utils.get_total_norm([p for p in params if p.requires_grad])
