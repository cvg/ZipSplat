"""Timers, step and epoch breakdowns, running statistics and device stats.

Author: Paul-Edouard Sarlin (skydes) and contributors.
Partly adapted from gluefactory (https://github.com/cvg/glue-factory), Apache-2.0.
"""

import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from tabulate import tabulate

from splatfactory import get_logger

logger = get_logger(__name__)


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
