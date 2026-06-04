"""
A generic, flexible trainer.

Author: Philipp Lindenberger

Adapted from gluefactory (https://github.com/cvg/glue-factory), Apache-2.0.
"""

import collections
import gc
import math
import signal
from pathlib import Path
from typing import Any, TypeAlias

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from splatfactory import __module_name__, datasets, eval, logger, models, settings
from splatfactory.models import BaseModel
from splatfactory.models.misc import format_module_params
from splatfactory.utils import experiments, mappings, tools
from splatfactory.utils.distributed import get_distributed_env
from splatfactory.utils.metrics import AverageMetric, MedianMetric, PRMetric, RecallMetric
from splatfactory.utils.summary_writer import SummaryWriter

Args: TypeAlias = DictConfig
Batch: TypeAlias = Any
Predictions: TypeAlias = Any
LossMetrics: TypeAlias = dict[str, torch.Tensor]
Writer: TypeAlias = SummaryWriter | None


def compose_loss(loss_dict: LossMetrics, compose_str: str) -> torch.Tensor:
    """Compose a loss from a string, e.g. '1.0*loss1 + 0.1*loss2'."""
    loss = 0.0
    for term in compose_str.split("+"):
        term = term.strip()
        if "*" in term:
            weight_str, key = term.split("*")
            weight = float(weight_str)
        else:
            weight = 1.0
            key = term
        key = key.strip()
        if key not in loss_dict:
            raise KeyError(f"Key {key} not found in loss dict.")
        loss = loss + weight * loss_dict[key]
    return loss


def get_batch_size(loader_or_dataset) -> int:
    """Get the nominal batch size (for logging, not sample counting)."""
    if hasattr(loader_or_dataset, "max_img_per_gpu"):
        return loader_or_dataset.max_img_per_gpu
    loader = loader_or_dataset
    if hasattr(loader, "batch_size") and loader.batch_size is not None:
        return loader.batch_size
    elif hasattr(loader, "batch_sampler") and loader.batch_sampler is not None:
        return getattr(loader.batch_sampler, "max_img_per_gpu", 1)
    return 1


def _infer_batch_size(data: dict, fallback: int) -> int:
    """Infer actual batch size from a data dict. Falls back to nominal value."""
    if "scene" in data:
        return len(data["scene"])
    for key in ("context", "target"):
        if isinstance(data.get(key), dict):
            for v in data[key].values():
                if hasattr(v, "shape") and len(v.shape) >= 1:
                    return v.shape[0]
    return fallback


@torch.compiler.set_stance("force_eager")
@torch.no_grad()
def run_evaluation(
    model: BaseModel | torch.nn.parallel.DistributedDataParallel,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    conf: DictConfig,
    rank: int = 0,
    pbar: bool = True,
    max_iters: int | None = None,
    compose_loss_str: str | None = None,
) -> tuple[Any, ...]:
    model.eval()
    is_ddp = isinstance(model, torch.nn.parallel.DistributedDataParallel)
    model = model.module if is_ddp else model  # Get the original model
    results = {}
    pr_metrics = collections.defaultdict(lambda: PRMetric(distributed=is_ddp))
    figures = []

    # choose plot ids in first 10 batches
    plot_ids = np.random.choice(
        min(len(loader), 10), min(len(loader), conf.num_eval_plots), replace=False
    )
    max_iters = max_iters or len(loader)
    max_iters = min(max_iters, len(loader))
    eval_iter = iter(loader)
    for i in tqdm(range(len(loader)), desc="Evaluation", ascii=True, disable=not pbar):
        if i >= max_iters:
            break
        try:
            data = next(eval_iter)
            local_done = torch.zeros(1, device=device, dtype=torch.int32)
        except StopIteration:
            local_done = torch.ones(1, device=device, dtype=torch.int32)

        if is_ddp:
            dist.all_reduce(local_done, op=dist.ReduceOp.MAX)

        if local_done.item():
            logger.info("Eval DataLoader exhausted on at least one rank, stopping.")
            break

        data = mappings.batch_to_device(data, device, non_blocking=True)
        pred = model(data)
        losses, metrics = model.loss_metrics(pred, data)
        pr_metrics_i = model.pr_metrics(pred, data)
        losses, metrics, pr_metrics_i = [
            mappings.batch_to_device(x, "cpu", non_blocking=False)
            for x in (losses, metrics, pr_metrics_i)
        ]
        if is_ddp:
            pr_metrics_i = mappings.tree_all_gather(pr_metrics_i)
        if compose_loss_str is not None:
            losses["total"] = compose_loss(losses, compose_loss_str)
        if i in plot_ids:
            figures.append(model.visualize(pred, data))
        for k, (labels, preds) in pr_metrics_i.items():
            pr_metrics[k].update(labels, preds)
        del pred, data
        numbers = {**metrics, **{"loss/" + k: v for k, v in losses.items()}}
        del losses, metrics, pr_metrics_i
        for k, v in numbers.items():
            if k not in results:
                results[k] = AverageMetric(distributed=is_ddp)
                if k in conf.median_metrics:
                    results[k + "_median"] = MedianMetric(distributed=is_ddp)
                if k in conf.recall_metrics.keys():
                    q = conf.recall_metrics[k]
                    results[k + f"_recall{int(q)}"] = RecallMetric(q, distributed=is_ddp)
            results[k].update(v)
            if k in conf.median_metrics:
                results[k + "_median"].update(v)
            if k in conf.recall_metrics.keys():
                q = conf.recall_metrics[k]
                results[k + f"_recall{int(q)}"].update(v)
        del numbers

    del eval_iter
    results = {k: results[k].compute() for k in results}
    # Compute and reset PR metrics to free memory
    computed_pr = {}
    for k, v in pr_metrics.items():
        computed_pr[k] = v.compute()
        v.reset()
    del pr_metrics
    return results, computed_pr, figures


class Trainer:
    """
    Trainer class for managing the training process.

    Maintains model, params and training state (optim, step, ...)
    """

    default_conf = {
        "seed": "???",  # training seed
        # Training parameters
        "epochs": None,  # number of epochs
        "num_steps": None,  # number of steps, overwrites epochs
        "eval_every_iter": None,  # interval for evaluation (iteration-based)
        "eval_every_epoch": 1,  # interval for evaluation on the validation set
        "mixed_precision": None,
        "num_devices": 0,  # 0 means sequential.
        "detect_anomaly": False,  # Enable anomaly detection
        "matmul_precision": None,  # Set torch.matmul precision [None, highest, high, medium, low]
        "overfit": False,  # Overfit a single batch
        # Optimizer parameters
        "optimizer": "Adam",  # name of optimizer torch.optim.* e.g Adam, SGD, ...
        "opt_regexp": None,  # regular expression to filter parameters to optimize
        "optimizer_options": {},  # optional arguments passed to the optimizer
        "lr": 0.001,  # learning rate
        "lr_schedule": {
            "type": None,
            "start": 0,
            "exp_div_10": 0,
            "on_epoch": False,
            "factor": 1.0,
        },
        "lr_scaling": {},  # learning rate scaling for parameter name patterns
        "clip_grad": None,
        "gradient_accumulation_steps": 1,  # Accumulate gradients over N steps
        # Data parameters
        "train_split": "train",  # split to use for training
        "eval_split": "val",  # split to use for evaluation
        "test_every_epoch": 1,  # interval for evaluation on the test benchmarks
        "benchmark_every_epoch": 1,  # interval for evaluation on the test benchmarks
        "run_benchmarks": (),
        # Logging and checkpointing
        "writer": "tensorboard",  # options: [tensorboard, wandb]
        "best_key": "loss/total",  # key to use to select the best checkpoint
        "save_every_iter": 5000,  # interval for saving the current checkpoint
        "log_every_iter": 200,  # interval for logging the loss to the console
        "log_grad_every_iter": None,  # interval for logging gradient hists
        "keep_last_checkpoints": 1,  # keep only the last X checkpoints
        "median_metrics": [],  # add the median of some metrics
        "recall_metrics": {},  # add the recall of some metrics
        "pr_curves": {},  # add pr curves, set labels/predictions/mask keys
        "num_eval_plots": 4,  # Number of plots to show during evaluation (0=skip)
        "plot_every_iter": None,  # plot figures every X iterations
        "stdout_metrics": ["loss/total"],  # List of metrics to print to stdout (None=all)
        "record_memory": None,  # Record memory usage during training (# record steps)
        "log_it": False,  # Log tensorboard on iteration (default is num_samples)
        "print_arch": 1,  # Print model architecture (None=skip, int=depth)
        # Restarting and loading
        "load_experiment": None,  # initialize the model from a previous experiment
        "submodules": [],
        # Misc options
        "compile": None,  # Compilation mode for the model. [None, default, ...]
        "profile": None,  # Profile the training with PyTorch profiler (# prof steps)
        "profile_every_epoch": None,  # Profile every N epochs, None means only first
        "ddp_find_unused_parameters": False,  # DDP find_unused_parameters
        "stop_immediately": True,  # Stop training immediately on SIGINT
        "debug_nan": False,  # Log detailed info and raise on NaN gradients
        "project_name": __module_name__,  # wandb project name
    }

    def __init__(
        self,
        conf: DictConfig,
        model: BaseModel,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LambdaLR,
        device: torch.device | str | None = None,
    ):
        # Initialize conf, model, optimizer and LR
        self.default_conf = OmegaConf.create(self.default_conf)
        self.conf = OmegaConf.merge(self.default_conf, conf)
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.env = get_distributed_env()
        self.distributed = self.env.is_distributed
        self.num_gpus = self.env.world_size if self.distributed else 1

        # Initialize rank
        if self.distributed:
            assert dist.is_initialized(), "Torch Distributed not initialized"
        self.rank = self.env.rank

        # Initialize device
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        if self.device is None:
            device = (
                self.rank if self.distributed else "cuda" if torch.cuda.is_available() else "cpu"
            )
            self.device = torch.device(device)

        # Initialize model params and conf
        self.model_conf = self.model.conf
        if self.conf.print_arch is not None:
            self.info(
                f"Model architecture:\n{format_module_params(self.model, self.conf.print_arch)}"
            )

        # Setup scaler and dtype
        self.use_mp = self.setup_dtype_scaler(conf.mixed_precision)

        # Initialize step timer
        self.step_timer = tools.StepTimer()
        self.epoch_timer = tools.EpochTimer()

        # Setup counters
        self.epoch = 0
        self.step_in_epoch = 0
        self.tot_n_samples = 0
        self.tot_it = 0

        assert (
            conf.num_steps is not None or conf.epochs is not None
        ), "At least one of num_steps or epochs must be specified."

        # Handle KeyboardInterrupt
        self.setup_sigint_handler()

        self.prepare_model()

        # Named benchmark configs
        self.benchmarks = {}

        # Setup torch global variables
        self.setup_torch()

    # ------------------------------------------------------------------------
    # Utility Initializers
    # ------------------------------------------------------------------------

    @classmethod
    def init(cls, conf: DictConfig, model: BaseModel, **kwargs) -> "Trainer":
        """Create a Trainer instance from a config."""
        conf = OmegaConf.merge(cls.default_conf, conf)
        optimizer = cls.construct_optimizer(conf, model)
        lr_scheduler = tools.get_lr_scheduler(optimizer=optimizer, conf=conf.lr_schedule)
        return cls(
            conf=conf,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            **kwargs,
        )

    # ------------------------------------------------------------------------
    # Setup helper functions (public)
    # ------------------------------------------------------------------------

    def register_benchmark(
        self, benchmark_name: str, benchmark_conf: str, every_epoch: int | None = None
    ):
        every_epoch = every_epoch or self.conf.benchmark_every_epoch
        self.info(f"Registering benchmark {benchmark_name} (every={every_epoch}).")
        self.benchmarks[benchmark_name] = (benchmark_conf, every_epoch)

    def sequential_model(self) -> BaseModel:
        """Get the original model (without DDP)."""
        return (
            self.model.module
            if isinstance(self.model, torch.nn.parallel.DistributedDataParallel)
            else self.model
        )

    def load_checkpoint(
        self,
        checkpoint: Any,
        strict: bool = True,
        load_state: bool = False,
        load_modelconfig: bool = False,
    ):
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            # Fix distributed model naming
            checkpoint["model"] = {
                (k if k.startswith("module.") else "module." + k): v
                for k, v in checkpoint["model"].items()
            }
        missing, unexpected = self.model.load_state_dict(checkpoint["model"], strict=strict)
        self.info(
            f"state_dict loaded. Missing keys: {missing or 'None'}. Unexpected keys: {unexpected or 'None'}."
        )
        if load_modelconfig:
            self.conf.model = OmegaConf.merge(
                OmegaConf.create(checkpoint["conf"]).model, self.conf.model
            )
            self.info("Model config loaded.")
        if load_state:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            if "lr_scheduler" in checkpoint:
                self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            self.epoch = checkpoint["epoch"]
            for metric in ["tot_it", "tot_n_samples"]:
                if metric in checkpoint:
                    setattr(self, metric, checkpoint[metric])
                    self.info(f"Loaded {metric}={getattr(self, metric)} ({checkpoint[metric]})")
            self.info(f"Training state loaded. Resuming at epoch {self.epoch}.")

    def maybe_load_checkpoint(self):
        if self.conf.load_experiment:
            if self.conf.get("load_last_checkpoint", True):
                self.info(
                    "Loading last checkpoint from experiment %s",
                    self.conf.load_experiment,
                )
                init_cp = experiments.get_last_checkpoint(self.conf.load_experiment)
            else:
                self.info(
                    "Loading best checkpoint from experiment %s",
                    self.conf.load_experiment,
                )
                init_cp = experiments.get_best_checkpoint(self.conf.load_experiment)
            self.info("Loading checkpoint %s", str(init_cp))
            init_cp = experiments.load_checkpoint(
                init_cp,
                map_location="cpu",
                weights_only=self.conf.get("load_weights_only", not settings.ALLOW_PICKLE),
            )
            self.load_checkpoint(
                init_cp,
                load_state=self.conf.get("load_state", False),
                load_modelconfig=self.conf.get("load_modelconfig", False),
                strict=self.conf.get("load_strict", True),
            )

    def save_checkpoint(
        self,
        output_dir: Path,
        conf: DictConfig,  # This is the full conf!
        results: dict | None = None,
        iter_i: int = 0,
        best_eval: float | None = None,
        **kwargs,
    ) -> float | None:
        """Save checkpoint and return updated best_eval value."""
        if self.rank == 0:
            return experiments.save_experiment(
                self.model,
                self.optimizer,
                self.lr_scheduler,
                conf,
                results,
                iter_i=iter_i,
                epoch=self.epoch,
                output_dir=output_dir,
                custom={
                    "tot_it": self.tot_it,
                    "tot_n_samples": self.tot_n_samples,
                },
                distributed=self.distributed,
                best_eval=best_eval,
                **kwargs,
            )
        return best_eval

    # ------------------------------------------------------------------------
    # Setup helper functions (internal)
    # ------------------------------------------------------------------------

    def info(self, pattern: str, *args, **kwargs):
        if self.rank == 0:
            logger.info(pattern, *args, **kwargs)

    def warn(self, pattern: str, *args, **kwargs):
        if self.rank == 0:
            logger.warning(pattern, *args, **kwargs)

    def learning_rate_step(self, verbose: bool = False):
        old_lr = self.optimizer.param_groups[0]["lr"]
        self.lr_scheduler.step()
        if verbose:
            self.info(f"lr changed from {old_lr} to {self.optimizer.param_groups[0]['lr']}")

    def _dump_nan_debug(self, data: Batch, pred: Predictions, loss_metrics: LossMetrics):
        """Dump detailed diagnostics when NaN gradients are detected (debug_nan=True)."""
        R = self.rank
        logger.error(f"[rank {R}] NaN gradients at epoch={self.epoch}, iter={self.step_in_epoch}")

        # Dump batch contents (skip large tensors)
        try:
            skip_keys = {"image", "depth"}

            def _dump(d, prefix=""):
                for k, v in d.items() if isinstance(d, dict) else []:
                    if k in skip_keys:
                        logger.error(f"  [rank {R}] {prefix}{k}: shape={list(v.shape)}")
                    elif isinstance(v, dict):
                        _dump(v, prefix=f"{prefix}{k}.")
                    elif isinstance(v, torch.Tensor):
                        logger.error(f"  [rank {R}] {prefix}{k}: {v}")
                    else:
                        logger.error(f"  [rank {R}] {prefix}{k}: {v}")

            _dump(data)
        except Exception as e:
            logger.error(f"  [rank {R}] Failed to dump batch: {e}")

        # Per-param non-finite grad info
        try:
            for name, p in self.model.named_parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    bad = ~torch.isfinite(p.grad)
                    bad_frac = bad.float().mean().item()
                    finite = p.grad[torch.isfinite(p.grad)]
                    grad_max = finite.abs().max().item() if finite.numel() > 0 else float("nan")
                    n_nan = torch.isnan(p.grad).sum().item()
                    n_inf = torch.isinf(p.grad).sum().item()
                    logger.error(
                        f"  [rank {R}] Bad grad: {name} shape={list(p.grad.shape)} "
                        f"bad_frac={bad_frac:.3f} nan={n_nan} inf={n_inf} max_abs={grad_max:.4e}"
                    )
        except Exception as e:
            logger.error(f"  [rank {R}] Failed to inspect gradients: {e}")

        # Loss components (per-rank, before all-reduce)
        try:
            for k, v in loss_metrics.items():
                if "loss" in k or "psnr" in k:
                    logger.error(f"  [rank {R}] {k}={v.item():.4e}")
        except Exception as e:
            logger.error(f"  [rank {R}] Failed to dump losses: {e}")

        # Save debug artifacts per rank
        nan_dir = self.output_dir / f"debug_nan_e{self.epoch}_it{self.step_in_epoch}" / f"rank{R}"
        nan_dir.mkdir(parents=True, exist_ok=True)
        for fname, save_fn in [
            ("batch.pt", lambda: mappings.batch_to_device(data, "cpu", detach=True)),
            ("pred.pt", lambda: mappings.batch_to_device(pred, "cpu", detach=True)),
            ("loss_metrics.pt", lambda: {k: v.cpu().detach() for k, v in loss_metrics.items()}),
        ]:
            try:
                torch.save(save_fn(), nan_dir / fname)
                logger.error(f"  [rank {R}] Saved {nan_dir / fname}")
            except Exception as e:
                logger.error(f"  [rank {R}] Failed to save {fname}: {e}")

        # Only save model once (rank 0)
        if R == 0:
            try:
                model_path = nan_dir.parent / "model.pt"
                torch.save(self.sequential_model().state_dict(), model_path)
                logger.error(f"  [rank {R}] Saved {model_path}")
            except Exception as e:
                logger.error(f"  [rank {R}] Failed to save model: {e}")

        raise RuntimeError(f"[rank {R}] NaN detected in gradients (debug_nan=True)")

    def _step_view_schedule(self):
        """Linear ramp max_views: min -> target over scheduled fraction of training."""
        schedule_frac = self._view_schedule_frac
        if not schedule_frac or not self.conf.num_steps:
            return
        if not hasattr(self, "_view_target"):
            self._view_target = self._view_schedule_range[1]
        min_views = self._view_schedule_range[0]
        T = int(self.conf.num_steps * schedule_frac)
        t = min(self.tot_it, T)
        current_max = min_views + (self._view_target - min_views) * (t / T)
        current_max = int(round(current_max))
        self._train_dataset.update_image_num_range(new_max=current_max)

    def _step_query_ratio_schedule(self):
        """Cosine anneal query ratio_min: 1.0 -> target over scheduled fraction of training."""
        model = self.model.module if self.distributed else self.model
        schedule_frac = getattr(model.conf, "query_ratio_schedule", 0.0)
        if not schedule_frac or not self.conf.num_steps:
            return

        if not hasattr(self, "_qr_target"):
            self._qr_target = model.conf.query_sample_ratio[0]

        T = int(self.conf.num_steps * schedule_frac)
        t = min(self.tot_it, T)
        ratio_min = self._qr_target + (1.0 - self._qr_target) * 0.5 * (
            1 + math.cos(math.pi * t / T)
        )
        model.query_sample_ratio[0] = ratio_min

    def setup_sigint_handler(self):
        def sigint_handler(signal, frame):
            logger.info("Caught keyboard interrupt signal, will terminate")
            if self.stop or self.conf.stop_immediately:
                raise KeyboardInterrupt
            self.stop = True

        self.stop = False
        signal.signal(signal.SIGINT, sigint_handler)

    def setup_torch(self):
        torch.backends.cudnn.benchmark = True
        if self.conf.detect_anomaly:
            torch.autograd.set_detect_anomaly(True)
        if self.conf.matmul_precision is not None:
            torch.set_float32_matmul_precision(self.conf.matmul_precision)

    def prepare_model(self):
        if self.conf.compile is not None:
            # Compile before DDP
            self.model = self.model.compile(mode=self.conf.compile)
        if self.distributed:
            self.model = self.model.make_ddp(
                device_ids=[self.device],
                find_unused_parameters=self.conf.ddp_find_unused_parameters,
            )

    def construct_profiler(
        self, output_dir: Path, store_raw_trace: bool = False
    ) -> torch.profiler.profile:
        return torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=5, warmup=1, active=self.conf.profile, repeat=1, skip_first=10
            ),
            on_trace_ready=experiments.tensorboard_trace_handler(
                str(output_dir), use_gzip=not store_raw_trace, epoch=self.epoch
            ),
            record_shapes=False,
            profile_memory=False,
            with_stack=True,
        )

    @classmethod
    def construct_optimizer(cls, conf: DictConfig, model: torch.nn.Module) -> torch.optim.Optimizer:
        """Construct the optimizer for training."""
        optimizer_fn = getattr(torch.optim, conf.optimizer)

        params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        if conf.opt_regexp:
            params = tools.filter_parameters(params, conf.opt_regexp)
        lr_params = tools.pack_lr_parameters(params, conf.lr, conf.lr_scaling)
        optimizer = optimizer_fn(lr_params, lr=conf.lr, **conf.optimizer_options)
        return optimizer

    def setup_dtype_scaler(self, mixed_precision: str | None) -> bool:
        use_mp = mixed_precision is not None
        self.scaler = (
            torch.amp.GradScaler("cuda", enabled=use_mp)
            if hasattr(torch.amp, "GradScaler")
            else torch.cuda.amp.GradScaler(enabled=use_mp)
        )
        self.info(f"Training with mixed_precision={mixed_precision}")

        self.dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            None: torch.float32,  # we disable it anyway
        }[mixed_precision]

        return use_mp

    def get_writer(self, output_dir: Path, log_conf: DictConfig) -> Writer:
        if self.rank == 0:
            writer = SummaryWriter(
                log_dir=output_dir,
                writer=self.conf.writer,
                project=self.conf.project_name,
                conf=log_conf,
                # WandB options
                run_id=self.conf.get("run_id", None),
                name_as_run_id=self.conf.get("name_as_run_id", False),
                reload_run_id=self.conf.get("reload_run_id", True),
            )
        else:
            writer = None
        return writer

    def should_stop(self, max_iters: int | None = None) -> bool:
        """Check if training should stop."""
        if self.stop:  # SIGINT handler
            return True

        if max_iters is not None and self.step_in_epoch >= max_iters:
            self.info(f"Reached max iters {max_iters}, stopping epoch {self.epoch}.")
            return True

        if self.conf.num_steps and self.tot_it >= self.conf.num_steps:
            self.info(f"Reached max steps {self.conf.num_steps}, stopping.")
            return True

        if self.conf.epochs and self.epoch >= self.conf.epochs:
            return True

        return False

    def should_evaluate(self) -> bool:
        """Check if should run evaluation."""
        # Always eval at the end of training
        if self.should_stop():
            return True

        # Iteration-based evaluation
        if self.conf.eval_every_iter and self.step_in_epoch % self.conf.eval_every_iter == 0:
            return True

        # Epoch-based evaluation (at end of epoch)
        if self.epoch % self.conf.eval_every_epoch == 0 or self.epoch == self.conf.epochs:
            return True

        return False

    # ------------------------------------------------------------------------
    # Logging functions
    # ------------------------------------------------------------------------

    @property
    def current_it(self):
        """Get the current iteration identifier."""
        return self.tot_it if self.conf.log_it else self.tot_n_samples

    def record_memory(self, output_dir: Path, it: int, offset: int = 2):
        if it == offset:
            self.info(
                f"Recording memory usage over {self.conf.record_memory} iterations "
                f"(skipped first {offset} it)."
            )
            torch.cuda.memory._record_memory_history(enabled="all")
        elif it == offset + self.conf.record_memory:
            # Record memory usage every self.conf.record_memory iterations
            snapshot_path = output_dir / f"memory_snapshot_epoch{self.epoch}.json"
            torch.cuda.memory._dump_snapshot(snapshot_path)

    def log_train(
        self,
        writer: Writer,
        it: int,
        train_loss_metrics: LossMetrics,
        memory_used: float = 0.0,
        memory_total: float = 0.0,
        steps_per_sec: float = 0.0,
    ):
        tot_n_samples = self.current_it
        all_params = self.model.parameters()
        writer.add_scalar("l2/param_norm", tools.param_norm(all_params), tot_n_samples)
        loss_metrics = {
            k: v.compute() if hasattr(v, "compute") else v for k, v in train_loss_metrics.items()
        }
        if self.conf.stdout_metrics is None:
            str_loss_metrics = [f"{k} {v:.3E}" for k, v in loss_metrics.items()]
        else:
            str_loss_metrics = [
                f"{k} {v:.3E}" for k, v in loss_metrics.items() if k in self.conf.stdout_metrics
            ]

        # Progress and ETA
        total_steps = self.conf.num_steps or float("inf")
        pct = int(self.tot_it / total_steps * 100) if total_steps < float("inf") else 0
        remaining_its = max(0, total_steps - self.tot_it) if total_steps < float("inf") else 0
        eta_str = self.epoch_timer.format_eta(remaining_its, steps_per_sec)

        # Write training losses with new format
        logger.info(
            "[E {} | it {} | {:.1f}/{:.1f} GB | {}/{} ({}%) | {:.1f} it/s | eta {}] loss {{{}}}".format(
                self.epoch,
                it,
                memory_used,
                memory_total,
                tools._format_count(self.tot_it),
                tools._format_count(int(total_steps)) if total_steps < float("inf") else "?",
                pct,
                steps_per_sec,
                eta_str,
                ", ".join(str_loss_metrics),
            )
        )
        tools.write_dict_summaries(writer, "training", loss_metrics, tot_n_samples)
        writer.add_scalar("training/lr", self.optimizer.param_groups[0]["lr"], tot_n_samples)

        model = self.model.module if self.distributed else self.model
        if getattr(model.conf, "query_ratio_schedule", 0.0):
            writer.add_scalar(
                "schedule/query_ratio_min", model.query_sample_ratio[0], tot_n_samples
            )

        if getattr(self, "_view_schedule_frac", 0.0):
            writer.add_scalar(
                "schedule/max_views",
                self._train_dataset.image_num_range[1],
                tot_n_samples,
            )

        # Write Epoch
        writer.add_scalar("training/epoch", self.epoch, tot_n_samples)
        writer.add_scalar("training/tot_it", self.tot_it, tot_n_samples)

    def log_eval(self, writer: Writer, it: int, eval_results: Any):
        tot_n_samples = self.current_it
        results, pr_metrics, figures = eval_results
        str_results = [f"{k} {v:.3E}" for k, v in results.items() if isinstance(v, float)]
        logger.info(f'[Validation] {{{", ".join(str_results)}}}')
        tools.write_dict_summaries(writer, "eval", results, tot_n_samples)
        tools.write_dict_summaries(writer, "eval", pr_metrics, tot_n_samples)
        if figures is not None:
            tools.write_image_summaries(writer, "eval", figures, tot_n_samples)

    def log_time_and_memory(
        self,
        writer: Writer,
        it: int,
        batch_size: int,
    ) -> tuple[float, float, float]:
        """Log timing and memory stats. Returns (memory_used, memory_total, steps_per_sec)."""
        tot_n_samples = self.current_it
        steps_per_sec = 0.0
        if self.step_timer.num_steps() > 1:
            self.step_timer.log_stats()

            step_duration, section_times = self.step_timer.compute()
            steps_per_sec = 1 / step_duration
            writer.add_scalar("step/total", step_duration, tot_n_samples)
            writer.add_scalar("step/_per_sec", steps_per_sec, tot_n_samples)
            writer.add_scalar(
                "step/_samples_per_sec",
                steps_per_sec * batch_size * self.num_gpus,
                tot_n_samples,
            )
            # Write section timings and fractions of step duration.
            for section_name, duration in section_times.items():
                writer.add_scalar(f"step/{section_name}", duration, tot_n_samples)

            writer.add_scalar(
                "step/io_fraction",
                (section_times["data"] + section_times["to_device"]) / step_duration,
                tot_n_samples,
            )

            if it % (self.conf.log_every_iter * 2) == 0:
                # Plot at reduced frequency
                writer.add_figure("step/sections", self.step_timer.plot(), tot_n_samples)

        # Reset the stats after logging
        self.step_timer.stats.clear()

        # Log memory stats
        memory_used, memory_total = 0.0, 0.0
        if torch.cuda.is_available():
            device_stats = tools.collect_device_stats()
            memory_used = device_stats["global_used"]
            memory_total = device_stats["global_total"]
            tools.write_dict_summaries(writer, "memory", device_stats, tot_n_samples)

        return memory_used, memory_total, steps_per_sec

    def log_data(
        self, writer: Writer, it: int, loader: torch.utils.data.DataLoader, split: str
    ) -> str:
        tot_n_samples = self.current_it

        name = f"{split}_data"

        if hasattr(loader.dataset, "stats"):
            data_metrics, data_figures = loader.dataset.stats()
            tools.write_dict_summaries(writer, name, data_metrics, tot_n_samples)
            tools.write_image_summaries(writer, name, data_figures, tot_n_samples)

        writer.add_scalar(f"{name}/num_batches", len(loader), tot_n_samples)
        writer.add_scalar(
            f"{name}/batch_size", get_batch_size(loader) * self.num_gpus, tot_n_samples
        )
        writer.add_scalar(
            f"{name}/num_samples",
            len(loader) * get_batch_size(loader) * self.num_gpus,
            tot_n_samples,
        )

    def log_data_stats(
        self, writer: Writer, it: int, scene_num_samples: dict[str, int], split: str
    ) -> None:
        if len(scene_num_samples) == 0:
            return

        tot_n_samples = self.current_it

        writer.add_scalar(f"{split}_data/num_scenes", len(scene_num_samples), tot_n_samples)
        writer.add_scalar(
            f"{split}_data/avg_frames_per_scene",
            np.mean(list(scene_num_samples.values())),
            tot_n_samples,
        )
        writer.add_scalar(
            f"{split}_data/min_frames_per_scene",
            np.min(list(scene_num_samples.values())),
            tot_n_samples,
        )

    # ------------------------------------------------------------------------
    # Step functions (train, eval, visualize, ...)
    # ------------------------------------------------------------------------

    def train_step(
        self, data: Batch, do_update: bool = True, log_grad_norm: bool = False
    ) -> tuple[Predictions, LossMetrics]:
        with torch.autocast(device_type=self.device.type, enabled=self.use_mp, dtype=self.dtype):
            data = mappings.batch_to_device(data, self.device, non_blocking=True)
            torch.cuda.current_stream().synchronize()  # KEEP FOR DEBUGGING TIMING ISSUES
            self.step_timer.measure("to_device")

            pred = self.model(data)
            self.step_timer.measure("forward")

            losses, metrics = self.model.loss_metrics(pred, data)
            if self.conf.get("compose_loss", None) is not None:
                losses["total"] = compose_loss({**metrics, **losses}, self.conf.compose_loss)
            loss = torch.mean(losses["total"])
            loss = loss / self.conf.gradient_accumulation_steps
            loss_metrics = {**metrics, **{"loss/" + k: v for k, v in losses.items()}}
            self.step_timer.measure("loss_fn")

            for k, v in loss_metrics.items():
                loss_metrics[k] = v.detach()
            self.step_timer.measure("loss_fn_sync")

            do_backward = loss.requires_grad
            self.step_timer.measure("backward_check")

            if do_backward:
                self.scaler.scale(loss).backward()
                self.step_timer.measure("backward")

                if self.conf.detect_anomaly:
                    # Check for params without any gradient which causes
                    # problems in distributed training with checkpointing
                    detected_anomaly = False
                    for name, param in self.model.named_parameters():
                        if param.grad is None and param.requires_grad:
                            logger.warning(f"param {name} has no gradient.")
                            detected_anomaly = True
                    if detected_anomaly:
                        raise RuntimeError("Detected anomaly in training.")

                if do_update:
                    self.scaler.unscale_(self.optimizer)
                    if self.conf.get("clip_grad", None):
                        try:
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(),
                                max_norm=self.conf.clip_grad,
                                error_if_nonfinite=True,
                            )
                            if log_grad_norm:
                                loss_metrics["l2/grad_norm"] = torch.Tensor(
                                    [tools.grad_norm(self.model.parameters())]
                                )
                            self.scaler.step(self.optimizer)
                        except RuntimeError:
                            if self.conf.debug_nan:
                                self._dump_nan_debug(data, pred, loss_metrics)
                            scene = data.get("name", data.get("scene", "unknown"))
                            logger.warning(
                                f"NaN detected in gradients. Skipping iteration. " f"scene={scene}"
                            )
                        self.scaler.update()
                    else:
                        if log_grad_norm:
                            loss_metrics["l2/grad_norm"] = torch.Tensor(
                                [tools.grad_norm(self.model.parameters())]
                            )
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    self.optimizer.zero_grad()

                self.step_timer.measure("step")
                if not self.conf.lr_schedule.on_epoch:
                    self.learning_rate_step()
                self._step_query_ratio_schedule()
                self._step_view_schedule()
            else:
                self.warn("Skip iteration due to detach.")

        # detach to save memory
        pred = mappings.batch_to_device(pred, self.device, detach=True, non_blocking=True)
        loss_metrics = mappings.batch_to_device(
            loss_metrics, self.device, detach=True, non_blocking=True
        )
        return pred, loss_metrics

    def eval_step(self, data: Batch) -> tuple[Predictions, LossMetrics]:
        raise NotImplementedError()

    def visualize(self, data: Batch):
        raise NotImplementedError()

    # ------------------------------------------------------------------------
    # Main loops (train_epoch, eval_loop, test_loop)
    # ------------------------------------------------------------------------

    def train_epoch(
        self,
        output_dir: Path,
        dataloader: torch.utils.data.DataLoader,
        writer: Writer,
        max_iters: int | None = None,
    ):
        if self.rank == 0:
            self.log_data(writer, 0, dataloader, "training")

        do_profile = self.conf.profile
        if self.conf.profile_every_epoch is not None:
            do_profile = do_profile and ((self.epoch % self.conf.profile_every_epoch) == 0)
        else:
            do_profile = do_profile and (self.epoch == 0)
        profiler = self.construct_profiler(output_dir) if do_profile else None

        # Setup epoch
        _dist = self.distributed
        train_loss_metrics = collections.defaultdict(lambda: AverageMetric(distributed=_dist))
        pr_metrics = collections.defaultdict(lambda: PRMetric(distributed=_dist))
        self.step_timer.hard_reset()
        self.optimizer.zero_grad()
        scene_num_samples = collections.defaultdict(int)

        self.step_in_epoch = 0
        self.output_dir = output_dir

        train_iter = iter(dataloader)
        for it in range(len(dataloader)):
            self.step_in_epoch = it
            if self.should_stop(max_iters=max_iters):  # Check for early stopping
                break

            try:
                data = next(train_iter)
                local_done = torch.zeros(1, device=self.device, dtype=torch.int32)
            except StopIteration:
                local_done = torch.ones(1, device=self.device, dtype=torch.int32)

            if self.distributed:
                dist.all_reduce(local_done, op=dist.ReduceOp.MAX)

            if local_done.item():
                self.info("Epoch ended (dataloader exhausted on at least one rank).")
                break

            if "scene" in data:
                for scene_id in data["scene"]:
                    scene_num_samples[scene_id] += 1

            if self.rank == 0 and it == 0 and self.epoch == 0:
                # Log a single batch of data
                mappings.print_summary(data)
            self.step_timer.measure("data")
            nominal_bs = get_batch_size(
                self._train_dataset if hasattr(self, "_train_dataset") else dataloader
            )
            actual_bs = _infer_batch_size(data, nominal_bs)
            self.tot_n_samples += actual_bs * self.num_gpus
            self.tot_it += 1

            self.model.train()

            # Perform gradient accumulation
            do_update = ((it + 1) % self.conf.gradient_accumulation_steps) == 0

            do_log = it % self.conf.log_every_iter == 0
            pred, loss_metrics = self.train_step(data, do_update=do_update, log_grad_norm=do_log)
            if pred is None:
                continue  # skip iteration due to NaN

            # All ranks accumulate (distributed compute() needs all ranks)
            for k, val in loss_metrics.items():
                train_loss_metrics[k].update(val)
            for k, labels_preds in self.model.pr_metrics(pred, data).items():
                pr_metrics[k].update(*labels_preds)

            # Run profiler (stack trace, ...)
            if profiler is not None:
                profiler.step()

            # Record memory usage
            if self.conf.record_memory:
                self.record_memory(output_dir, it)

            # Log training metrics (loss, ...) and hardware usage
            if do_log:
                # All ranks must call compute() for distributed allreduce
                computed_metrics = {k: v.compute() for k, v in train_loss_metrics.items()}
                computed_metrics.update({k: v.compute() for k, v in pr_metrics.items()})
                train_loss_metrics.clear()
                pr_metrics.clear()

                if self.rank == 0:
                    mem_used, mem_total, its_per_sec = self.log_time_and_memory(
                        writer, it, get_batch_size(dataloader)
                    )
                    self.log_train(
                        writer,
                        it,
                        computed_metrics,
                        memory_used=mem_used,
                        memory_total=mem_total,
                        steps_per_sec=its_per_sec,
                    )
                    self.log_data_stats(writer, it, scene_num_samples, split="training")

            # Make plots of training steps
            if self.conf.plot_every_iter is not None:
                if it % self.conf.plot_every_iter == 0 and self.rank == 0:
                    with torch.no_grad():
                        figures = self.model.visualize(pred, data)

                    tools.write_image_summaries(writer, "training", figures, self.current_it)
                    del figures

            # Log gradients
            if self.conf.log_grad_every_iter is not None:
                raise NotImplementedError()

            del pred, data, loss_metrics
            if it == 0:
                torch.cuda.empty_cache()  # should be cleared at the first iter

            self.step_timer.measure("end")
            self.step_timer.reset()
        self.optimizer.zero_grad()

        del train_loss_metrics, train_iter
        gc.collect()
        if self.distributed:
            dist.barrier()

    def eval_loop(
        self,
        output_dir: Path,
        loader: torch.utils.data.DataLoader,
        max_iters: int | None = None,
    ):
        """Run evaluation loop."""
        self.model.eval()
        with torch.no_grad():
            with tools.fork_rng(seed=self.conf.seed):
                results, pr_metrics, figures = run_evaluation(
                    self.model,
                    loader,
                    self.device,
                    self.conf,
                    self.rank,
                    pbar=(self.rank == 0),
                    max_iters=max_iters,
                    compose_loss_str=self.conf.get("compose_loss", None),
                )
        return results, pr_metrics, figures

    @torch.compiler.set_stance("force_eager")
    def test_loop(
        self,
        output_dir: Path,
        benchmark_name: str,
        benchmark_conf: str,
        writer: SummaryWriter | None = None,
    ):
        """Interface for test loop."""
        logger.info(f"Running eval on {benchmark_name}")
        model = self.sequential_model()  # no DDP
        self.info("Configuration: \n%s", OmegaConf.to_yaml(benchmark_conf))
        eval_dir = output_dir / f"test_{self.epoch}" / benchmark_name
        with tools.fork_rng(seed=self.conf.seed):
            summaries, figures, _ = eval.run_benchmark(
                benchmark_name,
                benchmark_conf,
                eval_dir,
                model.eval(),
            )
        # Create symlink to eval_dir at head
        symlink_dir = output_dir / benchmark_name
        symlink_dir.unlink(missing_ok=True)
        symlink_dir.symlink_to(eval_dir)
        str_summaries = [f"{k} {v:.3E}" for k, v in summaries.items() if isinstance(v, float)]
        logger.info(f'[{benchmark_name}] {{{", ".join(str_summaries)}}}')
        if writer is not None:
            step = self.current_it
            tools.write_dict_summaries(writer, f"test_{benchmark_name}", summaries, step)
            tools.write_image_summaries(writer, f"test_{benchmark_name}", figures, step)
        del figures
        return summaries

    def run_all_benchmarks(self, output_dir: Path, writer: Writer = None, force: bool = False):
        epoch = 0 if force else self.epoch
        for bench_name, (bench_conf, every_epoch) in self.benchmarks.items():
            if epoch % every_epoch == 0 and self.rank == 0:
                # TODO: Make benchmarks distributed!
                self.test_loop(output_dir, bench_name, bench_conf, writer)
            if self.distributed:
                dist.barrier()

    def run_eval(
        self,
        output_dir: Path,
        val_dataset: datasets.BaseDataset,
        writer: Writer = None,
        max_iters: int | None = None,
    ) -> tuple[Any, ...]:
        eval_loader = val_dataset.get_loader(
            distributed=self.distributed,
            pinned=True,
        )
        if self.rank == 0 and writer is not None:
            self.log_data(writer, 0, eval_loader, "eval")
        self.info(f"Evaluation loader has {len(eval_loader)} batches")
        eval_results = self.eval_loop(output_dir, eval_loader, max_iters=max_iters)
        del eval_loader
        if self.rank == 0 and writer is not None:
            self.log_eval(writer, 0, eval_results)
        # Free eval figures/tensors before returning (only keep scalar results)
        results = eval_results[0]
        del eval_results
        gc.collect()
        plt.close("all")
        return results, None, None

    # ------------------------------------------------------------------------
    # Run full training on dataset (train multiple epochs + validation + test)
    # ------------------------------------------------------------------------

    def train_loop(
        self,
        output_dir: Path,
        train_dataset: datasets.BaseDataset,
        val_dataset: datasets.BaseDataset | None = None,
        writer: Writer = None,
    ):
        """The main function."""
        self._train_dataset = train_dataset
        self._val_dataset = val_dataset

        # Initialize writer
        full_conf = OmegaConf.create(
            {"data": train_dataset.conf, "model": self.model_conf, "train": self.conf}
        )

        if writer is None:
            writer = self.get_writer(output_dir, full_conf)

        if self.conf.get("eval_init", False) and val_dataset is not None:
            self.run_eval(output_dir, val_dataset, writer)
            self.run_all_benchmarks(output_dir, writer, force=True)

        # Cache view schedule params
        self._view_schedule_frac = train_dataset.conf.get("view_schedule", 0.0)
        self._view_schedule_range = list(train_dataset.image_num_range)

        # Apply view schedule before first loader so epoch 0 workers get correct range
        self._step_view_schedule()

        best_eval = None

        # Start Loop
        while not self.should_stop():
            self.info(f"Starting epoch {self.epoch}")
            self.epoch_timer.start_epoch()

            # Re-seed epoch
            tools.set_seed(self.conf.seed + self.epoch)

            # Recreate loader each epoch so forkserver workers get updated state
            train_dataset.set_epoch(self.epoch)
            train_loader = train_dataset.get_loader(
                distributed=self.distributed,
                pinned=True,
            )

            if self.conf.lr_schedule.on_epoch and self.epoch > 0:
                self.learning_rate_step(verbose=True)

            self.info("Start training")
            self.train_epoch(output_dir, train_loader, writer)
            self.epoch_timer.measure("train")

            self.epoch += 1

            # Eval before checkpoint so timeout during eval doesn't skip it on resume
            eval_results = None
            if self.should_evaluate() and val_dataset is not None:
                eval_results, _, _ = self.run_eval(output_dir, val_dataset, writer)
                self.epoch_timer.measure("eval")

            # Checkpointing with eval results and best tracking
            best_eval = self.save_checkpoint(
                output_dir, full_conf, results=eval_results, best_eval=best_eval
            )
            self.epoch_timer.measure("ckpt")

            # Run test loops
            self.run_all_benchmarks(output_dir, writer)
            self.epoch_timer.measure("bench")

            # Log epoch summary and finalize
            self.epoch_timer.log_epoch_summary(self.epoch - 1)
            self.epoch_timer.end_epoch()

            if self.distributed:
                dist.barrier()  # Sync before next epoch
            del train_loader
            gc.collect()
            plt.close("all")

        # Final evals
        if val_dataset is not None:
            self.run_eval(output_dir, val_dataset, writer)
        self.run_all_benchmarks(output_dir, writer, force=True)

        if writer is not None:
            writer.close()


def scale_by_device_count(
    data_conf: DictConfig, num_gpus: int, batch_size_per_gpu: bool | None = None
) -> DictConfig:
    """Scale data conf by device count (Maybe)."""
    batch_size_per_gpu = (
        batch_size_per_gpu
        if batch_size_per_gpu is not None
        else data_conf.get("batch_size_per_gpu", False)
    )
    # adjust batch size and num of workers since these are per GPU
    if "batch_size" in data_conf and not batch_size_per_gpu:
        data_conf.batch_size = int(data_conf.batch_size / num_gpus)

    logger.info(
        f"Batch size: global={data_conf.batch_size * num_gpus}, per-device={data_conf.batch_size}"
    )
    if "train_batch_size" in data_conf and not batch_size_per_gpu:
        data_conf.train_batch_size = int(data_conf.train_batch_size / num_gpus)

    return data_conf


def launch_training(output_dir: Path, conf: DictConfig, device: torch.device):
    tools.set_seed(conf.train.seed)
    data_conf = scale_by_device_count(conf.data, conf.train.num_devices or 1)

    # Create separate dataset instances for train and val splits
    DatasetClass = datasets.get_dataset(data_conf.name)
    train_dataset = DatasetClass(data_conf, split="train")
    val_dataset = DatasetClass(data_conf, split="val")

    if conf.train.get("reload_model"):
        assert conf.train.load_experiment is not None
        pretrain_dir = settings.TRAINING_PATH / conf.train.load_experiment
        logger.info(f"Finetuning: Loading model config from {pretrain_dir}.")
        pretrain_conf = OmegaConf.load(pretrain_dir / "config.yaml")
        conf.model = OmegaConf.merge(pretrain_conf.model, conf.model)
    model = models.get_model(conf.model.name)(conf.model).to(device)
    if conf.get("lazy_init", True):
        logger.info("Running dummy forward pass to initialize lazy modules.")
        dummy_batch = train_dataset.get_dummy_batch()
        dummy_batch = mappings.batch_to_device(dummy_batch, device, non_blocking=False)
        with torch.no_grad():
            model(dummy_batch)
        del dummy_batch
        logger.info("Dummy forward pass completed.")
    trainer = Trainer.init(conf.train, model, device=device)

    # Register benchmarks (e.g. MegaDepth1500)
    for bench in conf.train.get("run_benchmarks", ()):
        bench_name, every_epoch = (bench, None) if isinstance(bench, str) else bench
        eval.get_benchmark(bench_name)  # Check if benchmark exists
        bench_conf = {} if conf.get("benchmarks") is None else conf.benchmarks.get(bench_name, {})
        bench_conf = OmegaConf.merge({"eval": conf.get("eval", {})}, OmegaConf.create(bench_conf))
        trainer.register_benchmark(bench_name, bench_conf, every_epoch=every_epoch)
    # Maybe load experiment
    trainer.maybe_load_checkpoint()

    # Run actual training loop
    trainer.train_loop(output_dir, train_dataset, val_dataset)
