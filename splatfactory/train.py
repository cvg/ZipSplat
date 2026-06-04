"""
Simple training launcher.

Author: Paul-Edouard Sarlin (skydes), Philipp Lindenberger (Phil26AT)

Adapted from gluefactory (https://github.com/cvg/glue-factory), Apache-2.0.
"""

import argparse
import datetime
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Sequence

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from omegaconf import DictConfig, OmegaConf

from splatfactory import __module_name__, get_logger, logger, settings, trainer
from splatfactory.utils import experiments, stdout_capturing, tools
from splatfactory.utils.distributed import get_distributed_env, is_torchrun_launch

# Logger for init messages that should appear on all ranks
init_logger = get_logger(__name__, level=logging.INFO)

# --- Environment Variable Setup for Performance and Debugging --- (from VGGT)
# Helps with memory fragmentation in PyTorch's memory allocator.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# Specifies the threading layer for MKL, can prevent hangs in some environments.
# os.environ["MKL_THREADING_LAYER"] = "GNU"
# Provides full Hydra stack traces on error for easier debugging.
# os.environ["HYDRA_FULL_ERROR"] = "1"
# Enables asynchronous error handling for NCCL, which can prevent hangs.
os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"


mp.set_sharing_strategy("file_system")


def init_process_tcp(env) -> int:
    """Initialize distributed process group using TCP.

    Args:
        env: Distributed environment with rank, local_rank, world_size, etc.

    Returns:
        Device ID (local_rank) to use for this process.
    """
    assert torch.cuda.is_available(), "Distributed training requires CUDA"

    torch.distributed.init_process_group(
        backend="nccl",
        init_method=f"tcp://{env.master_addr}:{env.master_port}",
        world_size=env.world_size,
        rank=env.rank,
        device_id=torch.device(f"cuda:{env.local_rank}"),
        timeout=datetime.timedelta(minutes=120),  # Increased for multi-node with slow interconnect
    )

    init_logger.info(
        f"[Rank {env.rank}] Initializing distributed process group: "
        f"rank={env.rank}/{env.world_size}, local_rank={env.local_rank}, "
        f"num_nodes={env.num_nodes}, master={env.master_addr}:{env.master_port}"
    )

    dist.barrier()

    if env.rank == 0:
        logger.info("[OK] Distributed process group initialized successfully.")

    # CRITICAL: Use local_rank for device assignment, not global rank!
    device = env.local_rank
    torch.cuda.set_device(device)
    return device


def create_training_dir(experiment_name: str, args: argparse.Namespace) -> Path:
    """Create the training output directory."""
    output_dir = settings.TRAINING_PATH / experiment_name

    env = get_distributed_env()
    if env.is_distributed and not env.is_main_process:
        return output_dir

    if args.ablate:
        subdirs = [int(p.stem) for p in output_dir.glob("*/") if p.stem.isdigit()]
        ablate_id = max(subdirs, default=-1) + 1
        output_dir = output_dir / str(ablate_id)
        logger.info(f"Creating ablation folder {output_dir}")
    # Setup output directory
    exist_ok = args.restore or args.overwrite or args.clean
    if output_dir.exists() and not exist_ok:
        raise FileExistsError(
            f"Output directory {output_dir} already exists. "
            "Use --restore to continue training or --overwrite to delete it."
        )
    if output_dir.exists() and args.clean:
        logger.info(f"Cleaning output directory {output_dir}")
        shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(exist_ok=exist_ok, parents=True)
    logger.info(f"Output directory: {output_dir}")
    return output_dir


def compose_cli_config(
    output_dir: Path, args: argparse.Namespace, save_config: bool = True
) -> DictConfig:
    """Compose configuration from command-line arguments."""
    if save_config:
        OmegaConf.save(OmegaConf.from_cli(args.dotlist), str(output_dir / "cli_config.yaml"))

    if args.conf:
        # Route Hydra defaults overrides (e.g. data=composed_252) to hydra.compose(),
        # keep regular overrides for OmegaConf merge (preserves backward compat).
        conf_dir = experiments.parse_config_path(args.conf).parent
        defaults_groups = {p.name for p in conf_dir.iterdir() if p.is_dir()}

        def _is_hydra_override(arg):
            key = arg.split("=", 1)[0]
            return key in defaults_groups or "/" in key

        hydra_overrides = [a for a in args.dotlist if _is_hydra_override(a)]
        regular_dotlist = [a for a in args.dotlist if not _is_hydra_override(a)]

        conf_path, raw_conf = experiments.compose_config(
            args.conf,
            overrides=hydra_overrides or None,
            sweep_idx=args.sweep_idx,
            resolve=False,
        )
        OmegaConf.set_struct(raw_conf, args.strict)
        conf = OmegaConf.merge(raw_conf, OmegaConf.from_cli(regular_dotlist))
        OmegaConf.resolve(conf)
        if save_config:
            shutil.copy(conf_path, output_dir / "raw_config.yaml")
    else:
        conf = OmegaConf.from_cli(args.dotlist)
    if args.restore:
        restore_conf = OmegaConf.load(output_dir / "config.yaml")
        conf = OmegaConf.merge(restore_conf, conf)
        conf.train.load_experiment = args.experiment
        conf.train.load_state = True
    else:
        if conf.train.seed is None:
            conf.train.seed = torch.initial_seed() & (2**32 - 1)

    if save_config:
        OmegaConf.save(conf, str(output_dir / "config.yaml"))
    return conf


def main_worker(conf, output_dir, args):
    """Main worker function for training."""
    env = get_distributed_env()
    distributed = env.is_distributed

    if env.is_main_process and not args.quiet:
        logger.info("Starting training with configuration:\n%s", OmegaConf.to_yaml(conf))

    if distributed:
        # device = init_process_tcp(env)
        device = torch.device(f"cuda:{env.local_rank}")
    else:
        device = tools.get_device()

    if env.is_main_process:
        with stdout_capturing.capture_outputs(
            output_dir / "log.log", cleanup_interval=args.cleanup_interval
        ):
            res = trainer.launch_training(output_dir, conf, device)
    else:
        res = trainer.launch_training(output_dir, conf, device)

    if distributed:
        dist.destroy_process_group()
    return res


def save_code_snapshot(
    output_dir: Path,
    extra_module_names: Sequence[str] = (),
    compression: str | None = "zip",
):
    """Create a snapshot of the codebase."""
    for module in [__module_name__] + list(extra_module_names):
        mod_dir = Path(__import__(str(module)).__file__).parent
        if compression:
            shutil.make_archive(output_dir / module, compression, mod_dir)
        else:
            shutil.copytree(mod_dir, output_dir / module, dirs_exist_ok=True)


def parse_args():
    """Parse command line arguments and return them."""
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=str)
    default_config_names = experiments.list_configs(Path(__file__).parent / "configs")
    parser.add_argument(
        "--conf",
        type=str,
        help=f"Configuration path (.yaml) or one of: {default_config_names}",
    )
    parser.add_argument(
        "--cleanup_interval",
        default=120,  # Cleanup log files every 120 seconds.
        type=int,
        help="Interval in seconds to cleanup log files",
    )
    parser.add_argument("--restore", action="store_true", help="Restore from previous experiment")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing experiment directory",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete the output directory if it exists",
    )
    parser.add_argument(
        "--ablate",
        action="store_true",
        help="Create an ablation folder (/XID) that increments on each run",
    )
    parser.add_argument(
        "--compress_snapshot",
        "--cs",
        type=str,
        default="zip",
    )
    parser.add_argument(
        "--sweep_idx", "-sid", type=int, default=None, help="Index of the sweep run"
    )
    parser.add_argument("--quiet", action="store_true", help="Mute some logging")
    parser.add_argument("--distributed", action="store_true", help="Run in distributed mode")
    parser.add_argument("--strict", action="store_true", help="Strict config merge")
    parser.add_argument("dotlist", nargs="*")
    args = parser.parse_intermixed_args()
    return args


def setup_experiment_distributed(args, env):
    """
    Setup experiment directory and config for distributed training.

    In multi-node SLURM jobs, only the first task (SLURM_PROCID=0) creates
    the directory and saves code. Other tasks wait for setup to complete.

    Returns:
        tuple: (output_dir, conf) where output_dir is Path and conf is DictConfig
    """

    slurm_procid = int(os.environ.get("SLURM_PROCID", "0"))
    slurm_nodeid = int(os.environ.get("SLURM_NODEID", "0"))
    is_first_task = slurm_procid == 0

    logger.info(
        f"Setup: Task PROCID={slurm_procid}, NODEID={slurm_nodeid}, "
        f"is_first_task={is_first_task}"
    )

    if is_first_task:
        # Only first task creates directory and saves code
        logger.info("Task 0: Setting up experiment directory and config")
        output_dir = create_training_dir(args.experiment, args)
        conf = compose_cli_config(output_dir, args)

        # Detect GPU count
        conf.train.num_devices = conf.train.get("num_devices", 0)
        if conf.train.num_devices < 1:
            conf.train.num_devices = torch.cuda.device_count()
            assert conf.train.num_devices > 0, "No GPUs found for distributed training"

        # Save code snapshot
        save_code_snapshot(
            output_dir,
            conf.train.get("submodules", ()),
            compression=args.compress_snapshot,
        )

        # Signal to other tasks that setup is done
        setup_done_file = output_dir / ".setup_done"
        setup_done_file.touch()
        logger.info(f"Task 0: Setup complete, created {output_dir}")
    else:
        # Other tasks wait for first task to finish setup
        output_dir = settings.TRAINING_PATH / args.experiment
        setup_done_file = output_dir / ".setup_done"

        logger.info(f"Task {slurm_procid}: Waiting for task 0 to complete setup...")
        max_wait = 300  # 5 minutes
        waited = 0
        while not setup_done_file.exists() and waited < max_wait:
            time.sleep(1)
            waited += 1

        if not setup_done_file.exists():
            raise RuntimeError(
                f"Task {slurm_procid}: Timed out waiting for task 0 to setup directory"
            )

        logger.info(f"Task {slurm_procid}: Setup complete, loading config")

        # Load config created by first task
        conf = compose_cli_config(output_dir, args)

        # Each task independently detects its GPU count
        conf.train.num_devices = conf.train.get("num_devices", 0)
        if conf.train.num_devices < 1:
            conf.train.num_devices = torch.cuda.device_count()
            assert conf.train.num_devices > 0, "No GPUs found for distributed training"

    # Pass output_dir to spawned torchrun processes
    os.environ["SPLATFACTORY_OUTPUT_DIR"] = str(output_dir)

    return output_dir, conf


def build_torchrun_command(args, conf):
    """Build torchrun command for distributed training.

    Args:
        args: Parsed command-line arguments
        conf: Resolved configuration

    Returns:
        List of command arguments for os.execvp()
    """

    # Get distributed environment settings
    # CRITICAL: Read from SLURM first, fall back to config
    num_nodes = int(os.environ.get("SLURM_NNODES", conf.train.get("num_nodes", 1)))
    node_rank = int(os.environ.get("SLURM_NODEID", "0"))
    master_addr = os.environ.get("MASTER_ADDR", "localhost")
    master_port = os.environ.get("MASTER_PORT", "29500")

    logger.info(
        f"Torchrun config: num_nodes={num_nodes}, node_rank={node_rank}, "
        f"nproc_per_node={conf.train.num_devices}, "
        f"master={master_addr}:{master_port}"
    )

    # Build base torchrun command
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node",
        str(conf.train.num_devices),
        "--nnodes",
        str(num_nodes),
        "--node_rank",
        str(node_rank),
        "--master_addr",
        master_addr,
        "--master_port",
        master_port,
        "-m",
        "splatfactory.train",
    ]

    # Forward all original arguments except --distributed (already handled by torchrun)
    # This is more robust than manually reconstructing - won't break when adding new args
    # Note: experiment name is already in sys.argv[1], so no need to add it separately
    for arg in sys.argv[1:]:
        if arg != "--distributed":
            cmd.append(arg)

    return cmd


if __name__ == "__main__":
    # Initialize distributed
    env = get_distributed_env()
    device = init_process_tcp(env)

    args = parse_args()

    if env.is_main_process:
        logger.info("Distributed setup complete, setting up experiment directory and config")

    # Load config (already created by parent process if distributed)
    output_dir = create_training_dir(args.experiment, args)
    dist.barrier()

    conf = compose_cli_config(output_dir, args, save_config=env.is_main_process)

    if env.is_main_process:
        logger.info(f"Experiment directory ready: {output_dir}")

    # Update num_devices from environment if distributed
    if env.is_distributed:
        logger.info(f"Distributed: Setting num_devices to world_size={env.world_size}")
        conf.train.num_devices = env.world_size
    else:
        conf.train.num_devices = conf.train.get("num_devices", 0)
        logger.info(f"Non-distributed: num_devices={conf.train.num_devices}")

    logger.info(f"Launching main_worker (rank {env.rank})")
    main_worker(conf, output_dir, args)
