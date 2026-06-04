"""Distributed training utilities for torchrun.

Adapted from gluefactory (https://github.com/cvg/glue-factory), Apache-2.0.
"""

import os
from dataclasses import dataclass


@dataclass
class DistributedEnv:
    """Container for distributed environment variables."""

    rank: int
    local_rank: int
    world_size: int
    local_world_size: int
    master_addr: str
    master_port: str

    @property
    def is_distributed(self) -> bool:
        """Check if running in distributed mode."""
        return self.world_size > 1

    @property
    def is_main_process(self) -> bool:
        """Check if this is rank 0."""
        return self.rank == 0

    @property
    def num_nodes(self) -> int:
        """Calculate number of nodes."""
        return self.world_size // self.local_world_size if self.local_world_size > 0 else 1


def get_distributed_env() -> DistributedEnv:
    """Get distributed environment from environment variables set by torchrun."""
    return DistributedEnv(
        rank=int(os.environ.get("RANK", "0")),
        local_rank=int(os.environ.get("LOCAL_RANK", "0")),
        world_size=int(os.environ.get("WORLD_SIZE", "1")),
        local_world_size=int(os.environ.get("LOCAL_WORLD_SIZE", "1")),
        master_addr=os.environ.get("MASTER_ADDR", "localhost"),
        master_port=os.environ.get("MASTER_PORT", "29500"),
    )


def is_torchrun_launch() -> bool:
    """Check if running under torchrun by detecting environment variables."""
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ
