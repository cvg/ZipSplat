"""Base class for datasets.

Handles configuration merging and defines the interface that the trainer expects.
Subclasses implement _init(), __iter__() (for IterableDataset), and get_loader().

Author: Alexander Veicht
"""

from abc import ABCMeta, abstractmethod

from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from splatfactory import get_logger

logger = get_logger(__name__)


class BaseDataset(metaclass=ABCMeta):
    """Abstract base for all datasets.

    Subclasses must implement:
        _init(conf, split): setup for a specific split
        get_loader(**kwargs): return a DataLoader
    """

    base_default_conf = {
        "name": "???",
        "seed": 0,
    }
    default_conf = {}
    strict_conf = False

    def __init__(self, conf, split="train", **kwargs):
        """Merge configs, resolve, and call _init."""
        default_conf = OmegaConf.merge(
            OmegaConf.create(self.base_default_conf), OmegaConf.create(self.default_conf)
        )
        OmegaConf.set_struct(default_conf, self.strict_conf)
        if isinstance(conf, dict):
            conf = OmegaConf.create(conf)
        self.conf = OmegaConf.merge(default_conf, conf)
        OmegaConf.resolve(self.conf)
        OmegaConf.set_readonly(self.conf, True)
        self.seed = self.conf.seed
        self.split = split
        self._init(self.conf, split=split, **kwargs)
        logger.info(f"Creating dataset {self.__class__.__name__}(split={split}, seed={self.seed})")

    @abstractmethod
    def _init(self, conf, split="train", **kwargs):
        """Initialize dataset state for the given split."""
        raise NotImplementedError

    @abstractmethod
    def get_loader(self, **kwargs) -> DataLoader:
        """Create and return a DataLoader for this dataset."""
        raise NotImplementedError

    def get_dummy_batch(self, **kwargs):
        """Return a single batch for lazy model initialization."""
        loader = self.get_loader(num_workers=0)
        batch = next(iter(loader))
        del loader
        return batch

    def set_epoch(self, epoch: int):
        """Update epoch. Takes effect at next iter() call (worker re-fork)."""
        pass

    def update_image_num_range(self, new_min: int | None = None, new_max: int | None = None):
        """Update the view count range. Takes effect at next iter() call."""
        pass

    @property
    def image_num_range(self) -> list[int]:
        """Current [min, max] view count range."""
        return [1, 1]

    @property
    def max_img_per_gpu(self) -> int:
        """Maximum images (context views) per GPU."""
        return 1
