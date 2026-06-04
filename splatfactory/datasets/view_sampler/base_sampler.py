"""Abstract base class defining the view-sampler interface for multi-view datasets.

Author: Alexander Veicht
"""

from abc import ABC, abstractmethod

from omegaconf import DictConfig, OmegaConf

from splatfactory import get_logger

logger = get_logger(__name__)


class BaseViewSampler(ABC):
    """Abstract base class for view samplers in multi-view datasets."""

    base_default_conf = {
        "name": "???",
        "warm_up_steps": 0,
        "is_overfitting": False,
    }
    default_conf = {}

    def __init__(
        self,
        conf: DictConfig,
        split: str,
    ) -> None:
        default_conf = OmegaConf.merge(
            OmegaConf.create(self.base_default_conf), OmegaConf.create(self.default_conf)
        )
        OmegaConf.set_struct(default_conf, False)
        if isinstance(conf, dict):
            conf = OmegaConf.create(conf)
        self.conf = OmegaConf.merge(default_conf, conf)
        OmegaConf.set_readonly(self.conf, True)
        logger.info(f"Creating view sampler {self.__class__.__name__}({conf=}, {split=})")

        self.split = split
        self.is_overfitting = self.conf.is_overfitting

        self._init(self.conf)

    @abstractmethod
    def _init(self, conf):
        """To be implemented by the child class."""
        raise NotImplementedError

    @abstractmethod
    def min_required_images(self) -> int:
        """Minimum number of images required in a scene for this sampler to work."""
        raise NotImplementedError

    def linear_schedule(self, initial: int, final: int) -> int:
        """Return final value (warm-up scheduling removed)."""
        return final

    @abstractmethod
    def sample(
        self,
        scene_id: str | None = None,
        num_context_views: int | None = None,
        num_target_views: int | None = None,
        total_num_views: int | None = None,
    ):
        """Returns a dict with keys: context_indices, target_indices, overlap_scores"""
        raise NotImplementedError

    @property
    def is_test(self) -> bool:
        return self.split == "test"

    @property
    def is_train(self) -> bool:
        return self.split == "train"

    @property
    def is_val(self) -> bool:
        return self.split == "val"

    @property
    def num_target_views(self) -> int:
        return self.conf.num_target_views

    @property
    def num_context_views(self) -> int:
        return self.conf.num_context_views
