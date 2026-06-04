"""Multi-view IterableDataset backed by tar shards.

Streams scenes from tar shards with worker-local batching.
Each worker independently: splits shards -> shuffles -> streams scenes ->
samples views -> decodes -> normalizes -> collates -> yields batches.

DataLoader(batch_size=None) passes batches through unchanged.

Author: Alexander Veicht
"""

import os
import random
from collections.abc import Iterator
from itertools import chain
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset

from splatfactory import get_logger
from splatfactory.datasets.augmentations import get_augmentation
from splatfactory.datasets.base_dataset import BaseDataset
from splatfactory.datasets.utils import io, workers
from splatfactory.datasets.view_sampler import get_view_sampler
from splatfactory.geometry import Camera, Pose
from splatfactory.geometry.scale import compute_scene_scale
from splatfactory.utils import mappings
from splatfactory.utils.image import ImagePreprocessor
from splatfactory.utils.tools import StepTimer

logger = get_logger(__name__)


class MultiViewDataset(BaseDataset, IterableDataset):
    """Multi-view dataset that streams scenes from tar shards.

    Each DataLoader worker independently splits shards, samples view counts,
    assembles complete batches, and yields them via DataLoader(batch_size=None).
    """

    base_default_conf = {
        "name": "???",
        "dataset_name": "???",
        "dataset_dir": "???",
        "train_shard_dir": "${.dataset_dir}/train-shards",
        "test_shard_dir": "${.dataset_dir}/test-shards",
        "shard_glob": "shard-*.tar",
        "image_num_range": [2, 24],
        "aspect_ratio_range": [1.0, 1.0],
        "num_target_views": 4,
        "random_reference_view": True,
        "view_sampler": {"name": "bounded"},
        "augmentations": {"name": "identity"},
        "preprocessing": {},
        "use_relative_poses": True,
        "normalize_baseline": False,
        "near": 0.01,
        "far": 1000.0,
        "max_pose_jump_ratio": 10.0,  # 0 disables; otherwise skip if max/median consecutive step > ratio
        "train_batch_size": "???",
        "val_batch_size": 1,
        "test_batch_size": 1,
        "batch_size": 1,
        "num_workers": 8,
        "prefetch_factor": 2,
        "seed": 0,
        "max_samples": None,
    }
    default_conf = {}

    def _init(self, conf, split: str = "train") -> None:
        self.split = split
        self._epoch: int = 0
        self._image_num_range: list[int] = list(conf.image_num_range)
        self.step_timer = StepTimer()
        self.view_timer = StepTimer()

        shard_dir = Path(conf.train_shard_dir if split == "train" else conf.test_shard_dir)
        self.shard_paths: list[Path] = sorted(shard_dir.glob(conf.shard_glob))
        if not self.shard_paths:
            raise FileNotFoundError(f"No shards found in {shard_dir}")
        logger.info(f"[{split}] Found {len(self.shard_paths)} shards in {shard_dir}")

        self.view_sampler = get_view_sampler(conf.view_sampler.name)(conf.view_sampler, split=split)

        augmentation = get_augmentation(conf.augmentations) if split == "train" else None
        self.preprocessor = ImagePreprocessor(conf.preprocessing, augmentation=augmentation)

    # ---- BaseDataset interface -----------------------------------------------

    def get_loader(
        self, distributed: bool = False, pinned: bool = True, num_workers: int | None = None
    ) -> DataLoader:
        """Create a DataLoader that yields pre-collated batches."""
        if num_workers is None:
            num_workers = self._resolve_num_workers(distributed)
        if num_workers > len(self.shard_paths):
            logger.warning(
                f"num_workers ({num_workers}) > shards ({len(self.shard_paths)}), capping"
            )
            num_workers = len(self.shard_paths)
        return DataLoader(
            self,
            batch_size=None,
            num_workers=num_workers,
            pin_memory=pinned,
            prefetch_factor=self.conf.prefetch_factor if num_workers > 0 else None,
            persistent_workers=False,
            multiprocessing_context="forkserver" if num_workers > 0 else None,
        )

    def get_dummy_batch(self, **kwargs) -> dict:
        """Single batch for lazy model init."""
        return next(iter(self.get_loader(num_workers=0)))

    def set_epoch(self, epoch: int) -> None:
        """Update epoch - takes effect at next iter() call (worker re-fork)."""
        self._epoch = epoch

    def update_image_num_range(
        self, new_min: int | None = None, new_max: int | None = None
    ) -> None:
        """Update view count range - takes effect at next iter() call."""
        if new_min is not None:
            self._image_num_range[0] = int(new_min)
        if new_max is not None:
            self._image_num_range[1] = int(new_max)

    @property
    def image_num_range(self) -> list[int]:
        return self._image_num_range

    @property
    def max_img_per_gpu(self) -> int:
        return self.conf.get(self.split + "_batch_size", self.conf.batch_size)

    def __len__(self) -> int:
        """Approximate length. Eval: valid scenes from index; train: shard count."""
        if hasattr(self.view_sampler, "valid_scenes"):
            n = len(self.view_sampler.valid_scenes)
            if self.conf.max_samples is not None:
                n = min(n, self.conf.max_samples)
            return n
        return len(self.shard_paths)

    # ---- Iteration -----------------------------------------------------------
    def __iter__(self) -> Iterator[dict]:
        """Yield pre-collated batches. Each worker handles its own shard subset."""
        rank, world_size = workers.get_rank(), workers.get_world_size()
        wid, num_workers = workers.get_dataloader_worker_info()

        shards = self._split_shards(rank, world_size, wid, num_workers)
        logger.debug(
            f"[rank={rank}/{world_size}, worker={wid}/{num_workers}] "
            f"epoch={self._epoch}, {len(shards)}/{len(self.shard_paths)} shards, "
            f"views={self._image_num_range}"
        )
        if self.split == "train":
            rng = random.Random(self.seed + self._epoch + wid)
            rng.shuffle(shards)

        scene_stream = chain.from_iterable(io.iter_scenes_from_tar(str(s)) for s in shards)
        batch_rng = random.Random(self.seed + self._epoch + wid)
        is_train = self.split == "train"
        yielded = 0
        skipped = 0

        while True:
            vc, ar, bs = self._sample_batch_params(batch_rng, is_train)

            batch = []
            self.step_timer.reset()
            for raw_scene in scene_stream:
                self.step_timer.measure("tar_io")
                if not self._scene_ok(raw_scene, vc):
                    self.step_timer.reset()
                    continue
                try:
                    sample = self._process_scene(raw_scene, vc, ar)
                    batch.append(sample)
                except Exception as e:
                    logger.debug(f"Skipping scene {raw_scene.get('key', '?')}: {e}")
                    skipped += 1
                self.step_timer.reset()
                if len(batch) >= bs:
                    break

            if not batch:
                break

            yield workers.collate(batch)
            yielded += len(batch)
            if self.conf.max_samples is not None and yielded >= self.conf.max_samples:
                break

        if skipped > 0:
            logger.info(
                f"[worker={wid}] epoch={self._epoch}: skipped {skipped} scenes "
                f"({skipped / max(yielded + skipped, 1) * 100:.1f}%)"
            )

    # ---- Scene processing ----------------------------------------------------
    def _process_scene(
        self,
        raw_scene: dict,
        num_context_views: int,
        aspect_ratio: float = 1.0,
        *,
        view_sampler=None,
        preprocessor=None,
        dataset_name: str | None = None,
    ) -> dict:
        """Process a raw scene dict into a training sample.

        view sampling -> decode -> pose normalization -> scale normalization -> near/far.

        ComposedDataset passes child-specific view_sampler/preprocessor/dataset_name;
        single-dataset callers let them default to self.*.
        """
        if view_sampler is None:
            view_sampler = self.view_sampler
        if preprocessor is None:
            preprocessor = self.preprocessor
        if dataset_name is None:
            dataset_name = self.conf.dataset_name

        key = raw_scene["key"]
        meta = raw_scene["meta"]

        indices = view_sampler.sample(
            scene_id=key,
            num_context_views=num_context_views,
            num_target_views=self.conf.num_target_views,
            total_num_views=meta["num_views"],
        )

        if self.conf.max_pose_jump_ratio > 0:
            all_idx = sorted(indices["context_indices"] + indices["target_indices"])
            lo, hi = all_idx[0], all_idx[-1]
            if hi - lo >= 2:
                translations = raw_scene["poses"][lo : hi + 1, 9:12]
                steps = np.linalg.norm(np.diff(translations, axis=0), axis=1)
                median_step = np.median(steps)
                if median_step > 0 and steps.max() / median_step > self.conf.max_pose_jump_ratio:
                    raise ValueError(f"pose jump ratio {steps.max() / median_step:.0f}")

        if self.conf.random_reference_view and self.split == "train":
            np.random.shuffle(indices["context_indices"])

        context = mappings.stack_tree(
            [
                self._load_view(raw_scene, i, aspect_ratio, preprocessor)
                for i in indices["context_indices"]
            ]
        )
        target = mappings.stack_tree(
            [
                self._load_view(raw_scene, i, aspect_ratio, preprocessor)
                for i in indices["target_indices"]
            ]
        )
        self.step_timer.measure("decode")

        if self.conf.use_relative_poses:
            ref = context["pose"][0].inv()
            context["pose"] = ref @ context["pose"]
            target["pose"] = ref @ target["pose"]

        scale = compute_scene_scale(
            context["pose"],
            context["camera"],
            context["depth"],
            context["depth_mask"],
            num_context_views,
            self.conf.normalize_baseline,
        )
        inv_scale = 1.0 / scale
        context["pose"] = context["pose"].scale_translation(inv_scale)
        target["pose"] = target["pose"].scale_translation(inv_scale)
        for g in (context, target):
            g["depth"] = torch.where(g["depth_mask"], g["depth"] * inv_scale, g["depth"])

        near, far = self.conf.near / scale, self.conf.far / scale
        for g, n in (
            (context, len(indices["context_indices"])),
            (target, len(indices["target_indices"])),
        ):
            g["near"] = torch.full((n,), near)
            g["far"] = torch.full((n,), far)
        self.step_timer.measure("normalize")

        return {
            "name": f"{dataset_name}-{key}",
            "scene": f"{dataset_name}-{key}",
            "scene_index": 0,
            "context": context,
            "target": target,
            "overlap": indices["overlap_scores"],
        }

    # ---- View loading --------------------------------------------------------
    def _load_view(
        self, scene: dict, view_index: int, aspect_ratio: float = 1.0, preprocessor=None
    ) -> dict:
        """Decode and preprocess a single view from raw scene bytes."""
        if preprocessor is None:
            preprocessor = self.preprocessor
        self.view_timer.reset()
        image = io.decode_image(scene["images"][view_index])
        self.view_timer.measure("img_decode")

        camera = Camera(torch.as_tensor(scene["cameras"][view_index], dtype=torch.float32))
        pose = Pose(torch.as_tensor(scene["poses"][view_index], dtype=torch.float32))

        depth = self._decode_depth(scene, view_index)
        self.view_timer.measure("depth_decode")

        data = preprocessor(image, depth=depth, aspect_ratio=aspect_ratio)
        self.view_timer.measure("preprocess")

        camera = camera.compose_image_transform(data["transform"])
        hw = data["image"].shape[-2:]

        return {
            "image": data["image"],
            "camera": camera,
            "pose": pose,
            "depth": data.get("depth", torch.full(hw, io.INVALID_DEPTH)),
            "depth_mask": data.get("depth_mask", torch.zeros(hw, dtype=torch.bool)),
            "index": torch.tensor([view_index], dtype=torch.float32),
        }

    def _decode_depth(self, scene: dict, view_index: int):
        """Decode depth bytes to numpy float32. Returns None if unavailable."""
        if not scene["meta"].get("has_depth", False):
            return None
        if view_index not in scene.get("depths", {}):
            return None
        d_min, d_max = scene["depth_ranges"][view_index]
        return io.decode_depth(scene["depths"][view_index], d_min, d_max)

    # ---- Helpers -------------------------------------------------------------
    def _split_shards(
        self, rank: int, world_size: int, worker_id: int, num_workers: int
    ) -> list[Path]:
        """Assign shards to this worker via round-robin: first by rank, then by worker."""
        shards = self.shard_paths[rank::world_size]
        shards = shards[worker_id::num_workers]
        return list(shards)

    def _sample_batch_params(self, rng: random.Random, is_train: bool) -> tuple[int, float, int]:
        """Sample view count, aspect ratio, and batch size for one batch."""
        vc = rng.randint(self._image_num_range[0], self._image_num_range[1])
        ar = round(rng.uniform(*self.conf.aspect_ratio_range), 2)
        bs = max(1, self.max_img_per_gpu // vc)
        return vc, ar, bs

    def _scene_ok(self, raw_scene: dict, num_context_views: int, view_sampler=None) -> bool:
        """Check if a scene should be processed or skipped."""
        if view_sampler is None:
            view_sampler = self.view_sampler
        key = raw_scene.get("key")
        num_views = raw_scene.get("meta", {}).get("num_views", 0)

        if hasattr(view_sampler, "valid_scenes") and key not in view_sampler.valid_scenes:
            logger.debug(f"Scene {key} not in valid_scenes, skipping")
            return False

        if (
            num_context_views is not None
            and num_views < num_context_views + self.conf.num_target_views
        ):
            logger.debug(
                f"Scene {key} has {num_views} views, fewer than "
                f"context {num_context_views} + target {self.conf.num_target_views}, skipping"
            )
            return False

        min_required = view_sampler.min_required_images()
        if num_views < min_required:
            logger.debug(
                f"Scene {key} has {num_views} views, sampler {type(view_sampler).__name__} "
                f"requires >= {min_required}, skipping"
            )
            return False

        return True

    def _resolve_num_workers(self, distributed: bool) -> int:
        """Compute num_workers respecting CPU affinity and distributed training."""
        try:
            max_workers = len(os.sched_getaffinity(0))
        except AttributeError:
            max_workers = os.cpu_count() or 0

        if distributed:
            max_workers = max_workers // max(1, workers.get_world_size())

        num_workers = self.conf.get("num_workers", max_workers)
        if num_workers is None or num_workers < 0:
            num_workers = max_workers

        return min(num_workers, max_workers)
