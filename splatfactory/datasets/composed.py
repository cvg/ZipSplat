"""Composed dataset: several datasets mixed by weight.

Pools the shards of all children and oversamples the smaller ones so the scene draws match the
configured weights. Each child keeps its own view sampler, preprocessing and aspect ratios.
Adding a dataset to the mix is a config change.

Author: Alexander Veicht
"""

import json
import math
import random
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from omegaconf import OmegaConf

from splatfactory import get_logger
from splatfactory.datasets.augmentations import get_augmentation
from splatfactory.datasets.multi_view_dataset import MultiViewDataset, resolve_shard_paths
from splatfactory.datasets.utils import io, workers
from splatfactory.datasets.view_sampler import BaseViewSampler, get_view_sampler
from splatfactory.utils.image import ImagePreprocessor
from splatfactory.utils.timing import StepTimer

logger = get_logger(__name__)


@dataclass
class _ChildSource:
    """Per-child state: shards, sampler, preprocessor, and display names."""

    name: str
    dataset_name: str
    shard_paths: list[Path] = field(repr=False)
    view_sampler: BaseViewSampler = field(repr=False)
    preprocessor: ImagePreprocessor = field(repr=False)
    weight: float = 1.0
    num_scenes: int = 0
    aspect_ratios: list[float] = field(default_factory=lambda: [1.0])
    max_pose_jump_ratio: float = 0.0


class ComposedDataset(MultiViewDataset):
    """Shard dataset combining several children with weighted sampling."""

    base_default_conf = {
        **MultiViewDataset.base_default_conf,
        "dataset_name": "composed",
        "dataset_dir": ".",
        "train_shard_dir": ".",
        "test_shard_dir": ".",
        "childs": {},
        "samples_per_epoch": None,  # train: global scenes per epoch, split over ranks x workers
        "val_samples": None,  # val: global scenes per pass, a fixed subset
    }

    def _init(self, conf, split: str = "train") -> None:
        self.split = split
        self._epoch: int = 0
        self._image_num_range: list[int] = list(conf.image_num_range)
        self.step_timer = StepTimer()
        self.view_timer = StepTimer()

        self._children: list[_ChildSource] = [
            self._build_child(label, child_conf, split) for label, child_conf in conf.childs.items()
        ]
        self._tagged_shards: list[tuple[Path, int]] = self._build_weighted_pool()

    def _build_child(self, label: str, child_conf, split: str) -> _ChildSource:
        if isinstance(child_conf, dict):
            child_conf = OmegaConf.create(child_conf)

        dataset_dir = Path(child_conf.dataset_dir)
        default_dir = dataset_dir / ("train-scenes" if split == "train" else "test-scenes")
        dir_key = "train_shard_dir" if split == "train" else "test_shard_dir"
        raw_dir = child_conf.get(dir_key, str(default_dir))
        shards = resolve_shard_paths(raw_dir)
        if not shards:
            raise FileNotFoundError(f"No shards for '{label}' in {raw_dir}")

        num_scenes = self._count_scenes(raw_dir)
        vs_conf = child_conf.view_sampler
        view_sampler = get_view_sampler(vs_conf.name)(vs_conf, split=split)
        aug_conf = child_conf.get("augmentations", {"name": "identity"})
        augmentation = get_augmentation(aug_conf) if split == "train" else None
        preprocessor = ImagePreprocessor(
            child_conf.get("preprocessing", {}), augmentation=augmentation
        )

        logger.info(f"[{split}] {label}: {num_scenes} scenes in {len(shards)} shards ({raw_dir})")
        return _ChildSource(
            name=label,
            dataset_name=child_conf.get("dataset_name", label),
            shard_paths=shards,
            view_sampler=view_sampler,
            preprocessor=preprocessor,
            weight=child_conf.get("weight", 1.0),
            num_scenes=num_scenes,
            aspect_ratios=[
                float(a) for a in child_conf.get("aspect_ratios", self.conf.aspect_ratios)
            ],
            max_pose_jump_ratio=float(
                child_conf.get("max_pose_jump_ratio", self.conf.max_pose_jump_ratio)
            ),
        )

    @staticmethod
    def _count_scenes(shard_dirs) -> int:
        """Scenes in one or several shard directories, from index.json or by scanning tars."""
        if isinstance(shard_dirs, (str, Path)):
            shard_dirs = [shard_dirs]
        total = 0
        for d in shard_dirs:
            d = Path(str(d))
            index_path = d / "index.json"
            if index_path.exists():
                with open(index_path) as f:
                    total += len(json.load(f))
                continue
            count = sum(
                1
                for tar in sorted(d.glob("shard-*.tar"))
                for _ in io.iter_scenes_from_tar(str(tar))
            )
            logger.warning(f"No index.json in {d}, counted {count} scenes by scanning")
            total += count
        return total

    def _build_weighted_pool(self) -> list[tuple[Path, int]]:
        """Repeat shard lists so effective scene counts match weight ratios.

        repeat_i = round((weight_i / scenes_i) / min_j(weight_j / scenes_j)).
        Minimum repeat is 1 (oversample only, never undersample).
        """
        ratios = [c.weight / max(c.num_scenes, 1) for c in self._children]
        min_ratio = min(ratios)

        tagged: list[tuple[Path, int]] = []
        for i, child in enumerate(self._children):
            repeats = max(1, round(ratios[i] / min_ratio))
            tagged.extend((s, i) for _ in range(repeats) for s in child.shard_paths)
            effective = child.num_scenes * repeats
            logger.info(
                f"  {child.name}: {len(child.shard_paths)} shards x{repeats} "
                f"= {len(child.shard_paths) * repeats} effective shards "
                f"(~{effective} scenes, weight={child.weight:.2f})"
            )
        return tagged

    def __len__(self) -> int:
        return len(self._tagged_shards)

    @property
    def shard_paths(self) -> list[Path]:
        """Parent's DataLoader setup checks this for worker capping."""
        return [s for s, _ in self._tagged_shards]

    # ---- Iteration ---------------------------------------------------------------

    def __iter__(self) -> Iterator[dict]:
        """Yield pre-collated batches from interleaved child datasets."""
        rank, world_size = workers.get_rank(), workers.get_world_size()
        wid, num_workers = workers.get_dataloader_worker_info()

        tagged = self._tagged_shards[rank::world_size][wid::num_workers]
        total = self.conf.samples_per_epoch if self.split == "train" else self.conf.val_samples
        if self.split == "train":
            random.Random(self.seed + self._epoch + wid).shuffle(tagged)
        elif total:
            random.Random(self.seed + wid).shuffle(tagged)  # same val subset every pass
        cap = math.ceil(total / (world_size * num_workers)) if total else self.conf.max_samples

        logger.info(
            f"[rank={rank}/{world_size}, worker={wid}/{num_workers}] "
            f"epoch={self._epoch}, {len(tagged)} shards from "
            f"{len(self._children)} datasets, views={self._image_num_range}"
        )

        def tagged_scene_stream():
            for shard_path, child_idx in tagged:
                child = self._children[child_idx]
                for raw_scene in io.iter_scenes_from_tar(str(shard_path)):
                    yield raw_scene, child

        batch_rng = random.Random(self.seed + self._epoch + wid)
        is_train = self.split == "train"
        yielded = 0
        skipped = 0
        stream = tagged_scene_stream()
        pending = None  # scene of another child that closed the previous batch

        while True:
            vc, _, bs = self._sample_batch_params(batch_rng, is_train)

            # One child per batch: the aspect ratio comes from the child's preset list, and a
            # scene from another child closes the batch early and opens the next one.
            batch, batch_child, ar = [], None, None
            self.step_timer.reset()
            while True:
                if pending is not None:
                    raw_scene, child = pending
                    pending = None
                else:
                    raw_scene, child = next(stream, (None, None))
                    if raw_scene is None:
                        break
                self.step_timer.measure("tar_io")
                if batch_child is None:
                    batch_child = child
                    ar = float(batch_rng.choice(child.aspect_ratios))
                elif child is not batch_child:
                    pending = (raw_scene, child)
                    break
                if not self._scene_ok(raw_scene, vc, view_sampler=child.view_sampler):
                    self.step_timer.reset()
                    continue
                try:
                    sample = self._process_scene(
                        raw_scene,
                        vc,
                        ar,
                        view_sampler=child.view_sampler,
                        preprocessor=child.preprocessor,
                        dataset_name=child.dataset_name,
                        max_pose_jump_ratio=child.max_pose_jump_ratio,
                    )
                    sample["dataset"] = child.name
                    batch.append(sample)
                except Exception as e:
                    logger.debug(f"Skipping scene {raw_scene.get('key', '?')}: {e}")
                    skipped += 1
                self.step_timer.reset()
                if len(batch) >= bs:
                    break

            if not batch:
                if pending is None:
                    break  # stream exhausted
                continue  # every scene of this child was skipped, start on the pending one

            yield workers.collate(batch)
            yielded += len(batch)
            if cap is not None and yielded >= cap:
                break

        if skipped > 0:
            logger.info(
                f"[worker={wid}] epoch={self._epoch}: skipped {skipped} scenes "
                f"({skipped / max(yielded + skipped, 1) * 100:.1f}%)"
            )

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch


if __name__ == "__main__":
    import argparse

    from omegaconf import OmegaConf
    from tqdm import tqdm

    from splatfactory.utils import mappings

    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", type=str, default="composed_252")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-batches", type=int, default=10)
    args = parser.parse_args()

    conf_path = Path(__file__).parents[1] / "configs" / "data" / f"{args.conf}.yaml"
    conf = OmegaConf.load(conf_path)
    OmegaConf.resolve(conf)
    conf.num_workers = args.num_workers
    conf.prefetch_factor = None if args.num_workers == 0 else 2
    conf.train_batch_size = 2
    conf.image_num_range = [2, 2]

    dataset = ComposedDataset(conf, split=args.split)
    loader = dataset.get_loader(num_workers=args.num_workers)

    dataset_counts: dict[str, int] = {}
    for i, data in enumerate(tqdm(loader, total=args.num_batches)):
        for ds_name in data["dataset"]:
            dataset_counts[ds_name] = dataset_counts.get(ds_name, 0) + 1
        print(f"\nBatch {i}: datasets={data['dataset']}")
        mappings.print_summary(data)
        if i >= args.num_batches - 1:
            break

    print(f"\nDataset distribution over {sum(dataset_counts.values())} samples:")
    total = sum(dataset_counts.values())
    for name, count in sorted(dataset_counts.items()):
        print(f"  {name}: {count} ({count / total * 100:.1f}%)")
