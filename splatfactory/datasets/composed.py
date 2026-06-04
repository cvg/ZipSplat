"""Composed dataset - combines multiple datasets with weighted sampling.

Inherits from MultiViewDataset. Pools shards from all children, oversamples smaller
datasets so effective scene counts match configured weight ratios. Each shard is
tagged with its source child, and per-child view_sampler/preprocessor/dataset_name
are passed explicitly to the parent's scene processing.

Adding a new dataset requires only a config change.

Author: Alexander Veicht
"""

import json
import random
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from omegaconf import OmegaConf

from splatfactory import get_logger
from splatfactory.datasets.augmentations import get_augmentation
from splatfactory.datasets.multi_view_dataset import MultiViewDataset
from splatfactory.datasets.utils import io, workers
from splatfactory.datasets.view_sampler import BaseViewSampler, get_view_sampler
from splatfactory.utils.image import ImagePreprocessor
from splatfactory.utils.tools import StepTimer

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


class ComposedDataset(MultiViewDataset):
    """Shard dataset combining multiple shard sources with weighted sampling.

    Each tagged shard carries its child index; the child's view_sampler and
    preprocessor are passed to the parent's _process_scene per scene.
    """

    base_default_conf = {
        **MultiViewDataset.base_default_conf,
        "dataset_name": "composed",
        "dataset_dir": ".",
        "train_shard_dir": ".",
        "test_shard_dir": ".",
        "childs": {},
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
        shard_dir = Path(child_conf.get(dir_key, str(default_dir)))
        shards = sorted(shard_dir.glob("shard-*.tar"))
        if not shards:
            raise FileNotFoundError(f"No shards for '{label}' in {shard_dir}")

        num_scenes = self._count_scenes(shard_dir)
        vs_conf = child_conf.view_sampler
        view_sampler = get_view_sampler(vs_conf.name)(vs_conf, split=split)
        aug_conf = child_conf.get("augmentations", {"name": "identity"})
        augmentation = get_augmentation(aug_conf) if split == "train" else None
        preprocessor = ImagePreprocessor(
            child_conf.get("preprocessing", {}), augmentation=augmentation
        )

        logger.info(f"[{split}] {label}: {num_scenes} scenes in {len(shards)} shards ({shard_dir})")
        return _ChildSource(
            name=label,
            dataset_name=child_conf.get("dataset_name", label),
            shard_paths=shards,
            view_sampler=view_sampler,
            preprocessor=preprocessor,
            weight=child_conf.get("weight", 1.0),
            num_scenes=num_scenes,
        )

    @staticmethod
    def _count_scenes(shard_dir: Path) -> int:
        """Count scenes from index.json (fast) or by scanning tars (slow fallback)."""
        index_path = shard_dir / "index.json"
        if index_path.exists():
            with open(index_path) as f:
                return len(json.load(f))
        count = sum(
            1
            for tar in sorted(shard_dir.glob("shard-*.tar"))
            for _ in io.iter_scenes_from_tar(str(tar))
        )
        logger.warning(f"No index.json in {shard_dir}, counted {count} scenes by scanning")
        return count

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
        if self.split == "train":
            random.Random(self.seed + self._epoch + wid).shuffle(tagged)

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

        while True:
            vc, ar, bs = self._sample_batch_params(batch_rng, is_train)

            batch = []
            self.step_timer.reset()
            for raw_scene, child in stream:
                self.step_timer.measure("tar_io")
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
