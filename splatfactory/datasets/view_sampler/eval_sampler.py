"""Evaluation view sampler: replays fixed context/target view indices from a
precomputed indices file so benchmarks are deterministic and reproducible.

Author: Alexander Veicht
"""

import json

import numpy as np
import torch

from splatfactory.datasets.view_sampler import BaseViewSampler


class EvalViewSampler(BaseViewSampler):
    default_conf = {
        "indices_file": "???",
        "is_overfitting": False,
    }

    def _init(self, conf):
        with open(conf.indices_file, "r") as f:
            self.index = json.load(f)

        self.valid_scenes = [k for k in self.index.keys() if self.index.get(k) is not None]

    def sample(
        self,
        scene_id: str | None = None,
        num_context_views: int | None = None,
        num_target_views: int | None = None,
        total_num_views: int | None = None,
    ):
        """Returns a dict with keys: context_indices, target_indices, overlap_scores"""

        entry = self.index.get(scene_id, None)
        if entry is None:
            print(f"ERROR: {scene_id} is in valid scenes: {scene_id in self.valid_scenes}")
            raise ValueError(f"No indices available for scene {scene_id}.")

        return {
            "context_indices": entry["context"],
            "target_indices": entry["target"],
            "overlap_scores": entry.get("overlap", -1.0),
        }

    def min_required_images(self):
        return 0


if __name__ == "__main__":
    # Example usage and simple test
    from splatfactory.utils.tools import set_seed

    set_seed(0)

    sampler = EvalViewSampler(
        {
            "indices_file": "splatfactory/eval/indices/re10k/ctx_6v_tgt_8v.json",
            "is_overfitting": False,
        },
        split="train",
    )

    names = [n for n in sampler.index.keys() if sampler.index.get(n) is not None]
    for i in range(10):
        indices = sampler.sample(names[i])
        print(indices)
