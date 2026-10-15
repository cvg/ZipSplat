"""Covisibility-window view sampler for unordered or room-scan captures.

Needs the per-scene covisibility members written into the shards by
`splatfactory/datasets/scripts/precompute_rankings.py` (`{key}.rankings.npy`, `{key}.overlaps.npy`):
for every view its k most covisible views, with the measured overlap fraction. They arrive
with the raw scene, so nothing is held in memory across scenes.

Sampling, per scene: a random reference view; the far context view from the views whose
overlap with the reference lies in [min_overlap, max_overlap] (gap mode "min" takes the
least covisible of them = hard, "max" the most = easy); the remaining context and all
target views from the views with overlap >= min_overlap. Everything sampled therefore
shares at least min_overlap of its content with the reference, independent of scene size
and frame order. Scenes too sparse to fill the request fall back to the next most
covisible views.

Author: Alexander Veicht
"""

import random

import numpy as np

from splatfactory import get_logger
from splatfactory.datasets.view_sampler import BaseViewSampler

logger = get_logger(__name__)


class RankViewSampler(BaseViewSampler):
    """Samples context/target views by measured covisibility with a reference view."""

    default_conf = {
        "min_overlap": 0.2,  # every sampled view shares >= this with the reference
        "max_overlap": 0.8,  # far context view drawn from [min_overlap, max_overlap]
    }
    # Scene members this sampler reads; the dataset skips scenes that lack them.
    required_scene_keys = ("rankings", "overlaps")

    def _init(self, conf):
        pass

    def min_required_images(self) -> int:
        return 3

    def _fill(self, chosen: list[int], pool: list[int], fallback: list[int], n: int) -> list[int]:
        """n random views from pool (minus chosen); fallback order tops up sparse scenes."""
        avail = [v for v in pool if v not in chosen]
        out = random.sample(avail, min(n, len(avail)))
        for v in fallback:
            if len(out) >= n:
                break
            if v not in chosen and v not in out:
                out.append(v)
        return out

    def sample(
        self,
        scene_id: str | None = None,
        num_context_views: int | None = None,
        num_target_views: int | None = None,
        total_num_views: int | None = None,
        scene: dict | None = None,
    ):
        if scene is None or "rankings" not in scene:
            raise KeyError(f"Scene {scene_id} carries no covisibility rankings")
        ref = random.randint(0, len(scene["rankings"]) - 1)
        ranked = scene["rankings"][ref, 1:].tolist()  # most covisible first, self excluded
        # float32 on disk; rounding restores the 3-decimal scores the thresholds were tuned on
        scores = np.round(scene["overlaps"][ref, 1:].astype(np.float64), 3).tolist()
        score_of = dict(zip(ranked, scores))
        lo, hi = self.conf.min_overlap, self.conf.max_overlap
        pool = [v for v, s in zip(ranked, scores) if s >= lo]
        chosen = [ref]

        if num_context_views >= 2:
            window = [v for v in pool if score_of[v] <= hi] or pool[-1:] or ranked[:1]
            if self.gap_mode == "max":
                far = window[0]
            elif self.gap_mode == "min":
                far = window[-1]
            else:
                far = random.choice(window)
            chosen.append(far)
            chosen += self._fill(chosen, pool, ranked, num_context_views - 2)

        targets = self._fill(chosen, pool, ranked, num_target_views)
        return {
            "context_indices": chosen,
            "target_indices": sorted(targets),
            "overlap_scores": [1.0] + [score_of[v] for v in chosen[1:]],
        }
