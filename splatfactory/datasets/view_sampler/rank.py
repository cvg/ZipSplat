"""Rank-based view sampler using precomputed pose distance rankings.

Samples views based on camera pose proximity rather than frame order.
Requires precomputed rankings JSON from splatfactory/datasets/scripts/precompute_view_ranking.py.

Author: Alexander Veicht
"""

import json
import random
from pathlib import Path

import torch

from splatfactory import get_logger
from splatfactory.datasets.view_sampler import BaseViewSampler

logger = get_logger(__name__)


class RankViewSampler(BaseViewSampler):
    """View sampler that uses precomputed pose-distance rankings."""

    default_conf = {
        "ranking_file": None,
        "min_distance_to_context_views": 0,
        "warm_up_steps": 150_000,
        "min_distance_between_context_views": 8,
        "max_distance_between_context_views": 22,
        "initial_min_distance_between_context_views": 5,
        "initial_max_distance_between_context_views": 7,
    }

    def _init(self, conf):
        if conf.ranking_file is None:
            raise ValueError("RankViewSampler requires ranking_file to be set")

        ranking_path = Path(conf.ranking_file)
        if not ranking_path.exists():
            raise FileNotFoundError(f"Ranking file not found: {ranking_path}")

        logger.info(f"Loading rankings from {ranking_path}")
        with open(ranking_path, "r") as f:
            rankings_dict = json.load(f)

        # Check format: new format has {"rankings": ..., "overlaps": ...}
        sample_value = next(iter(rankings_dict.values()))
        has_overlaps = isinstance(sample_value, dict) and "rankings" in sample_value

        if has_overlaps:
            self.rankings = {
                scene_id: torch.tensor(data["rankings"], dtype=torch.long)
                for scene_id, data in rankings_dict.items()
            }
            self.overlaps = {
                scene_id: torch.tensor(data["overlaps"], dtype=torch.float32)
                for scene_id, data in rankings_dict.items()
            }
            logger.info(f"Loaded rankings with overlap scores for {len(self.rankings)} scenes")
        else:
            self.rankings = {
                scene_id: torch.tensor(ranking, dtype=torch.long)
                for scene_id, ranking in rankings_dict.items()
            }
            self.overlaps = None
            logger.info(f"Loaded rankings for {len(self.rankings)} scenes")

        self.valid_scenes = set(self.rankings.keys())

    def min_required_images(self) -> int:
        # Need at least: 2 context + 1 target
        return max(3, self.conf.min_distance_between_context_views + 1)

    def sample(
        self,
        scene_id: str | None = None,
        num_context_views: int | None = None,
        num_target_views: int | None = None,
        total_num_views: int | None = None,
    ):
        """Sample views based on pose-distance ranking."""
        if scene_id not in self.rankings:
            raise KeyError(f"Scene {scene_id} not found in rankings")

        ranking = self.rankings[scene_id]  # (N, k)
        n_views, k = ranking.shape

        # Handle single context view case
        if num_context_views == 1:
            ref = random.randint(0, n_views - 1)
            return {
                "context_indices": [ref],
                "target_indices": [ref],
                "overlap_scores": [1.0],
            }

        # Get gap parameters (with warmup scheduling)
        min_gap = self.linear_schedule(
            self.conf.initial_min_distance_between_context_views,
            self.conf.min_distance_between_context_views,
        )
        max_gap = self.linear_schedule(
            self.conf.initial_max_distance_between_context_views,
            self.conf.max_distance_between_context_views,
        )

        # Clamp to available rankings
        max_gap = min(max_gap, k - 1)
        min_gap = min(min_gap, max_gap)

        # 1. Pick random reference view
        ref = random.randint(0, n_views - 1)
        ref_ranking = ranking[ref]  # top-k closest to ref

        # 2. Sample gap (distance in ranking)
        gap = random.randint(min_gap, max_gap)

        # 3. Right context is the gap-th closest view
        right_context = ref_ranking[gap].item()

        # 4. Middle views are those between ref and right_context in ranking
        middle = ref_ranking[1:gap].tolist()

        # 5. Ensure enough views for targets + extra context
        num_extra = num_context_views - 2
        num_needed = num_extra + num_target_views

        if len(middle) < num_needed:
            # Expand gap to get more middle views
            new_gap = min(num_needed + 1, k - 1)
            middle = ref_ranking[1:new_gap].tolist()
            right_context = ref_ranking[new_gap].item()

        # 6. Sample extra context and targets from middle (no overlap)
        if len(middle) >= num_needed:
            sampled = random.sample(middle, num_needed)
        else:
            # Not enough middle views, sample what we can
            sampled = middle.copy()
            # Fill remaining from views beyond right_context
            remaining_k = k - gap - 1
            if remaining_k > 0:
                extra_pool = ref_ranking[gap + 1 :].tolist()
                needed = num_needed - len(sampled)
                sampled.extend(random.sample(extra_pool, min(needed, len(extra_pool))))

        extra = sampled[:num_extra]
        targets = sampled[num_extra : num_extra + num_target_views]

        # Ensure at least 1 target
        if len(targets) < 1:
            # Fallback: use right_context as target
            targets = [right_context]
            # Pick a new right_context from further in ranking if possible
            if gap + 1 < k:
                right_context = ref_ranking[gap + 1].item()

        context = [ref] + sorted(extra) + [right_context]

        # Get overlap scores if available
        if self.overlaps is not None:
            ref_overlaps = self.overlaps[scene_id][ref]  # (k,)
            # Map context indices to their position in ranking to get overlap
            overlap_scores = []
            for ctx_idx in context:
                if ctx_idx == ref:
                    overlap_scores.append(1.0)
                else:
                    # Find position of ctx_idx in ref's ranking
                    pos = (ref_ranking == ctx_idx).nonzero()
                    if len(pos) > 0:
                        overlap_scores.append(ref_overlaps[pos[0].item()].item())
                    else:
                        overlap_scores.append(0.5)
        else:
            overlap_scores = [0.5] * len(context)

        return {
            "context_indices": context,
            "target_indices": sorted(targets),
            "overlap_scores": overlap_scores,
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ranking_file", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=10)
    args = parser.parse_args()

    sampler = RankViewSampler(
        {"ranking_file": args.ranking_file},
        split="train",
    )

    # Get first scene
    scene_id = list(sampler.rankings.keys())[0]
    n_views = sampler.rankings[scene_id].shape[0]

    print(f"Scene: {scene_id}, {n_views} views")
    for i in range(args.num_samples):
        result = sampler.sample(
            scene_id=scene_id,
            num_context_views=2,
            num_target_views=4,
            total_num_views=n_views,
        )
        print(f"  [{i}] context={result['context_indices']}, target={result['target_indices']}")
