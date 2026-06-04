"""Random view sampler: draws context/target views uniformly at random,
without distance or ordering constraints.

Author: Alexander Veicht
"""

import numpy as np

from splatfactory.datasets.view_sampler import BaseViewSampler


class RandomViewSampler(BaseViewSampler):
    """Samples random views without any distance constraints."""

    default_conf = {}

    def _init(self, conf):
        pass

    def min_required_images(self) -> int:
        return 1

    def sample(
        self,
        scene_id: str | None = None,
        num_context_views: int | None = None,
        num_target_views: int | None = None,
        total_num_views: int | None = None,
    ):
        # Sample random unique indices for context
        all_indices = np.random.permutation(total_num_views)
        context_indices = sorted(all_indices[:num_context_views].tolist())
        target_indices = sorted(
            all_indices[num_context_views : num_context_views + num_target_views].tolist()
        )

        return {
            "context_indices": context_indices,
            "target_indices": target_indices,
            "overlap_scores": [0.5] * num_context_views,
        }
