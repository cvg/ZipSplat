"""G-Objaverse dataset configuration.

Rendered Objaverse objects (40 views per object on orbits looking at the origin),
converted to tar shards under `data/gobjaverse/512x512/`. Depth is the rendered
ground truth. A stable per-object hash assigns a fraction of the objects to the test split.

Author: Alexander Veicht
"""

from splatfactory.datasets.multi_view_dataset import MultiViewDataset


class GObjaverse(MultiViewDataset):
    default_conf = {
        "name": "dataset_gobjaverse",
        "dataset_name": "gobjaverse",
        "dataset_dir": "data/gobjaverse/512x512",
        "train_shard_dir": "${.dataset_dir}/train-scenes",
        "test_shard_dir": "${.dataset_dir}/test-scenes",
        # Views
        "image_num_range": [6, 6],
        "aspect_ratios": [1.0],
        "num_target_views": 8,
        "random_reference_view": False,
        "view_sampler": {"name": "bounded", "warm_up_steps": 0},
        "max_pose_jump_ratio": 0,  # orbit renders, no shot cuts
        # Normalization
        "near": 0.1,
        "far": 100.0,
    }


if __name__ == "__main__":
    from splatfactory.datasets.utils.debug import debug_dataset

    debug_dataset(GObjaverse)
