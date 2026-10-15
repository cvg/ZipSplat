"""Hypersim dataset configuration.

Photorealistic synthetic indoor scenes, converted to tar shards under
`data/ml-hypersim/768x1024/` with one scene record per camera trajectory. Depth is the
rendered ground truth. Splits follow the official scene split, with val merged into train.

Author: Alexander Veicht
"""

from splatfactory.datasets.multi_view_dataset import MultiViewDataset


class Hypersim(MultiViewDataset):
    default_conf = {
        "name": "dataset_hypersim",
        "dataset_name": "hypersim",
        "dataset_dir": "data/ml-hypersim/768x1024",
        "train_shard_dir": "${.dataset_dir}/train-scenes",
        "test_shard_dir": "${.dataset_dir}/test-scenes",
        # Views
        "image_num_range": [6, 6],
        "aspect_ratios": [1.0],
        "num_target_views": 8,
        "random_reference_view": False,
        "view_sampler": {"name": "bounded", "warm_up_steps": 0},
        "max_pose_jump_ratio": 0,  # rendered trajectories, not a video
        # Normalization
        "near": 0.1,
        "far": 100.0,
    }


if __name__ == "__main__":
    from splatfactory.datasets.utils.debug import debug_dataset

    debug_dataset(Hypersim)
