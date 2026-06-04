"""Mip-NeRF 360 dataset configuration.

Unbounded indoor/outdoor scenes. Long sequences (~150-300 views per scene),
no depth. One tar shard per scene (`{scene}.tar`).

Author: Alexander Veicht
"""

from splatfactory.datasets.multi_view_dataset import MultiViewDataset


class MipNeRF360(MultiViewDataset):
    default_conf = {
        "name": "dataset_mipnerf360",
        "dataset_name": "mipnerf360",
        "dataset_dir": "data/mipnerf360/822x1236",
        "train_shard_dir": "${.dataset_dir}/test-scenes",
        "test_shard_dir": "${.dataset_dir}/test-scenes",
        "shard_glob": "*.tar",  # one tar per scene, not `shard-*.tar`
        # Views
        "image_num_range": [6, 6],
        "aspect_ratio_range": [1.0, 1.0],
        "num_target_views": 8,
        "random_reference_view": False,
        # View sampler - eval uses external indices via eval_sampler; default bounded is a sane fallback.
        "view_sampler": {"name": "bounded", "warm_up_steps": 0},
        "max_pose_jump_ratio": 0,  # static scenes - not a video
        # Normalization
        "near": 0.1,
        "far": 100.0,
    }


if __name__ == "__main__":
    from splatfactory.datasets.utils.debug import debug_dataset

    debug_dataset(MipNeRF360)
