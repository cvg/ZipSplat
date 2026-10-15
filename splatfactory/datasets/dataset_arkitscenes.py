"""ARKitScenes dataset configuration.

Indoor iPhone video scans with LiDAR depth, converted to tar shards under
`data/arkitscenes/<HxW>/`. Videos are rotated upright at convert time, so landscape
captures live in `480x640` and portrait ones in `640x480`; train pools both trees.
Splits follow the official Training / Validation lists.

Author: Alexander Veicht
"""

from splatfactory.datasets.multi_view_dataset import MultiViewDataset


class ARKitScenes(MultiViewDataset):
    default_conf = {
        "name": "dataset_arkitscenes",
        "dataset_name": "arkitscenes",
        "dataset_dir": "data/arkitscenes",
        "train_shard_dir": [
            "${..dataset_dir}/480x640/train-scenes",
            "${..dataset_dir}/640x480/train-scenes",
        ],
        "test_shard_dir": [
            "${..dataset_dir}/480x640/test-scenes",
            "${..dataset_dir}/640x480/test-scenes",
        ],
        # Views
        "image_num_range": [6, 6],
        "aspect_ratios": [1.0],
        "num_target_views": 8,
        "random_reference_view": False,
        "view_sampler": {"name": "bounded", "warm_up_steps": 0},
        "max_pose_jump_ratio": 0,  # irregular hand-held motion, not shot cuts
        # Normalization
        "near": 0.1,
        "far": 100.0,
    }


if __name__ == "__main__":
    from splatfactory.datasets.utils.debug import debug_dataset

    debug_dataset(ARKitScenes)
