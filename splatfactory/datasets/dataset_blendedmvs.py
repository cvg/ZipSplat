"""BlendedMVS dataset configuration.

Unordered MVS captures (objects, buildings, aerial), converted to tar shards
under `data/blendedmvs/576x768/`. Train holds the 106 scenes of the official
list, test the 7 scenes of `validation_list.txt`. Depth comes from the dataset's
mesh-rendered depth maps.

Author: Alexander Veicht
"""

from splatfactory.datasets.multi_view_dataset import MultiViewDataset


class BlendedMVS(MultiViewDataset):
    default_conf = {
        "name": "dataset_blendedmvs",
        "dataset_name": "blendedmvs",
        "dataset_dir": "data/blendedmvs/576x768",
        "train_shard_dir": "${.dataset_dir}/train-scenes",
        "test_shard_dir": "${.dataset_dir}/test-scenes",
        # Views
        "image_num_range": [6, 6],
        "aspect_ratios": [1.0],
        "num_target_views": 8,
        "random_reference_view": False,
        "view_sampler": {"name": "bounded", "warm_up_steps": 0},
        "max_pose_jump_ratio": 0,  # unordered MVS captures - not a video
        # Normalization
        "near": 0.1,
        "far": 100.0,
    }


if __name__ == "__main__":
    from splatfactory.datasets.utils.debug import debug_dataset

    debug_dataset(BlendedMVS)
