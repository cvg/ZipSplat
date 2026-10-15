"""UnrealStereo4K dataset configuration.

Static Unreal Engine environments rendered from scattered stereo viewpoints, converted to
tar shards under `data/unrealstereo4k/540x960/`. Each sequence is split into chunks of
consecutive stereo pairs, one scene record per chunk. Depth is the rendered ground truth.

Author: Alexander Veicht
"""

from splatfactory.datasets.multi_view_dataset import MultiViewDataset


class UnrealStereo4K(MultiViewDataset):
    default_conf = {
        "name": "dataset_unrealstereo4k",
        "dataset_name": "unrealstereo4k",
        "dataset_dir": "data/unrealstereo4k/540x960",
        "train_shard_dir": "${.dataset_dir}/train-scenes",
        "test_shard_dir": "${.dataset_dir}/test-scenes",
        # Views
        "image_num_range": [6, 6],
        "aspect_ratios": [1.0],
        "num_target_views": 8,
        "random_reference_view": False,
        "view_sampler": {"name": "bounded", "warm_up_steps": 0},
        "max_pose_jump_ratio": 0,  # scattered viewpoints, not a video
        # Normalization
        "near": 0.1,
        "far": 100.0,
    }


if __name__ == "__main__":
    from splatfactory.datasets.utils.debug import debug_dataset

    debug_dataset(UnrealStereo4K)
