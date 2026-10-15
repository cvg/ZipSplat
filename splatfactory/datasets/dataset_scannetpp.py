"""ScanNet++ dataset configuration.

Indoor scenes, converted to tar shards under `data/scannetpp_v2/<HxW>/<device>/`.
Train pools both device captures (dslr + iphone); test is dslr only -- the
canonical NVS split the shipped eval indices are built on (iphone has no NVS
test split, so its test-scenes are left unused).

Author: Alexander Veicht
"""

from splatfactory.datasets.multi_view_dataset import MultiViewDataset


class ScanNetPP(MultiViewDataset):
    default_conf = {
        "name": "dataset_scannetpp",
        "dataset_name": "scannetpp",
        "dataset_dir": "data/scannetpp_v2/720x1080",
        "train_shard_dir": [
            "${..dataset_dir}/dslr/train-scenes",
            "${..dataset_dir}/iphone/train-scenes",
        ],
        "test_shard_dir": "${.dataset_dir}/dslr/test-scenes",
        # Views
        "image_num_range": [6, 6],
        "aspect_ratios": [1.0],
        "num_target_views": 8,
        "random_reference_view": False,
        "view_sampler": {"name": "bounded", "warm_up_steps": 0},
        "max_pose_jump_ratio": 0,  # static scenes - not a video
        # Normalization
        "near": 0.1,
        "far": 100.0,
    }


if __name__ == "__main__":
    from splatfactory.datasets.utils.debug import debug_dataset

    debug_dataset(ScanNetPP)
