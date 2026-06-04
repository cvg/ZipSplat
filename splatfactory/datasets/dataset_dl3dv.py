"""DL3DV dataset configuration.

Author: Alexander Veicht
"""

from splatfactory.datasets.multi_view_dataset import MultiViewDataset


class DL3DV(MultiViewDataset):
    default_conf = {
        "name": "dataset_dl3dv",
        "dataset_name": "dl3dv",
        "dataset_dir": "data/dl3dv/540x960",
        "train_shard_dir": "${.dataset_dir}/train-scenes",
        "test_shard_dir": "${.dataset_dir}/test-scenes",
        # Views
        "image_num_range": [2, 2],
        "aspect_ratio_range": [1.0, 1.0],
        "num_target_views": 4,
        "random_reference_view": True,
        # View sampler (DL3DV has shorter sequences than RE10K)
        "view_sampler": {
            "name": "bounded",
            "warm_up_steps": 0,
            "min_distance_between_context_views": 8,
            "initial_min_distance_between_context_views": 5,
            "gap_multiplier": 11,
            "max_distance_between_consecutive_views": 22,
        },
        # Normalization
        "near": 0.1,
        "far": 100.0,
    }


if __name__ == "__main__":
    from splatfactory.datasets.utils.debug import debug_dataset

    debug_dataset(DL3DV)
