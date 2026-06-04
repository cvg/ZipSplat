"""RE10K dataset configuration.

Author: Alexander Veicht
"""

from splatfactory.datasets.multi_view_dataset import MultiViewDataset


class RealEstate10K(MultiViewDataset):
    default_conf = {
        "name": "dataset_re10k",
        "dataset_name": "re10k",
        "dataset_dir": "data/re10k/360x640",
        "train_shard_dir": "${.dataset_dir}/train-scenes",
        "test_shard_dir": "${.dataset_dir}/test-scenes",
        # Views
        "image_num_range": [2, 24],
        "aspect_ratio_range": [1.0, 1.0],
        "num_target_views": 4,
        "random_reference_view": True,
        # View sampler
        "view_sampler": {
            "name": "bounded",
            "warm_up_steps": 0,
            "min_distance_between_context_views": 35,
            "gap_multiplier": 45,
            "max_distance_between_consecutive_views": 90,
        },
        # Normalization
        "near": 0.1,
        "far": 100.0,
    }


if __name__ == "__main__":
    from splatfactory.datasets.utils.debug import debug_dataset

    debug_dataset(RealEstate10K)
