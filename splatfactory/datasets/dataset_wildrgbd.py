"""WildRGB-D dataset configuration.

Object-centric iPhone RGB-D videos, converted to tar shards under `data/wildrgbd/640x480/`
(portrait). Frames are subsampled evenly per video, and a per-category fraction of the
videos forms the test split. Depth is the sensor depth.

Author: Alexander Veicht
"""

from splatfactory.datasets.multi_view_dataset import MultiViewDataset


class WildRGBD(MultiViewDataset):
    default_conf = {
        "name": "dataset_wildrgbd",
        "dataset_name": "wildrgbd",
        "dataset_dir": "data/wildrgbd/640x480",
        "train_shard_dir": "${.dataset_dir}/train-scenes",
        "test_shard_dir": "${.dataset_dir}/test-scenes",
        # Views
        "image_num_range": [6, 6],
        "aspect_ratios": [1.0],
        "num_target_views": 8,
        "random_reference_view": False,
        "view_sampler": {"name": "bounded", "warm_up_steps": 0},
        "max_pose_jump_ratio": 0,  # orbits around one object, no shot cuts
        # Normalization
        "near": 0.1,
        "far": 100.0,
    }


if __name__ == "__main__":
    from splatfactory.datasets.utils.debug import debug_dataset

    debug_dataset(WildRGBD)
