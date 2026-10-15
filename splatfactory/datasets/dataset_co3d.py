"""CO3Dv2 dataset configuration.

Object-centric phone orbits, converted to tar shards under `data/co3d/720s/` (the short
side capped at 720, so the tree is named by it rather than by a fixed HxW). Sequences are
selected as in DUSt3R, and a per-category fraction of them forms the test split. Depth is
the dataset's own.

Author: Alexander Veicht
"""

from splatfactory.datasets.multi_view_dataset import MultiViewDataset


class CO3D(MultiViewDataset):
    default_conf = {
        "name": "dataset_co3d",
        "dataset_name": "co3d",
        "dataset_dir": "data/co3d/720s",
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

    debug_dataset(CO3D)
