"""ScanNet++ Evaluation Pipeline, on the iPhone captures of the 46 NVS validation rooms.

Author: Alexander Veicht
"""

from pathlib import Path

from splatfactory.eval import io
from splatfactory.eval.multiview_benchmark import MultiViewBenchmark


class ScanNetPPBenchmark(MultiViewBenchmark):
    # context view count -> eval index (select with `eval.num_views`). FPS context views, 8
    # targets per scene.
    INDICES = {
        nv: f"splatfactory/eval/indices/scannetpp_iphone/ctx_{nv}v.json" for nv in (32, 64, 128)
    }
    default_conf = {
        "data": {
            "name": "dataset_scannetpp",
            "test_shard_dir": "${.dataset_dir}/iphone/test-scenes",
            "view_sampler": {
                "name": "eval_sampler",
                "indices_file": INDICES[32],
            },
        },
    }


if __name__ == "__main__":
    io.run_cli(ScanNetPPBenchmark, name=Path(__file__).stem)
