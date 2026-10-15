"""DL3DV Evaluation Pipeline.

Author: Alexander Veicht
"""

from pathlib import Path

from splatfactory.eval import io
from splatfactory.eval.multiview_benchmark import MultiViewBenchmark


class DL3DVBenchmark(MultiViewBenchmark):
    # context view count -> eval index (select with `eval.num_views`). The Long-LRM protocol:
    # every 8th frame is a target, the context views are k-means picks over the rest.
    INDICES = {
        nv: f"splatfactory/eval/indices/dl3dv/longlrm_ctx_{nv}v.json" for nv in (16, 32, 64, 128)
    }
    default_conf = {
        "data": {
            "name": "dataset_dl3dv",
            "view_sampler": {"indices_file": INDICES[16]},
        },
    }


if __name__ == "__main__":
    io.run_cli(DL3DVBenchmark, name=Path(__file__).stem)
