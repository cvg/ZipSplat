"""MipNeRF360 Evaluation Pipeline.

Author: Alexander Veicht
"""

from pathlib import Path

from splatfactory.eval import io
from splatfactory.eval.multiview_benchmark import MultiViewBenchmark


class MipNeRF360Benchmark(MultiViewBenchmark):
    # context view count -> shipped eval index (select with `--views`)
    INDICES = {
        32: "splatfactory/eval/indices/mipnerf360/ctx_32v.json",
        64: "splatfactory/eval/indices/mipnerf360/ctx_64v.json",
        128: "splatfactory/eval/indices/mipnerf360/ctx_128v.json",
    }
    default_conf = {
        "data": {
            "name": "dataset_mipnerf360",
            "view_sampler": {"name": "eval_sampler", "indices_file": INDICES[32]},
            "num_workers": 2,
            "prefetch_factor": None,
        },
    }


if __name__ == "__main__":
    io.run_cli(MipNeRF360Benchmark, name=Path(__file__).stem)
