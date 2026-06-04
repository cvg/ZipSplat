"""DL3DV Evaluation Pipeline.

Author: Alexander Veicht
"""

from pathlib import Path

from splatfactory.eval import io
from splatfactory.eval.multiview_benchmark import MultiViewBenchmark


class DL3DVBenchmark(MultiViewBenchmark):
    # context view count -> shipped eval index (select with `--views`)
    INDICES = {
        6: "splatfactory/eval/indices/dl3dv/ctx_6v_tgt_8v.json",
        12: "splatfactory/eval/indices/dl3dv/ctx_12v_tgt_8v.json",
        24: "splatfactory/eval/indices/dl3dv/ctx_24v_tgt_8v.json",
    }
    default_conf = {
        "data": {
            "name": "dataset_dl3dv",
            "view_sampler": {"indices_file": INDICES[6]},
        },
    }


if __name__ == "__main__":
    io.run_cli(DL3DVBenchmark, name=Path(__file__).stem)
