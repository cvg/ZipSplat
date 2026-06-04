"""RE10K Evaluation Pipeline.

Author: Alexander Veicht
"""

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from splatfactory.eval import io
from splatfactory.eval.multiview_benchmark import MultiViewBenchmark
from splatfactory.utils import metrics


def get_overlap_tag(overlap):
    if 0.05 <= overlap <= 0.3:
        return "small"
    elif overlap <= 0.55:
        return "medium"
    elif overlap <= 0.8:
        return "large"
    return "ignore"


class RE10KBenchmark(MultiViewBenchmark):
    # context view count -> shipped eval index (select with `--views`)
    INDICES = {6: "splatfactory/eval/indices/re10k/ctx_6v_tgt_8v.json"}
    default_conf = {
        "data": {
            "name": "dataset_re10k",
            "view_sampler": {"indices_file": INDICES[6]},
        },
    }

    def _extra_sample_fields(self, data):
        """Collect overlap for per-difficulty breakdowns."""
        if "overlap" in data:
            return {"overlap": data["overlap"]}
        return {}

    def run_eval(self, loader, pred_file):
        summaries, figures, results = super().run_eval(loader, pred_file)

        # Add per-difficulty breakdowns if overlap data is available
        if "overlap" not in results:
            return summaries, figures, results

        category_indices = defaultdict(list)
        for idx, overlap in enumerate(results["overlap"]):
            category_indices[get_overlap_tag(float(overlap))].append(idx)

        target_metrics = ["target-psnr", "target-ssim", "target-lpips"]
        for metric_name in target_metrics:
            if metric_name not in results:
                continue
            arr = np.array(results[metric_name])
            for category, indices in category_indices.items():
                if indices:
                    val = round(metrics.AverageMetric(arr[indices]).compute(), 3)
                else:
                    val = -1.0
                summaries[f"mean-{metric_name}-{category}"] = val

        # Scatter plot: overlap vs PSNR
        fig, ax = plt.subplots(1, 1)
        ax.scatter(results["overlap"], results["target-psnr"])
        ax.set_xlabel("Overlap")
        ax.set_ylabel("Target PSNR")
        figures["scatter"] = fig

        return summaries, figures, results


if __name__ == "__main__":
    io.run_cli(RE10KBenchmark, name=Path(__file__).stem)
