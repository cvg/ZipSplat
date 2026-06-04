"""Multi-View Evaluation Pipeline - shared base for all multi-view benchmarks.

Author: Alexander Veicht
"""

from collections import defaultdict

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from splatfactory import datasets, get_logger
from splatfactory.eval import eval_pipeline, io
from splatfactory.models.cache_loader import CacheLoader
from splatfactory.models.metrics import calculate_lpips, calculate_psnr, calculate_ssim
from splatfactory.utils import mappings, metrics, tools
from splatfactory.utils.export import export_predictions

logger = get_logger(__name__)


class MultiViewBenchmark(eval_pipeline.EvalPipeline):
    default_conf = {
        "data": {
            "train_batch_size": 1,
            "val_batch_size": 1,
            "test_batch_size": 1,
            "num_workers": 8,
            "max_pose_jump_ratio": 0,  # off at eval - videos-only filter
            "view_sampler": {
                "name": "eval_sampler",
            },
            "preprocessing": {
                "resize": 252,
            },
        },
        "model": {},
        "eval": {
            "align_poses": True,
            "align_context": False,
            # Final render + metric resolution. None = use target_camera as-is.
            # When set, pose alignment still runs at T (better signal), but the
            # final aligned render scales target_camera to (render_resize,
            # render_resize) and the GT target is interpolated to the same size.
            "render_resize": None,
            "pose_estimator": {"num_steps": 200, "patience_limit": 5, "batch_size": 10},
        },
    }

    def __init__(self, conf):
        self.default_conf = OmegaConf.create(self.default_conf)
        conf = OmegaConf.merge(MultiViewBenchmark.default_conf, self.default_conf, conf)
        super().__init__(conf)

    export_keys = ["gaussians"]

    optional_export_keys = [
        "context_rgb",
        "target_rgb",
        "context_depth",
        "target_depth",
        "pred_pose",
        "bg_color",
    ]

    def _init(self, conf):
        from splatfactory.geometry.pose_estimator import PoseEstimator

        self.eval_data_conf = conf.data
        self.pose_estimator = PoseEstimator(conf.eval.pose_estimator).to(tools.get_device())

    def get_dataloader(self, data_conf=None):
        data_conf = data_conf if data_conf else self.conf.data
        if self.max_samples is not None:
            logger.warning(
                f"max_samples={self.max_samples} set; forcing num_workers=0 "
                "(IterableDataset enforces max_samples per-worker)."
            )
            data_conf = OmegaConf.merge(
                data_conf, {"max_samples": self.max_samples, "num_workers": 0}
            )
        dataset = datasets.get_dataset(data_conf["name"])(data_conf, split="test")
        return dataset.get_loader()

    def get_predictions(self, experiment_dir, model=None, overwrite=False):
        pred_file = experiment_dir / "predictions.h5"
        if not pred_file.exists() or overwrite:
            if model is None:
                model = io.load_model(self.conf.model, self.conf.checkpoint)
            export_predictions(
                self.get_dataloader(self.conf.data),
                model,
                pred_file,
                keys=self.export_keys,
                optional_keys=self.optional_export_keys,
            )
        return pred_file

    @staticmethod
    def _batched_render(gaussians, cameras, poses, batch_size=10, **render_kwargs):
        """Render views in batches to avoid OOM. Returns (rgb, activated_count)."""
        n = cameras.shape[0]
        backgrounds = render_kwargs.pop("backgrounds", None)
        activated = set()
        chunks = []
        for i in range(0, n, batch_size):
            cs = min(batch_size, n - i)
            kw = render_kwargs
            if backgrounds is not None:
                kw = {**kw, "backgrounds": backgrounds[:cs]}
            out = gaussians.render_view(cameras=cameras[i : i + cs], poses=poses[i : i + cs], **kw)
            chunks.append(out["rendering"].cpu())
            if "activated" in out.get("info", {}):
                activated.update(torch.where(out["info"]["activated"][0] > 0)[0].cpu().tolist())
        return torch.cat(chunks).to(cameras.data_.device), len(activated)

    def _resize_views_to_render(self, data, pred):
        """Downsample images + scale cameras to eval.render_resize once.

        Pose alignment, render, and metric all subsequently operate at
        render_resize. Same filter used everywhere as the dataset preprocessor.
        """
        from splatfactory.geometry import Camera
        from splatfactory.utils.image import resize_to_cover

        rs = self.conf.eval.render_resize
        method = self.conf.data.preprocessing.get("resize_method", "pil_lanczos")

        for view in ("context", "target"):
            if view not in data or "image" not in data[view]:
                continue
            v = data[view]
            h, w = v["image"].shape[-2:]
            if (h, w) == (rs, rs):
                continue
            new_imgs = torch.stack(
                [resize_to_cover(im, rs, rs, method=method)[0] for im in v["image"]]
            )
            new_cam = self._scale_camera_to_size(v["camera"], rs)
            data[view] = {**v, "image": new_imgs, "camera": new_cam}

        if "pred_pose" in pred:
            for k in ("context_camera", "target_camera"):
                if k in pred["pred_pose"]:
                    cam = Camera(pred["pred_pose"][k]["data_"])
                    pred["pred_pose"][k] = {"data_": self._scale_camera_to_size(cam, rs).data_}

        for k in ("context_rgb", "target_rgb"):
            if k in pred and pred[k] is not None:
                imgs = pred[k]
                h, w = imgs.shape[-2:]
                if (h, w) != (rs, rs):
                    pred[k] = torch.stack(
                        [resize_to_cover(im, rs, rs, method=method)[0] for im in imgs]
                    )

        return data, pred

    @staticmethod
    def _compute_view_metrics(pred_imgs, gt_imgs, prefix):
        """Compute PSNR/SSIM/LPIPS averaged over views."""
        return {
            f"{prefix}-psnr": calculate_psnr(pred_imgs, gt_imgs).mean(),
            f"{prefix}-ssim": calculate_ssim(pred_imgs, gt_imgs).mean(),
            f"{prefix}-lpips": calculate_lpips(pred_imgs, gt_imgs).mean(),
        }

    @staticmethod
    def _scale_camera_to_size(camera, new_size):
        """Scale a Camera so each view's (W, H) becomes (new_size, new_size)."""
        s = new_size / camera.size  # camera.size is [..., 2] = [W, H]
        return camera.scale(s)

    def _align_poses(self, pred, data):
        from splatfactory.gaussians import Gaussians
        from splatfactory.geometry import Camera, Pose

        gaussians = Gaussians(pred["gaussians"]["data_"])

        # Use predicted cameras/poses if available, otherwise GT
        if "pred_pose" in pred:
            context_camera = Camera(pred["pred_pose"]["context_camera"]["data_"])
            target_camera = Camera(pred["pred_pose"]["target_camera"]["data_"])
            context_init_pose = Pose(pred["pred_pose"]["context_pose"]["data_"])
            target_init_pose = Pose(pred["pred_pose"]["target_pose"]["data_"])
        else:
            context_camera = data["context"]["camera"]
            target_camera = data["target"]["camera"]
            context_init_pose = data["context"]["pose"]
            target_init_pose = data["target"]["pose"]

        bs = self.conf.eval.pose_estimator.get("batch_size", 10)

        bg_color = pred.get("bg_color")

        def _make_bg(n):
            """Build backgrounds tensor for n views, or empty dict if no bg_color."""
            if bg_color is None:
                return {}
            device = gaussians.means.device
            if isinstance(bg_color, torch.Tensor):
                bg = bg_color.detach().clone().to(device=device).float()
            else:
                bg = torch.tensor(bg_color, device=device).float()
            return {"backgrounds": bg.unsqueeze(0).expand(n, -1)}

        render_kwargs = _make_bg(bs)

        # align context poses (optional - expensive at high view counts)
        if self.conf.eval.align_context:
            pose_opt = self.pose_estimator(
                data["context"]
                | {
                    "gaussians": gaussians,
                    "camera": context_camera,
                    "pose": context_init_pose,
                    "render_kwargs": _make_bg(context_camera.shape[0]),
                }
            )["pose"][0]
            context_rgb, ctx_activated = self._batched_render(
                gaussians, context_camera, pose_opt, bs, **render_kwargs
            )
        else:
            context_rgb = None
            ctx_activated = 0

        # align target poses
        pose_opt = self.pose_estimator(
            data["target"]
            | {
                "gaussians": gaussians,
                "camera": target_camera,
                "pose": target_init_pose,
                "render_kwargs": _make_bg(target_camera.shape[0]),
            }
        )["pose"][0]
        target_rgb, tgt_activated = self._batched_render(
            gaussians, target_camera, pose_opt, bs, **render_kwargs
        )

        return {
            "aligned_context_rgb": context_rgb,
            "aligned_target_rgb": target_rgb,
            "total_gaussians": gaussians.num_gaussians,
            "activated_gaussians": max(ctx_activated, tgt_activated),
        }

    def _extra_sample_fields(self, data):
        """Override to collect additional per-sample fields (e.g. overlap for RE10K)."""
        return {}

    def run_eval(self, loader, pred_file):
        assert pred_file.exists()
        results = defaultdict(list)

        cache_loader = CacheLoader({"path": str(pred_file), "collate": None}).eval()

        pbar = tqdm(loader, total=len(loader), desc="Evaluate", ncols=100)
        for i, data in enumerate(pbar):
            pred = cache_loader(data)
            data = mappings.remove_batch_dim(data)

            pred = mappings.batch_to_device(pred, tools.get_device(), non_blocking=False)
            data = mappings.batch_to_device(data, tools.get_device(), non_blocking=False)

            if self.conf.eval.render_resize is not None:
                data, pred = self._resize_views_to_render(data, pred)

            results_i = {}

            if self.conf.eval.align_poses:
                res = self._align_poses(pred, data)
                if res["aligned_context_rgb"] is not None:
                    pred["context_rgb"] = res["aligned_context_rgb"]
                pred["target_rgb"] = res["aligned_target_rgb"]
                results_i["total_gaussians"] = res["total_gaussians"]
                results_i["activated_gaussians"] = res["activated_gaussians"]

            if "context_rgb" in pred and pred["context_rgb"] is not None:
                results_i.update(
                    self._compute_view_metrics(
                        pred["context_rgb"], data["context"]["image"], "context"
                    )
                )
            results_i.update(
                self._compute_view_metrics(pred["target_rgb"], data["target"]["image"], "target")
            )

            results_i["names"] = data["name"]
            results_i["scenes"] = data["scene"]
            results_i.update(self._extra_sample_fields(data))

            results_i = mappings.batch_to_device(results_i, torch.device("cpu"))
            for k, v in results_i.items():
                results[k].append(v)

            m_psnr = metrics.AverageMetric(np.array(results["target-psnr"]))
            m_lpips = metrics.AverageMetric(np.array(results["target-lpips"]))
            pbar.set_postfix(
                {"PSNR": f"{m_psnr.compute():.2f}", "LPIPS": f"{m_lpips.compute():.3f}"}
            )

            # Release per-scene tensors so the caching allocator can defragment
            # before the next scene's pose-align allocation (~6 GiB at 64v
            # AnySplat / DA3 on 4090). Without this, reserved-but-unallocated
            # memory fragments and OOMs after a few scenes.
            del pred, data, results_i
            if self.conf.eval.align_poses:
                del res
            torch.cuda.empty_cache()

        summaries = {}
        for k, v in results.items():
            arr = np.array(v)
            if not np.issubdtype(arr.dtype, np.number):
                continue
            summaries[f"mean-{k}"] = round(metrics.AverageMetric(arr).compute(), 3)

        # per-scene breakdown (small benchmarks only)
        scenes = results.get("scenes", [])
        unique_scenes = set(scenes)
        if scenes and len(unique_scenes) < 25:
            scene_metrics = defaultdict(lambda: defaultdict(list))
            for idx, scene in enumerate(scenes):
                for k in [
                    "target-psnr",
                    "target-ssim",
                    "target-lpips",
                    "total_gaussians",
                    "activated_gaussians",
                ]:
                    if k in results and idx < len(results[k]):
                        scene_metrics[scene][k].append(float(results[k][idx]))
            for scene, vals in sorted(scene_metrics.items()):
                for k, v in vals.items():
                    summaries[f"scene-{scene}-{k}"] = round(np.mean(v), 3)

        figures = {}
        return summaries, figures, results
