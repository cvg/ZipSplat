"""Multi-view evaluation pipeline, the shared base of all benchmarks.

Author: Alexander Veicht
"""

from collections import defaultdict

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from splatfactory import datasets, get_logger
from splatfactory.eval import eval_pipeline, io
from splatfactory.gaussians import Gaussians
from splatfactory.geometry import Camera, Pose
from splatfactory.geometry.pose_aligner import PoseAligner
from splatfactory.models import metrics
from splatfactory.models.cache_loader import CacheLoader
from splatfactory.utils import mappings, tools
from splatfactory.utils.export import export_predictions
from splatfactory.utils.image import crop_to_ar, resize_to_cover
from splatfactory.utils.metrics import AverageMetric

logger = get_logger(__name__)


class MultiViewBenchmark(eval_pipeline.EvalPipeline):
    default_conf = {
        "data": {
            "train_batch_size": 1,
            "val_batch_size": 1,
            "test_batch_size": 1,
            "num_workers": 0,
            # the loader only forwards it when num_workers > 0
            "prefetch_factor": None,
            "max_pose_jump_ratio": 0,  # off at eval - videos-only filter
            "view_sampler": {
                "name": "eval_sampler",
            },
            "preprocessing": {
                "resize": 252,
            },
        },
        "model": {"timeit": True},  # report the forward wall clock as `forward_time`
        "eval": {
            "align_poses": True,
            "align_context": False,
            # Final render + metric resolution. None = use target_camera as-is.
            # When set, pose alignment still runs at T (better signal), but the
            # final aligned render scales target_camera to (render_resize,
            # render_resize) and the GT target is interpolated to the same size.
            "render_resize": None,
            # Target pose alignment with the LM PoseAligner. batch_size is the view batch of the
            # final metric render and the LM's view chunk, every other key goes to the aligner.
            "pose_estimator": {"batch_size": 10},
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
        "pose_pred",
        "camera_pred",
        "forward_time",
    ]

    def _init(self, conf):

        self.eval_data_conf = conf.data
        pe_conf = OmegaConf.to_container(conf.eval.pose_estimator, resolve=True)
        self.pose_estimator = PoseAligner(pe_conf).to(tools.get_device())
        # rasterizer args of the aligned render, declared by the model as `model.render_kwargs`
        self.render_kwargs = dict(OmegaConf.select(conf, "model.render_kwargs", default={}) or {})

    def get_dataloader(self, data_conf=None):
        data_conf = data_conf if data_conf else self.conf.data
        if self.max_samples is not None:
            logger.warning(
                f"max_samples={self.max_samples} set, forcing num_workers=0 "
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
                # No metric reads the context renders, and at high view counts they are most
                # of the work.
                batch_extras={"render_context": False},
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
        return torch.cat(chunks).to(cameras.device), len(activated)

    def _resize_views_to_render(self, data, pred):
        """Cover-resize and center-crop images and cameras to a square eval.render_resize.

        Pose alignment, render and metric all run at render_resize x render_resize, with the
        dataset preprocessor's resize filter.
        """

        rs = self.conf.eval.render_resize
        # the same filter as the encoder-side resize, so the metric GT holds no extra detail
        method = self.conf.data.preprocessing.get("resize_method", "torch_bilinear_aa")

        def _cover_square(im):
            """Cover-resize to >= rs then center-crop to (rs, rs). Returns (img, 3x3 T)."""
            im, t_resize = resize_to_cover(im, rs, rs, method=method)
            im, t_crop = crop_to_ar(im, 1.0)
            return im, t_crop @ t_resize

        for view in ("context", "target"):
            if view not in data or "image" not in data[view]:
                continue
            v = data[view]
            h, w = v["image"].shape[-2:]
            if (h, w) == (rs, rs):
                continue
            imgs, transform = [], None
            for im in v["image"]:
                im, transform = _cover_square(im)  # transform identical across views
                imgs.append(im)
            new_imgs = torch.stack(imgs)
            t = torch.as_tensor(transform, dtype=v["camera"].dtype, device=v["camera"].device)
            new_cam = v["camera"].compose_image_transform(t)
            data[view] = {**v, "image": new_imgs, "camera": new_cam}

        for k in ("context_rgb", "target_rgb"):
            if k in pred and pred[k] is not None:
                imgs = pred[k]
                h, w = imgs.shape[-2:]
                if (h, w) != (rs, rs):
                    pred[k] = torch.stack([_cover_square(im)[0] for im in imgs])

        return data, pred

    @staticmethod
    def _compute_view_metrics(pred_imgs, gt_imgs, prefix):
        """PSNR, SSIM and LPIPS averaged over the views."""
        return {
            f"{prefix}-psnr": metrics.calculate_psnr(pred_imgs, gt_imgs).mean(),
            f"{prefix}-ssim": metrics.calculate_ssim(pred_imgs, gt_imgs).mean(),
            f"{prefix}-lpips": metrics.calculate_lpips(pred_imgs, gt_imgs, batch_size=8).mean(),
        }

    def _align_poses(self, pred, data):
        """Align the GT target (and optionally context) poses to the predicted Gaussians."""

        gaussians = Gaussians(pred["gaussians"]["data_"])
        context_camera = data["context"]["camera"]
        target_camera = data["target"]["camera"]

        bs = self.conf.eval.pose_estimator.get("batch_size", 10)

        # align context poses (optional - expensive at high view counts)
        if self.conf.eval.align_context:
            pose_opt = self.pose_estimator(
                data["context"]
                | {
                    "gaussians": gaussians,
                    "camera": context_camera,
                    "pose": data["context"]["pose"],
                    "render_kwargs": {},
                }
            )["pose"][0]
            context_rgb, ctx_activated = self._batched_render(
                gaussians, context_camera, pose_opt, bs, **self.render_kwargs
            )
        else:
            context_rgb = None
            ctx_activated = 0

        pose_opt = self.pose_estimator(
            data["target"]
            | {
                "gaussians": gaussians,
                "camera": target_camera,
                "pose": data["target"]["pose"],
                "render_kwargs": self.render_kwargs,
            }
        )["pose"][0]
        target_rgb, tgt_activated = self._batched_render(
            gaussians, target_camera, pose_opt, bs, **self.render_kwargs
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

    def _calculate_pose_metrics(self, pred, data):
        """Per-pair context-pose errors (deg) vs GT, plus the relative fov error.

        Empty when the model predicts no poses. AUC/RRA/RTA are pooled over all pairs in
        run_eval.
        """

        if "pose_pred" not in pred:
            return {}
        pose = Pose(pred["pose_pred"]["data_"])
        camera = Camera(pred["camera_pred"]["data_"])

        r_deg, t_deg = metrics.calculate_pose_pair_errors(pose, data["context"]["pose"])
        gt_fov = data["context"]["camera"].hfov
        return {
            "pose-r": r_deg,
            "pose-t": t_deg,
            "pose-fov-err": ((camera.hfov - gt_fov).abs() / gt_fov).median(),
        }

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
            elif "gaussians" in pred:

                results_i["total_gaussians"] = Gaussians(pred["gaussians"]["data_"]).num_gaussians

            if "context_rgb" in pred and pred["context_rgb"] is not None:
                results_i.update(
                    self._compute_view_metrics(
                        pred["context_rgb"], data["context"]["image"], "context"
                    )
                )
            results_i.update(
                self._compute_view_metrics(pred["target_rgb"], data["target"]["image"], "target")
            )

            if "forward_time" in pred:
                results_i["forward_time"] = float(pred["forward_time"])

            results_i.update(self._calculate_pose_metrics(pred, data))

            results_i["names"] = data["name"]
            results_i["scenes"] = data["scene"]
            results_i.update(self._extra_sample_fields(data))

            results_i = mappings.batch_to_device(results_i, torch.device("cpu"))
            for k, v in results_i.items():
                results[k].append(v)

            m_psnr = AverageMetric(np.array(results["target-psnr"]))
            m_lpips = AverageMetric(np.array(results["target-lpips"]))
            pbar.set_postfix(
                {"PSNR": f"{m_psnr.compute():.2f}", "LPIPS": f"{m_lpips.compute():.3f}"}
            )

            # free the scene's tensors so the allocator can defragment before the next one
            del pred, data, results_i
            if self.conf.eval.align_poses:
                del res
            torch.cuda.empty_cache()

        summaries = {}
        for k, v in results.items():
            # per-pair pose errors are ragged when scenes have different view counts. The
            # flattened mean equals the 2D one AverageMetric computes for equal counts
            ragged = len({np.shape(x) for x in v}) > 1
            arr = np.concatenate([np.asarray(x).ravel() for x in v]) if ragged else np.array(v)
            if not np.issubdtype(arr.dtype, np.number):
                continue
            summaries[f"mean-{k}"] = round(AverageMetric(arr).compute(), 3)

        # Pose accuracy pools ALL pairs of ALL scenes: a mean of per-scene AUCs is a
        # different (and wrong) number.
        if "pose-r" in results:
            r = np.concatenate([np.asarray(x).ravel() for x in results["pose-r"]])
            t = np.concatenate([np.asarray(x).ravel() for x in results["pose-t"]])
            for th in (1, 3, 5, 10, 15, 20, 30):
                summaries[f"pose-auc_{th}"] = round(metrics.calculate_pose_auc(r, t, th), 4)
            for th in (5, 15, 30):
                summaries[f"pose-rra_{th}"] = round(float((r < th).mean()), 4)
                summaries[f"pose-rta_{th}"] = round(float((t < th).mean()), 4)
            summaries["pose-med_r"] = round(float(np.median(r)), 3)
            summaries["pose-med_t"] = round(float(np.median(t)), 3)
            summaries["pose-pairs"] = int(len(r))
            # Scene-mean aggregation (DA3-BENCH AUC, MASt3R-SfM RRA / RTA), next to the pooled
            # keys: metric over each sample's pairs (one sample per scene), then the mean.
            per_scene = [
                (np.asarray(r_i).ravel(), np.asarray(t_i).ravel())
                for r_i, t_i in zip(results["pose-r"], results["pose-t"])
            ]
            for th in (3, 5, 15, 30):
                auc = [metrics.calculate_pose_auc(r_i, t_i, th) for r_i, t_i in per_scene]
                summaries[f"pose-scene_auc_{th}"] = round(float(np.mean(auc)), 4)
            for th in (5, 15, 30):
                rra = np.mean([(r_i < th).mean() for r_i, _ in per_scene])
                rta = np.mean([(t_i < th).mean() for _, t_i in per_scene])
                summaries[f"pose-scene_rra_{th}"] = round(float(rra), 4)
                summaries[f"pose-scene_rta_{th}"] = round(float(rta), 4)

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
