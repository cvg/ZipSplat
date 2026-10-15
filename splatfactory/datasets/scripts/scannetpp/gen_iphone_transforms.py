"""Generate missing iPhone `nerfstudio/transforms.json` from `iphone/colmap/`.

Some ScanNet++ scenes ship the iPhone COLMAP registration but not the derived
nerfstudio transforms, so convert.py skips them. COLMAP reproduces the nerfstudio
transforms exactly -- intrinsics identical, c2w poses to ~2e-5 (verified) -- so this
writes the missing files from colmap. convert.py then needs no special-casing.

Originals are never overwritten. Generated files carry `"generated_from_colmap": true`
so they are distinguishable:

    grep -l generated_from_colmap data/scannetpp_v2/data/*/iphone/nerfstudio/transforms.json

Usage:
    python -m splatfactory.datasets.scripts.scannetpp.gen_iphone_transforms --num-workers 8
    python -m splatfactory.datasets.scripts.scannetpp.gen_iphone_transforms --dry-run

Author: Alexander Veicht
"""

import argparse
import json
from collections import Counter
from functools import partial
from pathlib import Path

import numpy as np
from tqdm import tqdm

from splatfactory import get_logger
from splatfactory.datasets.scripts.utils import parallel_map

logger = get_logger(__name__)

SPLIT_FILES = {"train": "nvs_sem_train.txt", "test": "nvs_sem_val.txt"}


def _read_split(path: Path) -> list[str]:
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def _quat_to_R(qw, qx, qy, qz) -> np.ndarray:
    n = (qw * qw + qx * qx + qy * qy + qz * qz) ** 0.5
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ]
    )


def _colmap_w2c_to_ns_c2w(w2c: np.ndarray) -> np.ndarray:
    """COLMAP world-to-camera -> nerfstudio cam2world (inverse of ns->colmap axis ops)."""
    c2w = np.linalg.inv(w2c)
    c2w[0:3, 1:3] *= -1  # undo negate cols 1,2
    c2w = c2w[[1, 0, 2, 3], :]  # undo swap rows 0,1
    c2w[2, :] *= -1  # undo negate row 2
    return c2w


def _colmap_to_transforms(colmap: Path) -> dict:
    """Reconstruct the nerfstudio transforms dict from an iPhone colmap model."""
    cam = next(l.split() for l in open(colmap / "cameras.txt") if not l.startswith("#"))
    w, h = int(cam[2]), int(cam[3])
    params = list(map(float, cam[4:]))
    fx, fy, cx, cy = params[:4]
    k1, k2, p1, p2 = (params[4:8] + [0.0, 0.0, 0.0, 0.0])[:4]

    frames = []
    lines = [l for l in open(colmap / "images.txt") if not l.startswith("#")]
    for i in range(0, len(lines), 2):  # one image = a pose line + a 2D-points line
        p = lines[i].split()
        qw, qx, qy, qz, tx, ty, tz = map(float, p[1:8])
        name = p[9]
        w2c = np.eye(4)
        w2c[:3, :3] = _quat_to_R(qw, qx, qy, qz)
        w2c[:3, 3] = [tx, ty, tz]
        frames.append({"file_path": name, "transform_matrix": _colmap_w2c_to_ns_c2w(w2c).tolist()})
    frames.sort(key=lambda f: f["file_path"])
    return {
        "camera_model": "OPENCV",
        "w": w,
        "h": h,
        "fl_x": fx,
        "fl_y": fy,
        "cx": cx,
        "cy": cy,
        "k1": k1,
        "k2": k2,
        "p1": p1,
        "p2": p2,
        "generated_from_colmap": True,
        "frames": frames,
    }


def gen_scene(scene_id: str, data_root: Path, overwrite: bool, dry_run: bool):
    iph = data_root / "data" / scene_id / "iphone"
    out = iph / "nerfstudio" / "transforms.json"
    colmap = iph / "colmap"
    if out.exists() and not overwrite:
        return scene_id, "exists", 0
    if not (colmap / "images.txt").exists():
        return scene_id, "no-colmap", 0
    if dry_run:
        return scene_id, "would-gen", 0
    tr = _colmap_to_transforms(colmap)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(tr))
    tmp.rename(out)
    return scene_id, "generated", len(tr["frames"])


def main():
    parser = argparse.ArgumentParser(description="Generate missing iPhone transforms from colmap")
    parser.add_argument("--data-root", type=Path, default=Path("data/scannetpp_v2"))
    parser.add_argument("--split", nargs="+", choices=["train", "test"], default=["train", "test"])
    parser.add_argument("--scene-ids", nargs="+", default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--overwrite", action="store_true", help="also regenerate scenes that already have one"
    )
    parser.add_argument("--dry-run", action="store_true", help="report what would be generated")
    args = parser.parse_args()

    scene_ids = args.scene_ids
    if scene_ids is None:
        scene_ids = []
        for split in args.split:
            p = args.data_root / "splits" / SPLIT_FILES[split]
            if p.exists():
                scene_ids += _read_split(p)
        scene_ids = sorted(set(scene_ids))

    fn = partial(
        gen_scene, data_root=args.data_root, overwrite=args.overwrite, dry_run=args.dry_run
    )
    counts: Counter = Counter()
    total_frames = 0
    results = parallel_map(fn, scene_ids, num_workers=args.num_workers, skip_label=lambda s: s)
    for _sid, status, n in tqdm(results, total=len(scene_ids), desc="gen", ncols=100):
        counts[status] += 1
        total_frames += n
    logger.info(f"Done: {dict(counts)}, {total_frames} frames written")


if __name__ == "__main__":
    from splatfactory import logger  # noqa: F811

    main()
