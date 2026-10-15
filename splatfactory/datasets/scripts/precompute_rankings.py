"""Precompute per-view covisibility rankings for the `rank` view sampler (tar shards).

For every scene, every ordered pair of views is scored by depth covisibility: a
--grid x --grid sample of the source view's GT depth is unprojected, projected into
the target view and counted when it lands inside the image, in front of the camera,
and within --tol (relative) of the target's full-resolution depth (occlusion-aware).
Views are ranked by mutual covisibility (the smaller of the two directions). The result is
written into each scene IN PLACE (shard rewrite via .tmp + atomic rename, as extract_depth.py)
as four tar members right after the scene's meta.json, the format `view_sampler/rank.py` reads:
    {key}.rankings.npy  (N, k) int16    k most covisible views per view, self first
    {key}.overlaps.npy  (N, k) float32  min of both directions
    {key}.seen_by.npy   (N, k) float32  fraction of the reference's pixels seen by the view
    {key}.sees.npy      (N, k) float32  fraction of the view's pixels seen by the reference
Both directions are stored so a different score never needs a recompute. Pose distance is
not used anywhere: in room scans the camera turns in place, so pose-nearest views often
share nothing. Scenes that already carry the members are skipped, so a rerun is a no-op.

Only the depth / camera / pose members of each tar are read (image bytes are
skipped). All pairs are scored on the GPU; a 1,400-view scene takes a few seconds.
`--from-json` packs a rankings.json produced by the earlier version of this script into
the shards without re-scoring (no GPU needed).

Usage:
    python -m splatfactory.datasets.scripts.precompute_rankings \
        data/scannetpp_v2/720x1080/dslr/train-scenes data/scannetpp_v2/720x1080/dslr/test-scenes
    python -m splatfactory.datasets.scripts.precompute_rankings \
        data/scannetpp_v2/720x1080/dslr/train-scenes data/scannetpp_v2/720x1080/dslr/test-scenes \
        --from-json data/scannetpp_v2/720x1080/rankings.json

Author: Alexander Veicht
"""

import argparse
import io
import json
import os
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from splatfactory import get_logger
from splatfactory.datasets.utils import io as sio
from splatfactory.geometry import Camera, Pose

logger = get_logger(__name__)

MEMBERS = sio.RANKING_MEMBERS


def iter_scene_geometry(tar_path: Path):
    """Yield (key, scene) with meta/cameras/poses/depths/depth_ranges only."""
    scenes: dict[str, dict] = {}
    with tarfile.open(tar_path) as tar:
        for m in tar:
            if not m.isfile():
                continue
            name = m.name
            if name.endswith(".meta.json"):
                key, field, val = name[:-10], "meta", json.load(tar.extractfile(m))
            elif name.endswith(".cameras.npy"):
                key, field, val = (
                    name[:-12],
                    "cameras",
                    np.load(io.BytesIO(tar.extractfile(m).read())),
                )
            elif name.endswith(".poses.npy"):
                key, field, val = (
                    name[:-10],
                    "poses",
                    np.load(io.BytesIO(tar.extractfile(m).read())),
                )
            elif name.endswith(".depth_ranges.json"):
                key, field, val = name[:-18], "depth_ranges", json.load(tar.extractfile(m))
            elif ".dep" in name:
                stem, rest = name.rsplit(".dep", 1)
                key, field, val = (
                    stem,
                    ("depths", int(rest.split(".")[0])),
                    tar.extractfile(m).read(),
                )
            else:
                continue
            s = scenes.setdefault(key, {"depths": {}})
            if isinstance(field, tuple):
                s["depths"][field[1]] = val
            else:
                s[field] = val
    for key, s in scenes.items():
        if s.get("meta", {}).get("has_depth") and "cameras" in s and "poses" in s:
            yield key, s


def decode_depths(scene: dict, n: int, pool: ThreadPoolExecutor) -> torch.Tensor:
    def one(i):
        if i not in scene["depths"]:
            return None
        d_min, d_max = scene["depth_ranges"][i]
        return sio.decode_depth(scene["depths"][i], d_min, d_max)

    first = next(m for m in (one(i) for i in range(n)) if m is not None)
    D = torch.full((n, *first.shape), -1.0, dtype=torch.float16)  # fp16, no float32 copies

    def fill(i):
        m = one(i)
        if m is not None:
            D[i] = torch.from_numpy(m).to(torch.float16)

    list(pool.map(fill, range(n)))
    return D


@torch.no_grad()
def covisibility(scene: dict, D: torch.Tensor, grid: int, tol: float, dev: str) -> torch.Tensor:
    """(N, N) symmetric covisibility scores in [0, 1], diagonal 1."""
    D = D.to(dev, torch.float16)
    N, H, W = D.shape
    cams = Camera(torch.as_tensor(scene["cameras"], dtype=torch.float32).to(dev))
    poses = Pose(torch.as_tensor(scene["poses"], dtype=torch.float32).to(dev))
    w2c = poses.inv()
    ys = torch.linspace(0, H - 1, grid, device=dev).round().long()
    xs = torch.linspace(0, W - 1, grid, device=dev).round().long()
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    yy, xx = yy.reshape(-1), xx.reshape(-1)
    p2 = torch.stack([xx, yy], -1).float() + 0.5  # [P, 2]
    P = len(yy)
    rows = torch.arange(N, device=dev)[:, None]
    S = torch.zeros(N, N, device=dev)
    for a in range(N):
        d = D[a, yy, xx].float()
        ok = d > 0
        if not ok.any():
            continue
        p_cam = torch.cat([cams[a].normalize(p2), torch.ones(P, 1, device=dev)], -1) * d[:, None]
        pw = poses[a].transform(p_cam)
        pc = w2c.transform(pw[None].expand(N, P, 3))  # [N, P, 3] in every target
        uv, vis = cams.world2image(pc)
        u = uv[..., 0].long().clamp(0, W - 1)
        v = uv[..., 1].long().clamp(0, H - 1)
        db = D[rows, v, u].float()
        z = pc[..., 2]
        hit = vis & (z > 0) & (db > 0) & ((z - db).abs() < tol * db.clamp_min(1e-6)) & ok[None]
        S[a] = hit.float().sum(1) / ok.float().sum()
    S.fill_diagonal_(1.0)
    return S  # S[a, b] = fraction of a's pixels seen (and depth-consistent) in b


def score_scene(scene: dict, pool: ThreadPoolExecutor, args) -> dict[str, np.ndarray]:
    """The four (N, k) ranking arrays of one scene, scored on the GPU."""
    n = scene["meta"]["num_views"]
    D = decode_depths(scene, n, pool)
    S = covisibility(scene, D, args.grid, args.tol, args.device)
    k = min(args.k, n)
    # Rank by mutual covisibility (min of both directions): the mean lets a wide
    # view that merely contains the reference's area score high while itself being
    # barely covered (ref->view 0.49, view->ref 0.08). Both directions are stored
    # so any other score can be derived at load time without a recompute.
    M = torch.minimum(S, S.T)
    order = torch.argsort(M, dim=1, descending=True)[:, :k]
    r3 = lambda t: np.round(t.cpu().numpy(), 3)  # noqa: E731
    return {
        "rankings": order.cpu().numpy(),
        "overlaps": r3(torch.gather(M, 1, order)),  # min(seen_by, sees)
        "seen_by": r3(torch.gather(S, 1, order)),  # ref's pixels seen by the view
        "sees": r3(torch.gather(S.T, 1, order)),  # the view's pixels seen by ref
    }


def insert_members(tar_path: Path, arrays_by_key: dict[str, dict[str, np.ndarray]]):
    """Rewrite the shard with each scene's ranking members right after its meta.json.

    Streaming copy into .tmp + atomic rename (as extract_depth.py). Appending to the tar
    would be cheaper but puts the members after every scene, and the training reader streams
    scenes as contiguous member blocks.
    """
    tmp_path = tar_path.with_name(tar_path.name + ".tmp")
    with tarfile.open(tar_path) as src, tarfile.open(tmp_path, "w") as dst:
        for m in src:
            dst.addfile(m, src.extractfile(m) if m.isfile() else None)
            if not m.name.endswith(".meta.json"):
                continue
            key = m.name[: -len(".meta.json")]
            if key not in arrays_by_key:
                continue
            arrays = arrays_by_key[key]
            assert arrays["rankings"].max() < 2**15, f"{key}: too many views for int16"
            for name in MEMBERS:
                dtype = np.int16 if name == "rankings" else np.float32
                sio._tar_add(dst, f"{key}.{name}.npy", sio._npy_bytes(arrays[name].astype(dtype)))
    os.rename(tmp_path, tar_path)


def scene_keys(tar_path: Path) -> tuple[list[str], set[str]]:
    """(all scene keys, keys that already carry rankings) from the member names alone."""
    with tarfile.open(tar_path) as tar:
        names = tar.getnames()
    keys = [n[: -len(".meta.json")] for n in names if n.endswith(".meta.json")]
    done = {n[: -len(".rankings.npy")] for n in names if n.endswith(".rankings.npy")}
    return keys, done


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("shard_dirs", nargs="+", type=Path)
    parser.add_argument("--from-json", type=Path, help="Pack this rankings.json instead of scoring")
    parser.add_argument("--k", type=int, default=100, help="Most covisible views stored per view")
    parser.add_argument("--grid", type=int, default=32, help="Source depth samples per axis")
    parser.add_argument("--tol", type=float, default=0.05, help="Relative depth tolerance")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--decode-threads", type=int, default=8)
    args = parser.parse_args()

    shards = sorted(p for d in args.shard_dirs for p in d.glob("*.tar"))
    if not shards:
        raise FileNotFoundError(f"No tar shards under {args.shard_dirs}")
    packed = None
    if args.from_json is not None:
        logger.info(f"Packing {args.from_json} into {len(shards)} shards")
        with open(args.from_json) as f:
            packed = json.load(f)
    else:
        logger.info(f"Ranking {len(shards)} shards by covisibility (in place)")

    n_written = n_skipped = n_missing = 0
    t0 = time.time()
    with ThreadPoolExecutor(args.decode_threads) as pool:
        for shard in tqdm(shards, desc="shards"):
            keys, done = scene_keys(shard)
            results: list[tuple[str, dict[str, np.ndarray]]] = []
            if packed is not None:
                for key in keys:
                    if key in done:
                        n_skipped += 1
                    elif key not in packed:
                        n_missing += 1
                    else:
                        results.append((key, {m: np.asarray(packed[key][m]) for m in MEMBERS}))
            else:
                for key, scene in iter_scene_geometry(shard):
                    if key in done:
                        n_skipped += 1
                    else:
                        results.append((key, score_scene(scene, pool, args)))
            if results:
                insert_members(shard, dict(results))
                n_written += len(results)
    logger.info(
        f"Wrote rankings for {n_written} scenes, skipped {n_skipped} already ranked, "
        f"{n_missing} without an entry, in {time.time() - t0:.0f}s"
    )


if __name__ == "__main__":
    main()
