"""Voxelise the DeepSense LiDAR point clouds into the LiDAR grids the benchmark loads.

DeepSense ships one ASCII `.ply` point cloud per sample, but `alwl.data.InputStore`
expects a (20,20,20) occupancy grid saved as `.npy` -- the shape FLASH already publishes. This
script closes that gap so `modality: lidar` runs on both datasets under one encoding.

The grid is binary occupancy, matching FLASH's convention (0 empty, 1 occupied), over a box
centred on the sensor. Points outside the box are dropped; the defaults keep ~82% of them.
"""

from __future__ import annotations

import argparse
import logging
import re
import zipfile
from multiprocessing import Pool
from pathlib import Path

import numpy as np

GRID = 20
LOG = logging.getLogger("voxelize_deepsense_lidar")
_ARGS: dict = {}


def read_ply(handle) -> np.ndarray:
    """Return the Nx4 (x, y, z, intensity) table of an ASCII PLY, header length unassumed."""
    rows = []
    in_body = False
    for raw in handle:
        line = raw.decode("ascii", "replace").strip()
        if not in_body:
            if line == "end_header":
                in_body = True
            continue
        if line:
            rows.append(line.split())
    if not rows:
        raise ValueError("PLY has no vertices")
    return np.asarray(rows, dtype=np.float64)


def voxelise(points: np.ndarray, xy_half: float, z_lo: float, z_hi: float) -> np.ndarray:
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    inside = (np.abs(x) <= xy_half) & (np.abs(y) <= xy_half) & (z >= z_lo) & (z <= z_hi)
    grid = np.zeros((GRID, GRID, GRID), dtype=np.int8)
    if not inside.any():
        return grid
    xi = np.clip(((x[inside] + xy_half) / (2 * xy_half) * GRID).astype(int), 0, GRID - 1)
    yi = np.clip(((y[inside] + xy_half) / (2 * xy_half) * GRID).astype(int), 0, GRID - 1)
    zi = np.clip(((z[inside] - z_lo) / (z_hi - z_lo) * GRID).astype(int), 0, GRID - 1)
    grid[xi, yi, zi] = 1
    return grid


def _one(member: str):
    out = Path(_ARGS["out"]) / (Path(member).stem + ".npy")
    if out.exists():
        return member, -1.0
    if _ARGS.get("zip"):
        with zipfile.ZipFile(_ARGS["zip"]) as zf, zf.open(member) as fh:
            points = read_ply(fh)
    else:
        with open(member, "rb") as fh:
            points = read_ply(fh)
    grid = voxelise(points, _ARGS["xy_half"], _ARGS["z_lo"], _ARGS["z_hi"])
    np.save(out, grid)
    return member, float(grid.mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--zip", help="scenario archive holding unit1/lidar_data")
    source.add_argument("--dir", type=Path, help="extracted unit1/lidar_data folder")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--xy-half", type=float, default=20.0, help="half-width in metres")
    parser.add_argument("--z-lo", type=float, default=-3.0)
    parser.add_argument("--z-hi", type=float, default=5.0)
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    args.out.mkdir(parents=True, exist_ok=True)
    if args.zip:
        with zipfile.ZipFile(args.zip) as zf:
            names = zf.namelist()
    else:
        names = [str(p) for p in args.dir.glob("*.ply")]
    members = sorted(
        (n for n in names if re.search(r"lidar_data_\d+\.ply$", n)),
        key=lambda n: int(re.findall(r"(\d+)\.ply$", n)[0]),
    )
    if not members:
        raise FileNotFoundError(f"No lidar_data_*.ply in {args.zip or args.dir}")
    LOG.info("%d point clouds, %d workers", len(members), args.workers)

    _ARGS.update(
        zip=args.zip, out=str(args.out), xy_half=args.xy_half, z_lo=args.z_lo, z_hi=args.z_hi
    )
    occupancy = []
    with Pool(args.workers, initializer=_ARGS.update, initargs=(dict(_ARGS),)) as pool:
        for i, (_m, occ) in enumerate(pool.imap_unordered(_one, members, chunksize=8), 1):
            if occ >= 0:
                occupancy.append(occ)
            if i % 500 == 0 or i == len(members):
                LOG.info("%d/%d", i, len(members))
    if occupancy:
        LOG.info(
            "occupied cells per grid: mean %.2f%% (min %.2f%%, max %.2f%%)",
            np.mean(occupancy) * 100,
            np.min(occupancy) * 100,
            np.max(occupancy) * 100,
        )
    LOG.info("wrote %d grids to %s", len(members), args.out)


if __name__ == "__main__":
    main()
