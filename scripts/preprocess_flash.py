"""Convert the processed FLASH release into the sample table `scripts.build_sample_table` expects.

FLASH ships one directory per episode holding `rf.npz` (RSSI indexed by sector id),
`gps.npz` (local ENU metres) and the camera frames. This script flattens that tree into a
single parquet with the documented sample fields plus `gamma_0 .. gamma_33`.

Two properties of the release drive the options below:

* The TP-Link Talon radio defines 34 sectors (ids 1-31 and 61-63); the remaining columns of
  the 64-wide `rf` array are undefined and carry occasional spurious readings.
* Only part of the codebook is swept per sample -- on average 16 of the 34 defined sectors --
  so most gamma entries have no measurement behind them. `--unmeasured` decides what those
  become, and it changes the benchmark, so state the choice alongside any result.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

# Sector ids the radio actually defines; `rf.npz` is indexed by sector id directly.
DEFINED_SECTORS = list(range(1, 32)) + [61, 62, 63]

# FLASH stores local ENU metres. The release's own spreadsheet pins the exact affine back to
# WGS84 (fitted residual 0.00 m).
LAT0, LON0 = 42.338129, -71.087053
DEG_PER_M_LAT = 9.0000e-6
DEG_PER_M_LON = 1.21755e-5

# Synchronised streams are published at 10 Hz; only one episode ships absolute timestamps.
SAMPLE_PERIOD_S = 0.1009

LOG = logging.getLogger("preprocess_flash")


def _episodes(root: Path) -> list[Path]:
    return sorted(p.parent.parent for p in root.rglob("npz/rf.npz"))


def _scenario_of(episode: Path) -> str:
    # .../<Cat>/<scenario>/Extract/<run>/<episode_N>
    return episode.parent.name


def build(
    root: Path, images: Path, lidar_out: Path | None, unmeasured: str, min_measured: int
) -> pd.DataFrame:
    episodes = _episodes(root)
    if not episodes:
        raise FileNotFoundError(f"No */npz/rf.npz under {root}")
    LOG.info("found %d episodes", len(episodes))

    runs = sorted({_scenario_of(e) for e in episodes})
    scenario_id = {name: i + 1 for i, name in enumerate(runs)}

    rows = []
    dropped_no_measurement = 0
    for ep_index, episode in enumerate(episodes):
        rf = np.load(episode / "npz" / "rf.npz")["rf"]
        gps = np.load(episode / "npz" / "gps.npz")["gps"]
        if len(rf) != len(gps):
            raise ValueError(f"{episode}: rf has {len(rf)} rows, gps has {len(gps)}")

        gamma_lin = rf[:, DEFINED_SECTORS].astype(np.float64)
        measured = gamma_lin > 0
        keep = measured.sum(axis=1) >= min_measured
        dropped_no_measurement += int((~keep).sum())

        category = episode.relative_to(root).parts[0]
        run = _scenario_of(episode)
        frames = sorted((images / episode.relative_to(root)).glob("img_*.png"))
        if len(frames) < len(rf):
            raise FileNotFoundError(
                f"{episode}: {len(rf)} samples but {len(frames)} PNG frames under {images}"
            )

        # FLASH already ships the occupancy grid at the (20,20,20) shape BeamDataset wants, but
        # packed one array per episode; split it into the per-sample .npy files it loads.
        voxels = None
        if lidar_out is not None:
            voxels = np.load(episode / "npz" / "lidar.npz")["lidar"]
            if voxels.shape[0] != len(rf) or voxels.shape[1:] != (20, 20, 20):
                raise ValueError(f"{episode}: unexpected lidar shape {voxels.shape}")
            (lidar_out / episode.relative_to(root)).mkdir(parents=True, exist_ok=True)

        for j in np.nonzero(keep)[0]:
            lidar_path = ""
            if voxels is not None:
                lidar_path = str(lidar_out / episode.relative_to(root) / f"lidar_{j:04d}.npy")
                np.save(lidar_path, voxels[j].astype(np.int8))
            x, y = float(gps[j, 0]), float(gps[j, 1])
            rows.append(
                {
                    "sample_id": ep_index * 1000 + int(j),
                    "episode_id": ep_index,
                    "scenario_id": scenario_id[run],
                    "t": float(j) * SAMPLE_PERIOD_S,
                    "lat": LAT0 + y * DEG_PER_M_LAT,
                    "lon": LON0 + x * DEG_PER_M_LON,
                    "los_flag": category == "Cat1",
                    "image_path": str(frames[j]),
                    "lidar_path": lidar_path,
                    "_gamma_lin": gamma_lin[j],
                    "_measured": measured[j],
                }
            )

    frame = pd.DataFrame(rows)
    LOG.info(
        "kept %d samples, dropped %d with <%d measured sectors",
        len(frame),
        dropped_no_measurement,
        min_measured,
    )

    lin = np.stack(frame.pop("_gamma_lin").to_numpy())
    seen = np.stack(frame.pop("_measured").to_numpy())
    if unmeasured == "noise_floor":
        # Treat an unswept sector as sitting at the receiver noise floor rather than at zero,
        # which lin_to_db would map to -120 dB and drag the loss matrix with it.
        floor = np.percentile(lin[seen], 1.0)
        lin = np.where(seen, lin, floor)
        LOG.info("unmeasured sectors set to the 1st percentile of measured power (%.1f)", floor)
    elif unmeasured == "min_measured":
        per_sample = np.where(seen, lin, np.inf).min(axis=1, keepdims=True)
        lin = np.where(seen, lin, per_sample)
        LOG.info("unmeasured sectors set to each sample's weakest measured sector")
    elif unmeasured == "zero":
        LOG.warning("unmeasured sectors left at 0; lin_to_db will map them to -120 dB")
    else:
        raise ValueError(unmeasured)

    gamma_db = 10.0 * np.log10(np.maximum(lin, 1e-12))
    for k in range(gamma_db.shape[1]):
        frame[f"gamma_{k}"] = gamma_db[:, k].astype(np.float32)
    frame["measured_frac"] = seen.mean(axis=1).astype(np.float32)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="FLASH_Dataset_3_Processed")
    parser.add_argument("--images", required=True, type=Path, help="PNG tree mirroring --root")
    parser.add_argument(
        "--lidar-out",
        type=Path,
        help="write the per-sample (20,20,20) occupancy grids here and fill lidar_path",
    )
    parser.add_argument("--out", required=True, type=Path, help="destination samples.parquet")
    parser.add_argument(
        "--unmeasured",
        default="noise_floor",
        choices=["noise_floor", "min_measured", "zero"],
        help="value given to sectors the radio never swept for a sample",
    )
    parser.add_argument("--min-measured", type=int, default=1)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    frame = build(args.root, args.images, args.lidar_out, args.unmeasured, args.min_measured)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.out, index=False)
    LOG.info(
        "wrote %s (%d samples, %d episodes, K=%d)",
        args.out,
        len(frame),
        frame.episode_id.nunique(),
        len(DEFINED_SECTORS),
    )
    LOG.info(
        "measured sectors per sample: mean %.1f of %d",
        frame.measured_frac.mean() * len(DEFINED_SECTORS),
        len(DEFINED_SECTORS),
    )


if __name__ == "__main__":
    main()
