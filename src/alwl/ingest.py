"""Convert raw DeepSense scenario tables (or a generic table) into samples.parquet."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED = ["sample_id", "episode_id", "scenario_id", "t", "lat", "lon", "los_flag"]
LOG = logging.getLogger(__name__)


def lin_to_db(x):
    return 10.0 * np.log10(np.maximum(np.asarray(x), 1e-12))


def _absolute(base: Path, relative: str) -> Path:
    return (base / str(relative).removeprefix("./")).resolve()


def _timestamp_seconds(value: str) -> float:
    clock, fraction = str(value).split("-", 1)
    hour, minute, second = map(int, clock.split(":"))
    return hour * 3600 + minute * 60 + second + float(f"0.{fraction}")


def _scenario_number(path: Path) -> int:
    match = re.search(r"scenario[_ -]?(\d+)", path.name, flags=re.IGNORECASE)
    if not match:
        match = re.search(r"scenario[_ -]?(\d+)", str(path.parent), flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Cannot determine scenario number from {path}")
    return int(match.group(1))


def _read_location(path: Path) -> tuple[float, float]:
    value = np.loadtxt(path, dtype=np.float64).reshape(-1)
    if len(value) < 2 or not np.isfinite(value[:2]).all():
        raise ValueError(f"Invalid GPS location in {path}")
    return float(value[0]), float(value[1])


def _lidar_path(base: Path, relative) -> str:
    """The voxel grid written by scripts.voxelize_deepsense_lidar (unit1/lidar_voxel/<stem>.npy)
    when it exists, else the raw point cloud the scenario table points at."""
    if not relative or pd.isna(relative):
        return ""
    raw = _absolute(base, relative)
    voxel = raw.parent.parent / "lidar_voxel" / (raw.stem + ".npy")
    return str(voxel if voxel.exists() else raw)


def _ingest_deepsense(tables: list[Path], K: int) -> pd.DataFrame:
    rows = []
    warned_los = False
    repaired_samples = 0
    repaired_values = 0
    for table in tables:
        source = pd.read_csv(table)
        scenario = _scenario_number(table)
        required = {"index", "unit1_rgb", "unit1_pwr_60ghz", "unit2_loc", "seq_index"}
        if not required.issubset(source.columns):
            raise ValueError(f"{table} is not a supported DeepSense scenario table")
        for record in source.to_dict("records"):
            base = table.parent
            power_path = _absolute(base, record["unit1_pwr_60ghz"])
            power = np.loadtxt(power_path, dtype=np.float64).reshape(-1)
            if power.shape != (K,):
                raise ValueError(f"Expected {K} powers in {power_path}, got {power.shape}")
            invalid = ~np.isfinite(power)
            if invalid.any():
                finite = power[~invalid]
                if not len(finite):
                    raise ValueError(f"Power vector is entirely non-finite in {power_path}")
                # DeepSense's official baseline ignores missing entries when choosing the
                # optimum; assigning them the observed floor keeps that behaviour.
                power[invalid] = finite.min()
                repaired_samples += 1
                repaired_values += int(invalid.sum())
            # DeepSense stores normalized positive linear powers; store them in dB.
            power_db = lin_to_db(power) if np.all(power >= 0) else power
            lat, lon = _read_location(_absolute(base, record["unit2_loc"]))
            los = record.get("los_flag")
            if los is None or pd.isna(los):
                los = True
                if not warned_los:
                    LOG.warning("DeepSense table has no LOS labels; los_flag defaults to true")
                    warned_los = True
            index = int(record["index"])
            rows.append(
                {
                    "sample_id": scenario * 1_000_000 + index,
                    "episode_id": scenario * 100_000 + int(record["seq_index"]),
                    "scenario_id": scenario,
                    "t": _timestamp_seconds(record.get("time_stamp", "00:00:00-0")),
                    "lat": lat,
                    "lon": lon,
                    "los_flag": bool(los),
                    "gamma": np.asarray(power_db, dtype=np.float32),
                    "image_path": str(_absolute(base, record["unit1_rgb"])),
                    "lidar_path": _lidar_path(base, record.get("unit1_lidar")),
                }
            )
    if repaired_samples:
        LOG.warning(
            "Repaired %d non-finite power values across %d DeepSense samples",
            repaired_values,
            repaired_samples,
        )
    return pd.DataFrame(rows)


def _find_table(root: Path) -> Path:
    if root.is_file():
        return root
    preferred = [
        root / "samples.parquet",
        root / "samples.csv",
        root / "metadata.parquet",
        root / "metadata.csv",
    ]
    for path in preferred:
        if path.exists():
            return path
    found = list(root.glob("*.parquet")) + list(root.glob("*.csv"))
    if len(found) != 1:
        raise FileNotFoundError(f"Expected one source parquet/CSV under {root}; found {len(found)}")
    return found[0]


def ingest(root: str, out_path: str, K: int) -> None:
    root_path = Path(root)
    scenario_tables = sorted(root_path.rglob("scenario*.csv")) if root_path.is_dir() else []
    if scenario_tables:
        probe = pd.read_csv(scenario_tables[0], nrows=1)
        if "unit1_pwr_60ghz" in probe.columns:
            df = _ingest_deepsense(scenario_tables, K)
        else:
            source = _find_table(root_path)
            df = pd.read_parquet(source) if source.suffix == ".parquet" else pd.read_csv(source)
    else:
        source = _find_table(root_path)
        df = pd.read_parquet(source) if source.suffix == ".parquet" else pd.read_csv(source)
    missing = set(REQUIRED) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if "gamma" in df:
        if isinstance(df["gamma"].iloc[0], str):
            import json

            df["gamma"] = df["gamma"].map(json.loads)
    else:
        cols = [f"gamma_{i}" for i in range(K)]
        if not set(cols).issubset(df.columns):
            raise ValueError(f"Need gamma list or all columns gamma_0..gamma_{K - 1}")
        df["gamma"] = list(df[cols].to_numpy(dtype=np.float32))
    g = np.asarray(df["gamma"].tolist(), dtype=np.float32)
    if g.shape != (len(df), K) or not np.isfinite(g).all():
        raise ValueError(f"gamma must be finite with shape ({len(df)}, {K}), got {g.shape}")
    df["gamma"] = list(g)
    for optional in ["image_path", "lidar_path"]:
        if optional not in df:
            df[optional] = ""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df[REQUIRED + ["gamma", "image_path", "lidar_path"]].to_parquet(out, index=False)
