"""Sample tables, train/test splits and in-memory model inputs."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

LOG = logging.getLogger(__name__)

IMAGE_SIZE = (90, 160)  # default (height, width), as in Soltani et al. (2024)
LIDAR_SHAPE = (20, 20, 20)


# --------------------------------------------------------------------------- tables


def load_samples(cfg: dict) -> pd.DataFrame:
    """Read ``<processed>/samples.parquet`` and attach the label ``y`` (best beam index).

    The label is the beam with the highest received power, i.e. the outcome of an
    exhaustive beam search over the K beams in ``gamma``.
    """
    root = Path(cfg["dataset"]["processed"])
    frame = pd.read_parquet(root / "samples.parquet")
    gamma = np.asarray(frame["gamma"].tolist(), dtype=np.float32)
    K = int(cfg["dataset"]["K"])
    if gamma.shape != (len(frame), K):
        raise ValueError(f"gamma must have shape ({len(frame)}, {K}), got {gamma.shape}")
    frame["y"] = gamma.argmax(axis=1).astype(np.int64)
    return frame


def soft_targets(gamma: np.ndarray, temperature: float) -> np.ndarray:
    """Power-aware soft labels q_k ~ exp(gamma_k / T) from the full beam sweep (gamma in dB).

    The exhaustive search that yields the best-beam label measures every beam anyway, so
    the whole power profile comes at no extra labeling cost.
    """
    z = np.asarray(gamma, dtype=np.float64) / float(temperature)
    z -= z.max(axis=1, keepdims=True)
    q = np.exp(z)
    return (q / q.sum(axis=1, keepdims=True)).astype(np.float32)


# --------------------------------------------------------------------------- splits


def make_split(
    frame: pd.DataFrame,
    test_per_class: int,
    split_by: str,
    seed: int,
    max_test_fraction: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (pool_ids, test_ids).

    ``sample``: shuffle and hold out ``test_per_class`` samples of every class, giving a
    class-balanced test set (Soltani et al., 2024). ``max_test_fraction`` caps a class's
    test share so a rare class keeps members in the pool.
    ``episode``: hold out whole episodes until every class has ``test_per_class`` samples,
    so no trajectory appears in both sets.
    """
    rng = np.random.default_rng(seed)
    ids = frame["sample_id"].to_numpy(dtype=np.int64)
    y = frame["y"].to_numpy()
    if split_by == "sample":
        picked: list[int] = []
        for cls in np.unique(y):
            members = np.flatnonzero(y == cls)
            rng.shuffle(members)
            take = test_per_class
            if max_test_fraction is not None:
                take = min(take, int(len(members) * float(max_test_fraction)))
            picked.extend(members[:take])
        test_pos = np.asarray(picked, dtype=np.int64)
    elif split_by == "episode":
        episodes = frame["episode_id"].unique().copy()
        rng.shuffle(episodes)
        counts = {int(c): 0 for c in np.unique(y)}
        chosen = []
        episode_of = frame["episode_id"].to_numpy()
        for episode in episodes:
            if all(v >= test_per_class for v in counts.values()):
                break
            present = y[episode_of == episode]
            if any(counts[int(c)] < test_per_class for c in np.unique(present)):
                chosen.append(episode)
                for c in present:
                    counts[int(c)] += 1
        test_pos = np.flatnonzero(np.isin(episode_of, chosen))
    else:
        raise ValueError("split_by must be 'sample' or 'episode'")
    is_test = np.zeros(len(frame), dtype=bool)
    is_test[test_pos] = True
    return ids[~is_test], ids[is_test]


def split_path(cfg: dict, seed: int) -> Path:
    data = cfg["dataset"]
    name = data.get("split_name", data.get("split_by", "sample"))
    return Path(data["processed"]) / "splits" / f"{name}_seed{seed}.npz"


def load_split(cfg: dict, seed: int) -> tuple[np.ndarray, np.ndarray]:
    path = split_path(cfg, seed)
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; run `python -m scripts.split_pool_test` first")
    split = np.load(path)
    return split["pool_ids"].astype(np.int64), split["test_ids"].astype(np.int64)


# --------------------------------------------------------------------------- inputs


def _read_image(path: str, size: tuple[int, int] = IMAGE_SIZE) -> np.ndarray:
    with Image.open(path) as image:
        resized = image.convert("RGB").resize((size[1], size[0]), Image.BILINEAR)
    return np.asarray(resized, dtype=np.uint8)


def _read_lidar(path: str) -> np.ndarray:
    value = np.load(path)
    if value.shape == (1, *LIDAR_SHAPE):
        value = value[0]
    if value.shape != LIDAR_SHAPE:
        raise ValueError(f"LiDAR grid must have shape {LIDAR_SHAPE}, got {value.shape} ({path})")
    return value.astype(np.float32)


def image_size(cfg: dict) -> tuple[int, int]:
    """``dataset.image_size`` as (height, width); each dataset sets its own resolution."""
    return tuple(cfg["dataset"].get("image_size", IMAGE_SIZE))


class InputStore:
    """Model inputs decoded once and kept in memory, indexed by sample id.

    Decoding every PNG on every epoch made training CPU-bound; the decoded set is small
    (about 1.3 GB for 31k 90x160 images) and is shared with DataLoader workers by fork.
    """

    def __init__(self, ids: np.ndarray, inputs: np.ndarray, modality: str):
        self.ids = np.asarray(ids, dtype=np.int64)
        self.inputs = inputs
        self.modality = modality
        self._pos = {int(i): p for p, i in enumerate(self.ids)}

    @classmethod
    def load(cls, frame: pd.DataFrame, ids: np.ndarray, modality: str, workers: int = 16,
             image_size: tuple[int, int] = IMAGE_SIZE):
        rows = frame.set_index("sample_id").loc[np.asarray(ids, dtype=np.int64)]
        if modality == "image":
            size = (int(image_size[0]), int(image_size[1]))
            paths, reader = rows["image_path"].tolist(), lambda p: _read_image(p, size)
        elif modality == "lidar":
            paths, reader = rows["lidar_path"].tolist(), _read_lidar
        else:
            raise ValueError(f"Unknown modality {modality!r}; use 'image' or 'lidar'")
        LOG.info("decoding %d %s inputs", len(paths), modality)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            inputs = np.stack(list(pool.map(reader, paths, chunksize=64)))
        return cls(ids, inputs, modality)

    def positions(self, ids) -> np.ndarray:
        return np.asarray([self._pos[int(i)] for i in ids], dtype=np.int64)

    def tensor(self, pos: int) -> torch.Tensor:
        value = torch.from_numpy(self.inputs[pos])
        if self.modality == "image":
            return value.permute(2, 0, 1).float().div_(255.0)
        return value.unsqueeze(0)


class BeamDataset(Dataset):
    def __init__(self, store: InputStore, ids, labels=None, augment: bool = False, soft=None):
        self.store = store
        self.pos = store.positions(ids)
        self.labels = None if labels is None else torch.as_tensor(labels, dtype=torch.long)
        self.soft = None if soft is None else torch.as_tensor(soft, dtype=torch.float32)
        self.jitter = (
            transforms.ColorJitter(0.1, 0.1, 0.1, 0.05)
            if augment and store.modality == "image"
            else None
        )

    def __len__(self) -> int:
        return len(self.pos)

    def __getitem__(self, i):
        x = self.store.tensor(int(self.pos[i]))
        if self.jitter is not None:
            x = self.jitter(x)
        y = self.labels[i] if self.labels is not None else torch.tensor(-1)
        q = self.soft[i] if self.soft is not None else torch.zeros(0)
        return x, y, q
