"""Training from scratch and batched inference."""

from __future__ import annotations

import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from alwl.data import BeamDataset, InputStore
from alwl.models import build_model


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _loader(dataset, model_cfg: dict, shuffle: bool, generator=None) -> DataLoader:
    workers = int(model_cfg.get("num_workers", min(8, os.cpu_count() or 1)))
    return DataLoader(
        dataset,
        batch_size=int(model_cfg.get("batch", 64)),
        shuffle=shuffle,
        num_workers=workers,
        persistent_workers=workers > 0,
        generator=generator,
    )


def train_model(cfg: dict, store: InputStore, ids, labels, seed: int, soft=None):
    """Train a fresh network on (ids, labels); never warm-started across rounds.

    With ``soft`` (N, K) targets the loss is (1 - w) * CE(hard label) + w * CE(soft target),
    w = ``model.soft_labels.weight``.

    Training runs for at most ``model.epochs`` epochs. With ``model.early_stop`` it ends
    once the mean training loss has not improved by ``min_delta`` for ``patience`` epochs
    (and at least ``min_epochs`` have run): the network is "fully trained".
    Returns (model, epochs_run).
    """
    model_cfg = cfg["model"]
    torch.manual_seed(seed)
    dev = device()
    model = build_model(model_cfg["modality"], int(cfg["dataset"]["K"]),
                        bool(model_cfg.get("pretrained", False))).to(dev)
    generator = torch.Generator().manual_seed(seed)
    dataset = BeamDataset(store, ids, labels, augment=True, soft=soft)
    loader = _loader(dataset, model_cfg, True, generator)
    soft_cfg = model_cfg.get("soft_labels") or {}
    soft_weight = float(soft_cfg.get("weight", 0.0)) if soft is not None else 0.0
    optimizer = torch.optim.Adam(model.parameters(), lr=float(model_cfg.get("lr", 1e-3)))
    loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")
    early = model_cfg.get("early_stop") or {}
    patience = int(early.get("patience", 0))
    min_delta = float(early.get("min_delta", 1e-3))
    min_epochs = int(early.get("min_epochs", 1))
    best, stale, epoch = float("inf"), 0, 0
    model.train()
    for epoch in range(1, int(model_cfg.get("epochs", 60)) + 1):
        total = torch.zeros((), device=dev)
        for x, y, q in loader:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            if soft_weight:
                soft_loss = -(q.to(dev) * torch.log_softmax(logits, dim=1)).sum()
                loss = (1.0 - soft_weight) * loss + soft_weight * soft_loss
            (loss / len(y)).backward()
            optimizer.step()
            total += loss.detach()
        if patience:
            epoch_loss = float(total) / len(loader.dataset)
            if epoch_loss < best - min_delta:
                best, stale = epoch_loss, 0
            else:
                stale += 1
            if epoch >= min_epochs and stale >= patience:
                break
    return model, epoch


@torch.no_grad()
def predict(model, cfg: dict, store: InputStore, ids) -> tuple[np.ndarray, np.ndarray]:
    """Softmax probabilities (N, K) and penultimate embeddings (N, D) for ``ids``."""
    dev = next(model.parameters()).device
    loader = _loader(BeamDataset(store, ids), {**cfg["model"], "batch": 512}, False)
    model.eval()
    probs, embeddings = [], []
    for x, _, _ in loader:
        h = model.embed(x.to(dev, non_blocking=True))
        probs.append(torch.softmax(model.fc(h), dim=1).cpu())
        embeddings.append(h.cpu())
    return torch.cat(probs).numpy(), torch.cat(embeddings).numpy()


def head_parameters(model) -> tuple[np.ndarray, np.ndarray]:
    """Weights (K, D) and bias (K,) of the final linear layer."""
    return (
        model.fc.weight.detach().cpu().numpy().astype(np.float32),
        model.fc.bias.detach().cpu().numpy().astype(np.float32),
    )
