from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np


@dataclass
class Context:
    """Everything a query strategy may look at in one active-learning round.

    ``probs``/``emb`` are the current model's softmax outputs and penultimate features on
    the unlabeled pool; the ``labeled_*`` arrays are the same for the labeled set.
    ``oracle`` reveals a label and is only for strategies that query one sample at a time
    inside a batch (GALAXY); every sample it is called on must be returned as selected.
    ``labeled_gamma`` (|L|, K) is the received power (dB) of every beam for the labeled
    samples, i.e. the full sweep the exhaustive-search oracle measured for each of them.
    """

    pool_ids: np.ndarray
    probs: np.ndarray
    emb: np.ndarray
    labeled_ids: np.ndarray
    labeled_probs: np.ndarray
    labeled_emb: np.ndarray
    labels: np.ndarray
    head_weight: np.ndarray
    head_bias: np.ndarray
    num_classes: int
    rng: np.random.Generator
    params: dict[str, Any] = field(default_factory=dict)
    oracle: Callable[[int], int] | None = None
    labeled_gamma: np.ndarray | None = None


class Strategy(Protocol):
    name: str

    def select(self, ctx: Context, b: int) -> np.ndarray:
        """Return ``b`` distinct positions into ``ctx.pool_ids``."""
        ...
