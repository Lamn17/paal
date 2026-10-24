"""Query strategies, looked up by name with :func:`get_strategy`."""

from __future__ import annotations

from .badge import Badge
from .bait import Bait
from .bald import Bald
from .base import Context, Strategy
from .coreset import Coreset
from .galaxy import Galaxy
from .paal import Epl, Paal
from .probcover import ProbCover
from .typiclust import TypiClust
from .uncertainty import Entropy, LeastConfidence, Margin, Random

STRATEGIES: dict[str, type] = {
    cls.name: cls
    for cls in [
        Random,
        LeastConfidence,
        Margin,
        Entropy,
        Bald,
        Coreset,
        Badge,
        Bait,
        Galaxy,
        TypiClust,
        ProbCover,
        Epl,
        Paal,
    ]
}
ALIASES = {"confidence": "least_conf"}


def get_strategy(name: str) -> Strategy:
    key = ALIASES.get(name, name)
    if key not in STRATEGIES:
        raise ValueError(f"Unknown strategy {name!r}; choose from {sorted(STRATEGIES)}")
    return STRATEGIES[key]()


__all__ = ["Context", "Strategy", "STRATEGIES", "get_strategy"]
