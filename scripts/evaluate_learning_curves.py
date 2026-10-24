"""Summarise and plot a benchmark the way Soltani et al. (2024) report it.

* test accuracy vs. labeled samples, averaged over runs and smoothed with a moving
  average of ``--window`` iterations (Fig. 5), with the full-supervised accuracy as a line;
* per-class minimum queries vs. labeled samples (Fig. 6);
* labeled samples each strategy needs to reach every ``--targets`` accuracy and the
  saving relative to random sampling; mean standard error over iterations.

Several run directories can be compared in one figure, each tagged with a label, e.g. the same
strategies trained with hard and with power-aware soft labels:
  python -m scripts.evaluate_learning_curves --runs runs/flash/lidar:hard runs/flash_soft/lidar:soft \
      --out results/flash_lidar --targets 0.6 0.7
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from alwl.metrics import labels_to_reach

# Fixed categorical order: a strategy keeps its colour/marker in every figure. The proposed
# method (HIGHLIGHT) is drawn in red with square markers on top of the others.
ORDER = ["random", "least_conf", "margin", "entropy", "bald", "coreset", "badge", "bait",
         "galaxy", "typiclust", "probcover", "epl", "paal"]
COLORS = ["#8c564b", "#ff7f0e", "#bcbd22", "#2ca02c", "#e377c2", "#17becf", "#9467bd",
          "#7f7f7f", "#1f77b4", "#393b79", "#637939", "#aec7e8", "#d62728"]
MARKERS = ["o", "o", "v", "D", "P", "h", "X", "*", "^", "<", ">", "p", "s"]
NAMES = {"random": "Random", "least_conf": "Least conf.", "margin": "Margin",
         "entropy": "Entropy", "bald": "BALD", "coreset": "Core-set", "badge": "BADGE",
         "bait": "BAIT", "galaxy": "GALAXY", "typiclust": "TypiClust",
         "probcover": "ProbCover", "epl": "EPL", "paal": "PAAL"}
HIGHLIGHT = "paal"
# Legend text per run key from --rename (e.g. ablation names), overriding NAMES.
RENAME: dict[str, str] = {}
# When one strategy appears in several run groups in a figure, the first group is drawn
# dashed so that the later (e.g. soft-label) group stands out as solid.
STYLES = ["--", "-", ":", "-."]
STYLE = {
    "font.family": "serif",
    "font.size": 10,
    "axes.linewidth": 0.9,
    "axes.labelsize": 12,
    "axes.labelweight": "bold",
    "xtick.direction": "in",
    "ytick.direction": "in",
    "legend.fontsize": 9,
    "savefig.bbox": "tight",
}


def _base(key: str) -> str:
    return key.split(" (", 1)[0]


def load_runs(run_root: Path, label: str = "") -> dict[str, list[pd.DataFrame]]:
    """Logs under ``run_root`` grouped by strategy; keys are ``"<strategy> (<label>)"``."""
    runs: dict[str, list[pd.DataFrame]] = {}
    for path in sorted(run_root.glob("*_seed*/log.jsonl")):
        frame = pd.read_json(path, lines=True)
        if len(frame):
            name = str(frame["strategy"].iloc[0])
            runs.setdefault(f"{name} ({label})" if label else name, []).append(frame)
    order = lambda k: ORDER.index(_base(k)) if _base(k) in ORDER else 99  # noqa: E731
    return {k: runs[k] for k in sorted(runs, key=order)}


def curve(frames: list[pd.DataFrame], column: str, window: int) -> pd.DataFrame:
    n = min(len(f) for f in frames)
    values = np.array([f[column].to_numpy(dtype=float)[:n] for f in frames])
    mean = pd.Series(values.mean(axis=0))
    return pd.DataFrame(
        {
            "n_labels": frames[0]["n_labels"].to_numpy()[:n],
            "mean": mean,
            "smooth": mean.rolling(window, center=True, min_periods=1).mean(),
            "std": values.std(axis=0, ddof=1) if len(frames) > 1 else np.zeros(n),
            "stderr": values.std(axis=0, ddof=1) / np.sqrt(len(frames))
            if len(frames) > 1
            else np.zeros(n),
        }
    )


def _label(key: str, keys) -> str:
    """Display name; the run-group tag is kept only when it tells two lines apart."""
    if key in RENAME:
        return RENAME[key]
    base = _base(key)
    name = NAMES.get(base, base)
    clash = sum(_base(k) == base for k in keys) > 1
    return f"{name} ({key.split(' (', 1)[1]}" if clash and " (" in key else name


def plot(curves: dict[str, pd.DataFrame], column: str, ylabel: str, path: Path,
         reference: float | None = None, groups: dict[str, int] | None = None,
         x: str = "rounds") -> None:
    """Mean curve per strategy with a +-1 std band over seeds; the proposed method on top."""
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(6.4, 3.6))
        for strategy, c in curves.items():
            base = _base(strategy)
            i = ORDER.index(base) if base in ORDER else len(ORDER) - 1
            # line style only tells run groups apart when one strategy appears in several
            clash = sum(_base(k) == base for k in curves) > 1
            style = STYLES[(groups or {}).get(strategy, 1) % len(STYLES)] if clash else "-"
            lead = base == HIGHLIGHT
            xs = np.arange(1, len(c) + 1) if x == "rounds" else c["n_labels"]
            ax.fill_between(xs, c[column] - c["std"], c[column] + c["std"], color=COLORS[i],
                            alpha=0.12, linewidth=0, zorder=2)
            ax.plot(xs, c[column], color=COLORS[i], marker=MARKERS[i], linestyle=style,
                    markersize=5 if lead else 4.5, markerfacecolor="white",
                    markeredgewidth=1.4, linewidth=1.8 if lead else 1.4,
                    zorder=5 if lead else 3, label=_label(strategy, curves))
        if reference is not None:
            ax.axhline(reference, color="#555555", linestyle=(0, (4, 3)), linewidth=0.9,
                       label="Full supervision", zorder=1)
        if x == "rounds":
            ax.set_xticks(np.arange(1, max(len(c) for c in curves.values()) + 1))
            ax.set_xlabel("Rounds")
        else:
            ax.set_xlabel("Labeled samples")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.legend(ncol=2, loc="lower right", frameon=True, framealpha=0.95,
                  edgecolor="#bbbbbb", handlelength=2.2, columnspacing=1.2)
        fig.savefig(path.with_suffix(".pdf"))
        fig.savefig(path.with_suffix(".png"), dpi=300)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", required=True, nargs="+",
                        help="<run_dir>/<modality>[:<label>], one or more")
    parser.add_argument("--out", required=True)
    parser.add_argument("--strategies", nargs="*", default=None,
                        help="keep only these (strategy name or full key, e.g. 'paal (soft)')")
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--x", choices=["rounds", "labels"], default="rounds",
                        help="x axis: iteration number (from 1) or labeled samples")
    parser.add_argument("--targets", nargs="*", type=float, default=[])
    parser.add_argument("--rename", nargs="*", default=[], metavar="KEY=TEXT",
                        help="legend text for a run key, e.g. 'epl (soft)=w/o k-center'")
    args = parser.parse_args()
    RENAME.update(item.split("=", 1) for item in args.rename)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runs: dict[str, list[pd.DataFrame]] = {}
    groups: dict[str, int] = {}
    full: list[float] = []
    tagged = len(args.runs) > 1
    for g, spec in enumerate(args.runs):
        root, _, label = spec.partition(":")
        root = Path(root)
        found = load_runs(root, label or (root.parent.name if tagged else ""))
        if not found:
            raise FileNotFoundError(f"No */log.jsonl under {root}")
        if args.strategies is not None:
            keep = set(args.strategies)
            found = {k: v for k, v in found.items() if k in keep or _base(k) in keep}
        runs.update(found)
        groups.update({k: g for k in found})
        full += [json.loads(p.read_text())["top1"] for p in root.glob("full_supervised_seed*.json")]
    reference = float(np.mean(full)) if full else None
    random_key = next((k for k in runs if _base(k) == "random"), None)

    accuracy = {s: curve(f, "top1", args.window) for s, f in runs.items()}
    plot(accuracy, "smooth", "Top-1 accuracy", out / f"accuracy_vs_{args.x}", reference,
         groups if tagged else None, args.x)
    plot({s: curve(f, "per_class_min_q", 1) for s, f in runs.items()}, "mean",
         "Per-class min. queries", out / f"per_class_min_queries_vs_{args.x}",
         groups=groups if tagged else None, x=args.x)

    rows = []
    for strategy, c in accuracy.items():
        row = {
            "strategy": strategy,
            "runs": len(runs[strategy]),
            "final_labels": int(c["n_labels"].iloc[-1]),
            "final_top1": round(float(c["mean"].iloc[-1]), 4),
            "auc_top1": round(float(c["mean"].mean()), 4),
            "mean_stderr": round(float(c["stderr"].mean()), 4),
        }
        for target in args.targets:
            n = labels_to_reach(c["n_labels"], c["smooth"], target)
            base = (labels_to_reach(accuracy[random_key]["n_labels"],
                                    accuracy[random_key]["smooth"], target)
                    if random_key else float("nan"))
            row[f"labels@{target:g}"] = n
            row[f"saving@{target:g}"] = (round(1 - n / base, 3)
                                         if np.isfinite(n) and np.isfinite(base) else float("nan"))
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "summary.csv", index=False)
    if reference is not None:
        print(f"full-supervision top1: {reference:.4f}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
