"""Computational cost of the query step: wall-clock time to select one batch and extra
parameters of every strategy, plus the shared backbone's size and FLOPs.

Only ``strategy.select`` is timed, on the CPU, so the numbers isolate the acquisition rule from
the model inference all model-based strategies share (that cost is the backbone row). Each
strategy runs the real active-learning rounds (``al`` section of the config: initial and batch
fractions, number of rounds) on a random subset of at most ``--max-pool`` pool samples, with the
dataset's real labels and beam sweeps; the model outputs (softmax, embeddings)
are synthetic with the real shapes, since selection time depends on sizes, not on values.

  python -m scripts.computational_cost --jobs configs/flash.yaml:lidar configs/flash.yaml:image \\
      --processed <local copy> <local copy> --out results/cost
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode

from alwl.config import load_config
from alwl.data import image_size, load_samples, load_split
from alwl.models import build_model
from alwl.strategies import Context, get_strategy

NAMES = {"random": "Random", "entropy": "Entropy", "least_conf": "Least conf.",
         "galaxy": "GALAXY", "coreset": "Core-set", "badge": "BADGE", "bald": "BALD",
         "epl": "EPL", "paal": "PAAL (ours)"}
LABEL = {"flash": "FLASH", "deepsense": "DS32"}


def backbone_cost(modality: str, K: int, size: tuple[int, int]) -> tuple[float, float]:
    """(parameters in M, forward GFLOPs per sample) of the classifier."""
    model = build_model(modality, K).eval()
    x = torch.zeros((1, 3, *size) if modality == "image" else (1, 1, 20, 20, 20))
    with torch.no_grad(), FlopCounterMode(display=False) as counter:
        model(x)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    return params, counter.get_total_flops() / 1e9


def _context(y_pool, gamma_pool, labeled, K, D, rng, pool_ids):
    """Synthetic model outputs with the real shapes: ~50% top-1 softmax, class-clustered
    embeddings; labels and beam sweeps are the dataset's own."""
    y_noisy = np.where(rng.random(len(y_pool)) < 0.5, y_pool, rng.integers(K, size=len(y_pool)))
    logits = 3.0 * np.eye(K, dtype=np.float32)[y_noisy] + rng.normal(size=(len(y_pool), K))
    probs = np.exp(logits - logits.max(axis=1, keepdims=True)).astype(np.float32)
    probs /= probs.sum(axis=1, keepdims=True)
    centers = rng.normal(size=(K, D)).astype(np.float32)
    emb = centers[y_noisy] + rng.normal(size=(len(y_pool), D)).astype(np.float32)
    lab = np.zeros(len(y_pool), bool)
    lab[labeled] = True
    truth = dict(zip(pool_ids[~lab], y_pool[~lab]))
    return Context(
        pool_ids=pool_ids[~lab], probs=probs[~lab], emb=emb[~lab],
        labeled_ids=pool_ids[lab], labeled_probs=probs[lab], labeled_emb=emb[lab],
        labels=y_pool[lab], head_weight=rng.normal(size=(K, D)).astype(np.float32),
        head_bias=np.zeros(K, np.float32), num_classes=K, rng=np.random.default_rng(0),
        oracle=lambda i: int(truth[int(i)]), labeled_gamma=gamma_pool[lab])


def query_times(cfg: dict, strategies: list[str], seeds: list[int],
                max_pool: int | None = None, per_sample: bool = False) -> dict[str, list[float]]:
    """Time of each strategy's select(), one mean over the rounds per seed: ms per query round
    (selecting one batch), or with ``per_sample`` µs per unlabeled sample."""
    frame = load_samples(cfg)
    K = int(cfg["dataset"]["K"])
    D = build_model(cfg["model"]["modality"], K).fc.in_features
    y_of = dict(zip(frame["sample_id"], frame["y"]))
    gamma_of = dict(zip(frame["sample_id"], frame["gamma"]))
    al = cfg["al"]
    out: dict[str, list[float]] = {s: [] for s in strategies}
    for seed in seeds:
        pool_ids, _ = load_split(cfg, seed)
        if max_pool and len(pool_ids) > max_pool:
            pool_ids = np.random.default_rng(seed).choice(pool_ids, max_pool, replace=False)
        y_pool = np.asarray([y_of[int(i)] for i in pool_ids])
        gamma_pool = np.asarray([gamma_of[int(i)] for i in pool_ids], dtype=np.float32)
        n0 = max(1, int(round(al["initial_fraction"] * len(pool_ids))))
        b = max(1, int(round(al["batch_fraction"] * len(pool_ids))))
        rng = np.random.default_rng(seed)
        per_round: dict[str, list[float]] = {s: [] for s in strategies}
        for r in range(int(al["rounds"])):
            labeled = rng.choice(len(pool_ids), n0 + r * b, replace=False)
            ctx = _context(y_pool, gamma_pool, labeled, K, D, rng, pool_ids)
            for name in strategies:
                strategy = get_strategy(name)
                t0 = time.perf_counter()
                strategy.select(ctx, b)
                elapsed = time.perf_counter() - t0
                per_round[name].append(1e6 * elapsed / len(ctx.pool_ids) if per_sample
                                       else 1e3 * elapsed)
        for name in strategies:
            out[name].append(float(np.mean(per_round[name])))
    return out


def _mark(values: list[float], text: list[str]) -> tuple[list[str], list[str]]:
    """Bold the lowest, underline the second lowest (Markdown, LaTeX)."""
    order = np.argsort(values, kind="stable")
    md, tex = list(text), list(text)
    md[order[0]], tex[order[0]] = f"**{text[order[0]]}**", rf"\textbf{{{text[order[0]]}}}"
    if len(order) > 1:
        md[order[1]], tex[order[1]] = f"<u>{text[order[1]]}</u>", rf"\underline{{{text[order[1]]}}}"
    return md, tex


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--jobs", nargs="+", required=True, help="<config>:<modality>")
    parser.add_argument("--processed", nargs="*", default=[],
                        help="dataset.processed per job (local copies), in --jobs order")
    parser.add_argument("--strategies", nargs="+",
                        default=["random", "entropy", "least_conf", "galaxy", "coreset", "epl",
                                 "paal"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 10, 100])
    parser.add_argument("--max-pool", type=int, default=3000,
                        help="time on a random subset of at most this many pool samples")
    parser.add_argument("--per-sample", action="store_true",
                        help="report µs per unlabeled sample instead of ms per query round")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    columns, times, backbones = [], [], []
    for j, job in enumerate(args.jobs):
        config, modality = job.split(":")
        overrides = [f"model.modality={modality}"]
        if j < len(args.processed):
            overrides.append(f"dataset.processed={args.processed[j]}")
        cfg = load_config(config, overrides)
        name = f"{LABEL.get(cfg['dataset']['name'], cfg['dataset']['name'])} {modality}"
        print(f"timing {name} ...", flush=True)
        columns.append(name)
        times.append(query_times(cfg, args.strategies, args.seeds, args.max_pool, args.per_sample))
        params, gflops = backbone_cost(modality, int(cfg["dataset"]["K"]), image_size(cfg))
        backbones.append((name, params, gflops))

    cells = {s: [] for s in args.strategies}
    md_cols, tex_cols = [], []
    for col in times:
        means = [np.mean(col[s]) for s in args.strategies]
        text = [f"{np.mean(col[s]):.2f} ± {np.std(col[s]):.2f}" for s in args.strategies]
        md, tex = _mark(means, text)
        md_cols.append(md)
        tex_cols.append([t.replace("±", r"$\pm$") for t in tex])
    unit = "µs/sample" if args.per_sample else "ms/round"
    lines = ["| Method | Extra params (M) ↓ | " + " | ".join(f"{c} query ({unit}) ↓"
                                                         for c in columns) + " |",
             "|---|---|" + "---|" * len(columns)]
    tex = [r"\begin{tabular}{l c " + "c " * len(columns) + "}", r"\toprule",
           "Method & Extra params (M) $\\downarrow$ & "
           + " & ".join(f"{c} $\\downarrow$" for c in columns) + r" \\", r"\midrule"]
    for i, s in enumerate(args.strategies):
        lines.append(f"| {NAMES.get(s, s)} | **0** | " + " | ".join(c[i] for c in md_cols) + " |")
        tex.append(f"{NAMES.get(s, s)} & \\textbf{{0}} & "
                   + " & ".join(c[i] for c in tex_cols) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    lines += ["", "| Backbone (shared by all strategies) | Params (M) | GFLOPs / sample |",
              "|---|---|---|"]
    lines += [f"| {n} | {p:.2f} | {g:.3f} |" for n, p, g in backbones]
    (out / "cost_table.md").write_text("\n".join(lines) + "\n")
    (out / "cost_table.tex").write_text("\n".join(tex) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
