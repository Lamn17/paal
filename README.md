# PAAL: Power-Aware Active Learning

An implementation of **Power-Aware Active Learning (PAAL)** for pool-based
active learning in sensing-aided mmWave beam selection, together with a
benchmark of standard active-learning baselines on **FLASH** and
**DeepSense 6G**.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

> 🎯 **TL;DR**: every exhaustive beam sweep measures the power of *all* beams,
> not just the best one. PAAL reuses that power profile, at no extra
> measurement cost, to train on soft labels, score unlabeled samples by their
> expected power loss in dB, and select diverse batches with a loss-weighted
> k-center rule.

## 📖 About

Learning-based beam selection predicts the best mmWave beam from camera images
or LiDAR scans, but every training label costs one exhaustive beam sweep.
Active learning reduces that cost by labeling only the most useful samples.
Standard AL methods keep only the index of the strongest beam and treat every
misclassification as equally costly; PAAL instead uses the full measured
power profile $\gamma(x) \in \mathbb{R}^K$.

### Key Features

- 🎚️ **Power-aware soft labels**: trains on targets derived from the measured
  beam powers, so near-optimal beams are not penalized as hard errors.
- 📉 **Expected Power Loss (EPL)**: scores each unlabeled sample by the dB loss
  expected under the beam the model would pick.
- 📐 **EPL-weighted k-center**: builds diverse batches that cover high-loss
  regions of the embedding space without redundant queries.
- 🧪 **Baseline benchmark**: random, least confidence, margin, entropy, BALD,
  Core-set, BADGE, BAIT, GALAXY, TypiClust and ProbCover share one loop.
- ⚙️ **YAML experiments**: presets for FLASH and DeepSense 6G (scenarios 32
  and 33), camera and LiDAR modalities, resumable benchmarks.

The evaluation protocol follows Soltani et al., *"Learning from the Best:
Active Learning for Wireless Communications"*, IEEE Wireless Communications
Magazine, 2024 ([arXiv:2402.04896](https://arxiv.org/abs/2402.04896)).
When comparing strategies, use matched seeds, splits, initial labeled sets,
budgets and training targets (hard vs. soft labels).

## 🏃‍♂️ Quick Start

### Prerequisites

- Python 3.11 or higher
- CUDA-compatible GPU (recommended)
- [uv](https://github.com/astral-sh/uv)
- The [FLASH](https://genesys-lab.org/multimodal-fusion-nextg-v2v-communications)
  and/or [DeepSense 6G](https://www.deepsense6g.net/) datasets

### Installation

1. **Clone the repository**:

```bash
git clone https://github.com/Lamn17/paal.git
cd paal
```

2. **Install dependencies with uv**:

```bash
# Install uv first if needed: https://docs.astral.sh/uv/
uv sync --extra dev
export PYTHONPATH=src
```

> `pyproject.toml` pins PyTorch to the CPU wheel index. For GPU training,
> edit `[tool.uv.sources]` / `[[tool.uv.index]]` to point at the CUDA index
> matching your driver.

### Prepare a Dataset

Set `dataset.root` and `dataset.processed` in the chosen config to your local
paths (the preset paths are local templates, not downloaded data), then build
the sample table and the pool/test split:

```bash
# FLASH: flatten the processed release into samples.parquet (+ LiDAR grids)
uv run python -m scripts.preprocess_flash \
  --root <FLASH_Dataset_3_Processed> --images <png tree> \
  --lidar-out <lidar dir> --out <processed>/samples.parquet

# DeepSense 6G: voxelize raw point clouds into 20x20x20 occupancy grids
uv run python -m scripts.voxelize_deepsense_lidar \
  --dir <root>/unit1/lidar_data --out <root>/unit1/lidar_voxel

# Both: sample table and class-balanced pool/test split
uv run python -m scripts.build_sample_table --config configs/flash.yaml
uv run python -m scripts.split_pool_test   --config configs/flash.yaml
```

### Run Your First PAAL Experiment

```bash
uv run python -m scripts.run_active_learning \
  --config configs/flash.yaml --strategy paal --seed 1 --modality lidar \
  --override "model.soft_labels={temperature: 1.0, weight: 0.5}" \
  --override run_dir=runs/flash_soft
```

## 🧠 PAAL Method

For sensor input $x$ the predictor returns beam probabilities
$p(k \mid x)$ and selects $\hat a(x) = \arg\max_k p(k \mid x)$. The oracle
(an exhaustive sweep) returns the best beam $y^{*}(x)$ and the power
$\gamma_k(x)$ in dB of every beam $k$.

### 1. Power-Aware Soft Labels

Each power profile becomes a soft target with temperature $T$ (dB), and the
model is trained on a weighted sum of hard- and soft-label cross-entropy:

$$
q_k(x)=\frac{\exp(\gamma_k(x)/T)}{\sum_{j=1}^{K}\exp(\gamma_j(x)/T)},\qquad
\mathcal{L}_{\mathrm{train}}=(1-w)\,\mathrm{CE}(y^{*},p)+w\,\mathrm{CE}(q,p).
$$

### 2. Expected Power Loss Scoring

From the labeled set $\mathcal{L}$, PAAL estimates the mean power gap between
the true best beam $k$ and a selected beam $a$:

$$
D(k,a)=\frac{1}{|\mathcal{L}_k|}\sum_{x\in\mathcal{L}_k}\big(\gamma_k(x)-\gamma_a(x)\big),\qquad
\mathcal{L}_k=\{x\in\mathcal{L}:y^{*}(x)=k\}.
$$

Rows of beams with no labeled sample take the mean of the observed rows. Each
unlabeled sample is scored by its expected power loss:

$$
\mathrm{EPL}(x)=\sum_{k=1}^{K}p(k\mid x)\,D\big(k,\hat a(x)\big).
$$

Unlike entropy, EPL stays low when the model hesitates between beams of
similar power, because picking either one costs little.

### 3. EPL-Weighted k-Center Selection

With $h(x)$ the penultimate-layer embedding, the batch $\mathcal{S}$ is built
greedily:

$$
d(x)=\min_{x'\in\mathcal{L}\cup\mathcal{S}}\|h(x)-h(x')\|^{2},\qquad
x^{\star}=\arg\max_{x\in\mathcal{U}\setminus\mathcal{S}}\ \mathrm{EPL}(x)\cdot d(x).
$$

Distances are updated after each pick until $|\mathcal{S}|=b$. The cost is
$O(b\,|\mathcal{U}|\,m)$ for $m$-dimensional embeddings, the same order as
Core-set.

### Ablations

| Variant | How to run |
| --- | --- |
| PAAL without soft labels | `--strategy paal` without the `model.soft_labels` override |
| PAAL without EPL weight (= Core-set) | `--strategy coreset` with soft labels |
| PAAL without k-center (EPL ranking) | `--strategy epl` with soft labels |

## ⚙️ Configuration

Every dataset preset lives in `configs/`. Example:

```yaml
# configs/flash.yaml
dataset:
  name: flash
  root: /path/to/datasets/flash
  processed: /path/to/datasets/flash/processed
  K: 34
  split_by: sample
  test_per_class: 10
  max_test_fraction: 0.5

al:
  initial_fraction: 0.05
  batch_fraction: 0.05
  rounds: 9
  seeds: [1, 10, 100, 1000, 10000]

model:
  modality: lidar
  epochs: 60
  early_stop: {patience: 5, min_delta: 0.001, min_epochs: 10}
  lr: 1.0e-3
  batch: 64
  # soft_labels: {temperature: 1.0, weight: 0.5}   # enable power-aware soft labels

run_dir: runs/flash
```

Any key can be overridden from the command line with
`--override a.b=value` (values are parsed as YAML).

### Key Parameters

- `dataset.K`: codebook size (34 sectors for FLASH, 64 beams for DeepSense).
- `dataset.test_per_class`, `max_test_fraction`: class-balanced test set of
  10 samples per beam, or half of a beam's samples if it has fewer than 20.
- `dataset.image_size`: camera resolution (DeepSense 32 uses `[270, 480]`).
- `al.initial_fraction`, `al.batch_fraction`, `al.rounds`: initial labeled
  fraction, fraction queried per round and number of rounds
  (default 5% → 50% of the pool).
- `model.modality`: `lidar` (residual 3-D CNN on 20³ grids) or `image`
  (ResNet18, random init).
- `model.soft_labels.temperature`, `model.soft_labels.weight`: $T$ (dB) and
  $w$ of the soft-label loss; omit the block for hard labels.
- `benchmark.strategies`: strategies run by `scripts.run_benchmark`.
- `strategies.<name>`: per-strategy hyper-parameters (BALD, BAIT, TypiClust,
  ProbCover).

Included presets:

| Dataset | Configuration |
| --- | --- |
| FLASH | `configs/flash.yaml` |
| DeepSense 6G, scenario 33 | `configs/deepsense.yaml` |
| DeepSense 6G, scenario 32 | `configs/deepsense32.yaml` |

## 🔧 Usage

### Query Strategies (`src/alwl/strategies/`)

| Name | Family | Reference |
| --- | --- | --- |
| `random` | passive baseline | — |
| `least_conf` (alias `confidence`) | uncertainty | Settles, 2009 |
| `margin` | uncertainty | Scheffer et al., 2001 |
| `entropy` | uncertainty | Shannon-entropy sampling |
| `bald` | Bayesian uncertainty (MC dropout) | Gal et al., ICML 2017 |
| `coreset` | diversity (k-center) | Sener & Savarese, ICLR 2018 |
| `badge` | hybrid (gradient embeddings) | Ash et al., ICLR 2020 |
| `bait` | Fisher information | Ash et al., NeurIPS 2021 |
| `galaxy` | class imbalance (graph bisection) | Zhang et al., ICML 2022 |
| `typiclust` | low-budget typicality | Hacohen et al., ICML 2022 |
| `probcover` | low-budget coverage | Yehuda et al., NeurIPS 2022 |
| `epl` | PAAL ablation (EPL ranking) | this work |
| `paal` | **PAAL** | this work |

### Full Benchmark (resumable)

```bash
uv run python -m scripts.run_benchmark \
  --config configs/flash.yaml --modalities lidar image --seeds 1 10 100 \
  --strategies random least_conf entropy galaxy coreset paal epl \
  --override "model.soft_labels={temperature: 1.0, weight: 0.5}" \
  --override run_dir=runs/flash_soft
```

Finished runs are skipped, so an interrupted benchmark resumes where it
stopped.

### Full-Pool Reference

```bash
uv run python -m scripts.train_full_pool --config configs/flash.yaml --seed 1 --modality lidar
```

### Computational Cost

```bash
uv run python -m scripts.computational_cost \
  --jobs configs/flash.yaml:lidar configs/flash.yaml:image --out results/cost
```

## 📊 Outputs and Evaluation

Each run is saved to `<run_dir>/<modality>/<strategy>_seed<seed>/`:

- `log.jsonl`: one row per round with `n_labels`, `top1`, `top5`,
  `per_class_min_q`, `epochs` and timings.
- `queried.npz`: queried sample ids and the round in which each was queried.
- `stdout.log`: run log (when launched through `scripts.run_benchmark`).

Summarize learning curves, AUC and labels-to-target with:

```bash
uv run python -m scripts.evaluate_learning_curves \
  --runs runs/flash/lidar runs/flash_soft/lidar \
  --out results/flash_lidar --targets 0.6 0.7
```

Evaluate with **top-1 / top-5 accuracy** versus the labeled budget, the
**AUC** of the learning curve (mean accuracy over the budget points), and the
**number of sweeps** needed to reach a target accuracy relative to random
sampling.

## 📂 Project Structure

```text
paal/
├── configs/
│   ├── flash.yaml                       # FLASH preset (34 sectors)
│   ├── deepsense.yaml                   # DeepSense 6G scenario 33 (64 beams)
│   └── deepsense32.yaml                 # DeepSense 6G scenario 32 (extends deepsense.yaml)
├── scripts/
│   ├── preprocess_flash.py              # FLASH release -> samples.parquet
│   ├── voxelize_deepsense_lidar.py      # DeepSense point clouds -> 20^3 grids
│   ├── build_sample_table.py            # Sample table with power profiles
│   ├── split_pool_test.py               # Class-balanced pool/test split
│   ├── run_active_learning.py           # One strategy, one seed
│   ├── run_benchmark.py                 # Resumable benchmark runner
│   ├── train_full_pool.py               # Full-supervision reference
│   ├── evaluate_learning_curves.py      # Curves, AUC, labels-to-target
│   └── computational_cost.py            # Selection time, params, GFLOPs
├── src/alwl/
│   ├── config.py                        # YAML loading, extends, overrides
│   ├── data.py                          # Inputs, splits, soft targets
│   ├── ingest.py                        # DeepSense tables -> samples.parquet
│   ├── loop.py                          # Active-learning loop
│   ├── models.py                        # ResNet18 and 3-D CNN backbones
│   ├── train.py                         # Hard/soft-label training
│   ├── metrics.py                       # Accuracy and curve metrics
│   └── strategies/
│       ├── paal.py                      # PAAL and the EPL ablation
│       └── ...                          # Baseline strategies
├── pyproject.toml                       # Project metadata and dependencies
└── uv.lock                              # Locked dependency versions
```

## 🙏 Acknowledgments

- [FLASH](https://genesys-lab.org/multimodal-fusion-nextg-v2v-communications)
  (Salehi et al., INFOCOM 2022) and
  [DeepSense 6G](https://www.deepsense6g.net/) (Alkhateeb et al., 2023) for
  the datasets.
- Soltani et al., 2024, for the active-learning evaluation protocol.
- Built on [PyTorch](https://pytorch.org/) and
  [torchvision](https://pytorch.org/vision/).
