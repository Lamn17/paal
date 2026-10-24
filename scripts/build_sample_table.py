"""Build <processed>/samples.parquet from ``dataset.root``.

``dataset.root`` is either the DeepSense scenario folder (run `scripts.voxelize_deepsense_lidar`
first for LiDAR) or the table written by `scripts.preprocess_flash`.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from alwl.config import load_config
from alwl.ingest import ingest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config)
    data = cfg["dataset"]
    out = Path(data["processed"]) / "samples.parquet"
    ingest(data["root"], str(out), int(data["K"]))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
