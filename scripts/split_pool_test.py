"""Write the train/test split of every seed in ``al.seeds`` to <processed>/splits/."""

from __future__ import annotations

import argparse
import json

import numpy as np

from alwl.config import load_config
from alwl.data import load_samples, make_split, split_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    cfg = load_config(args.config, args.override)
    data = cfg["dataset"]
    frame = load_samples(cfg)
    y_of = frame.set_index("sample_id")["y"]
    manifest = []
    for seed in cfg["al"]["seeds"]:
        pool, test = make_split(
            frame,
            int(data.get("test_per_class", 10)),
            data.get("split_by", "sample"),
            int(seed),
            data.get("max_test_fraction"),
        )
        path = split_path(cfg, int(seed))
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, pool_ids=pool, test_ids=test)
        counts = np.bincount(y_of.loc[test], minlength=int(data["K"]))
        manifest.append(
            {
                "seed": int(seed),
                "path": str(path),
                "n_pool": len(pool),
                "n_test": len(test),
                "test_per_class_min": int(counts[counts > 0].min()),
                "test_per_class_max": int(counts.max()),
                "pool_classes": int(len(np.unique(y_of.loc[pool]))),
            }
        )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
