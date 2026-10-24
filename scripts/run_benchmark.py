"""Run every (modality, seed, strategy) listed under ``benchmark:`` in a config, one at a time.

Finished runs (log with all ``al.rounds + 1`` rows) are skipped, so an interrupted benchmark
resumes where it stopped. Each run is a separate process; a failure is reported and the
benchmark moves on.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime

from alwl.config import load_config
from scripts.run_active_learning import run_dir


def _finished(cfg: dict, strategy: str, seed: int) -> bool:
    log = run_dir(cfg, strategy, seed) / "log.jsonl"
    if not log.exists():
        return False
    rows = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
    return len(rows) >= int(cfg["al"]["rounds"]) + 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--modalities", nargs="*")
    parser.add_argument("--strategies", nargs="*")
    parser.add_argument("--seeds", nargs="*", type=int)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    cfg = load_config(args.config, args.override)
    bench = cfg.get("benchmark", {})
    modalities = args.modalities or bench.get("modalities", [cfg["model"]["modality"]])
    strategies = args.strategies or bench["strategies"]
    seeds = args.seeds or cfg["al"]["seeds"]
    for modality in modalities:
        for seed in seeds:
            for strategy in strategies:
                run_cfg = {**cfg, "model": {**cfg["model"], "modality": modality}}
                stamp = datetime.now().isoformat(timespec="seconds")
                if _finished(run_cfg, strategy, seed):
                    print(f"[{stamp}] skip {modality} {strategy} seed {seed} (done)", flush=True)
                    continue
                print(f"[{stamp}] run  {modality} {strategy} seed {seed}", flush=True)
                out = run_dir(run_cfg, strategy, seed)
                out.mkdir(parents=True, exist_ok=True)
                command = [
                    sys.executable, "-m", "scripts.run_active_learning",
                    "--config", args.config, "--strategy", strategy, "--seed", str(seed),
                    "--modality", modality, *[f"--override={o}" for o in args.override],
                ]
                with (out / "stdout.log").open("w") as fh:
                    code = subprocess.call(command, stdout=fh, stderr=subprocess.STDOUT)
                if code:
                    print(f"FAILED {modality} {strategy} seed {seed} (exit {code}); "
                          f"see {out / 'stdout.log'}", flush=True)
    print(f"[{datetime.now().isoformat(timespec='seconds')}] BENCHMARK_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
