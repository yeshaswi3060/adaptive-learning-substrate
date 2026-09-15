"""Command-line entry point for deterministic experiment runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .experiment import load_config, run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the minimal deterministic Stage-1 / Experiment-000 scaffold."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/experiment_000.toml"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/latest"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_experiment(load_config(args.config), args.output_dir)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
