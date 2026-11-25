"""CLI: python -m pm train --config configs/default.yaml"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="pm")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="Train one HGB model per horizon")
    p_train.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to YAML config",
    )
    p_train.add_argument(
        "--artifacts",
        type=Path,
        default=None,
        help="Artifacts directory (default: config.artifacts_dir or artifacts/run)",
    )

    args = parser.parse_args(argv)
    if args.cmd == "train":
        from pm.train import train_from_config

        result = train_from_config(args.config, artifacts_dir=args.artifacts)
        # Print compact metrics only
        summary = {
            "backend": result["backend"],
            "purge": result["purge"],
            "horizons": {
                h: {
                    "n_train": v["n_train"],
                    "n_holdout": v["n_holdout"],
                    "log_loss": v["metrics"]["log_loss"],
                    "brier_mean": v["metrics"]["brier_mean"],
                    "accuracy": v["metrics"]["accuracy"],
                }
                for h, v in result["horizons"].items()
            },
        }
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
