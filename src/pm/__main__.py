"""CLI: python -m pm train --config configs/default.yaml"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="pm")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser(
        "train", help="Train HGB probability models from config"
    )
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
        tid = result.get("target_id")
        if tid == "atr_barrier_first":
            m = result["head"]["metrics"]
            summary = {
                "backend": result["backend"],
                "target_id": tid,
                "horizon": result["horizon"],
                "barrier_tau": result["barrier_tau"],
                "purge": result["purge"],
                "n_train": result["head"]["n_train"],
                "n_holdout": result["head"]["n_holdout"],
                "log_loss": m["log_loss"],
                "brier_mean": m["brier_mean"],
                "accuracy": m["accuracy"],
                "class_frac_true": m["class_frac_true"],
                "first_side_hit_rate": m.get("first_side_hit_rate"),
            }
        elif tid == "atr_barrier_touch":
            summary = {
                "backend": result["backend"],
                "target_id": tid,
                "horizon": result["horizon"],
                "barrier_tau": result["barrier_tau"],
                "heads": {
                    name: {
                        "base_rate": v["metrics"]["base_rate"],
                        "log_loss": v["metrics"]["log_loss"],
                        "auc": v["metrics"]["auc"],
                    }
                    for name, v in result["heads"].items()
                },
            }
        else:
            summary = {
                "backend": result["backend"],
                "target_id": tid,
                "horizons": {
                    h: {
                        "log_loss": v["metrics"]["log_loss"],
                        "accuracy": v["metrics"]["accuracy"],
                    }
                    for h, v in result.get("horizons", {}).items()
                },
            }
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
