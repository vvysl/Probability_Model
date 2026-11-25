# Probability_Model

Open research lab: **probabilistic** forecasts of further price movement on
fixed horizons, from **continuous market state variables** (not discrete
pattern / indicator stacks).

## What this is

At each time \(t\), estimate something like:

\[
P(\text{outcome on horizon } h \mid \mathbf{x}_t)
\]

where \(\mathbf{x}_t\) is a vector of continuous, causally available market
constants (vol, location in range, distance to anchors in ATR units, session
progress, flow proxies, …) — not RSI-cross style pattern flags.

## What this is not

- Pattern detector / TradingView indicator soup
- Quanqora research pipeline / fill contracts / live passports
- A place for paid cache dumps or secrets (repo is **public**)

## Layers

See [ARCHITECTURE.md](ARCHITECTURE.md).

| Package | Role |
|---------|------|
| `pm.bars` | OHLCV load / schema |
| `pm.state` | Continuous \(\mathbf{x}_t\) (legacy v1 / alphabet — redesign next) |
| `pm.targets` | Legacy ATR barrier / first-touch (redesign next) |
| `pm.model` | Legacy HGB; next = LightGBM locally |
| `pm.eval` | Time splits + purge + metrics |
| `pm.train` | Legacy per-horizon loop |
| `pm.pipeline` | Wire layers from config |

**Invariants:** native bars = **1m**; research default symbol = **ES**; code is
asset-agnostic. Server is for **order execution only** — training runs on this Mac.

## Checkpoint (closed)

Today’s HGB line (ternary displacement → alphabet wave-1 → ATR barrier /
first-touch) is **archived as experience**. Side edge stayed ~coin; stacking
TV-style obvious features is not the path forward.

## Next (rebuild)

1. One horizon (not multi-h).
2. LightGBM locally (`lightgbm` was deferred — install here).
3. Three coupled models that share state: **price**, **position management**,
   **entries/exits**.
4. Features: real structure (e.g. Bollinger-class and other non-obvious), not
   RSI / MA / MI soup as the main bet.

## Setup

```bash
cd Probability_Model
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
python -m pm train --config configs/default.yaml
```

Requires `data_path` in config pointing at a local 1m OHLCV CSV
(`time,open,high,low,close,volume`).

## Data

Put local OHLCV under `data/` (gitignored). Export ES 1m from
Databento/Quanqora → CSV; do not commit caches or API keys.

## Status

**Checkpoint frozen.** Legacy train still works via
`python -m pm train --config configs/local_es.yaml` for reference only.
Rebuild starts next session.
