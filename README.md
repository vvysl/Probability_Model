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
| `pm.state` | Continuous \(\mathbf{x}_t\) — v1 catalog (~30, A–E) |
| `pm.targets` | Ternary displacement on 3/10/30 (1m bars) |
| `pm.model` | HGB ternary predictor (LGBM later/server) |
| `pm.eval` | Time splits + purge + metrics |
| `pm.train` | One model per horizon |
| `pm.pipeline` | Wire layers from config |

**Invariants:** native bars = **1m**; research default symbol = **ES**; code is
asset-agnostic (symbol/timeframe via config). Movement = displacement in ATR
units, not path length.

## Work order

1. **Structure** — layers + contracts ✅
2. **Horizons** — ternary \(y_{t,h}\) on 3/10/30 ✅
3. **Market constants** — v1 state catalog ✅

## Setup

```bash
cd Probability_Model
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
python -m pm train --config configs/default.yaml
```

Interactive train loop: [`notebooks/train.ipynb`](notebooks/train.ipynb)
(uses `data/ES_1m.csv` when present, else synthetic bars).

Requires `data_path` in config pointing at a local 1m OHLCV CSV
(`time,open,high,low,close,volume`).

## Data

Put local OHLCV under `data/` (gitignored). Export ES 1m from
Databento/Quanqora → CSV; do not commit caches or API keys.

## Status

HGB train loop ready (one model / horizon, balanced classes, purge=30).
Point `data_path` at ES 1m and run `python -m pm train`. LightGBM later on server.
