# Architecture — Probability_Model

## Goal

At each bar \(t\), map a continuous market state \(\mathbf{x}_t\) to a
probability of further price movement on horizon(s) \(h\):

\[
P(y_{t,h} \mid \mathbf{x}_t)
\]

This is **not** a pattern library. Features are continuous, causally available
state variables (“market constants”), not discrete indicator events.

## Layers

```
data/ (local, gitignored)
  → bars          normalize OHLCV
  → state         continuous x_t  (registry filled in step 3)
  → targets       y_{t,h}         (horizons filled in step 2)
  → model         Predictor       (fit / predict_proba)
  → eval          time-ordered splits (+ purge/embargo later)
  → artifacts/    run outputs (gitignored)
```

| Layer | Responsibility | Must not |
|-------|----------------|----------|
| `bars` | Load/normalize OHLCV schema | Invent features or labels |
| `state` | Continuous \(\mathbf{x}_t\) from past only | Discrete pattern flags; look-ahead |
| `targets` | Outcomes on horizons \(h\) | Bake trading costs/stops into label without declaring it |
| `model` | \(P(y \mid x)\) | Own data I/O or eval splits |
| `eval` | Time-safe train/holdout | Random K-fold; mix future into train |
| `pipeline` | Wire layers from config | Hard-code horizons/constants |

## Contracts

### Bars

Columns: `time, open, high, low, close, volume` (UTC). Index monotonic
increasing. **Native timeframe is always 1m** (horizons counted in 1m bars).
Symbol is a config parameter (research default `ES`); package code stays
asset-agnostic. Local files under `data/` only — never commit paid caches.

### StateVar

- Stable string `id`
- `compute(bars) -> Series | DataFrame` aligned to bars index
- Uses only information available at or before each \(t\) (incl. warmup NaNs)
- Catalog still empty until **step 3**

### TargetSpec

- Declares horizon set and outcome mapping \(y_{t,h}\)
- **Step 2 (current):** `ternary_displacement` on \(h \in \{3,10,30\}\)

**Movement** = close-to-close **displacement** in causal ATR units (not path
length; time is only the horizon container):

\[
z_h = \frac{\mathrm{close}_{t+h}-\mathrm{close}_t}{\mathrm{ATR}_t},\quad
y_{t,h}=
\begin{cases}
+1 & z_h > \tau \\
-1 & z_h < -\tau \\
0 & |z_h| \le \tau
\end{cases}
\]

Default \(\tau = 0.15\), ATR Wilder period 14. Labels at the tail (no
\(\mathrm{close}_{t+h}\)) and during ATR warmup are NaN.

Path metrics (MFE/MAE) and “invalidation machine” ideas are **out of** this
label; they may become later layers.

### Predictor

- `fit(X, y)` / `predict_proba(X)`
- Probability shape must match the active `TargetSpec`
- Concrete backend (HGB, LightGBM, …) deferred

### Eval

- Time-ordered holdout (and later purge/embargo for overlapping labels)
- Metrics after state features exist

## Work order

1. **Structure** — layers + empty registries ✅
2. **Horizons** — ternary displacement 3/10/30 ✅
3. **Market constants** — fill `state` registry / config

## Explicitly out of scope (for now)

- Indicator / Pine pattern ports
- Quanqora fill contracts, passports, live wiring
- Training loops, SHAP, stop/TP backtests
- Path-based labels / invalidation-time models
- Committing secrets or market data dumps
