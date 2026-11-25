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
  → state         continuous x_t  (v1 catalog ~30 features)
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
- **v1 catalog** (`pm.state.catalog.V1_STATE_IDS`, 30 ids) — continuous only:

| Block | Ids | Notes |
|-------|-----|-------|
| A returns | `ret_atr_{1,3,5,15,30,60}` | \(r_{\mathrm{lag}}/\mathrm{ATR}_{14}\) |
| B vol | `atr_14/60/240`, `atr_ratio_14_60`, `atr_ratio_60_240`, `rv_14/60` | Regime + expansion |
| C candle | `body_atr`, `upper_wick_atr`, `lower_wick_atr`, `range_atr`, `close_pos`, `dir_consist_15` | Structure in ATR; `close_pos=0.5` if range 0 |
| D trend | `dist_ema_{10,30,60}`, `ema_slope_10` (lag 5), `ema_slope_60` (lag 15), `eff_30` | Slopes /ATR14; efficiency ∈ [0,1] |
| E volume | `rvol_5/30`, `vol_z_60`, `vol_chg`, `vol_x_ret` | Relative only — no raw volume |

ATR denominators default to ATR(14) unless the feature *is* a vol scale/ratio.

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

Per-horizon realizable thresholds (ATR14): \(\tau_3=0.75\), \(\tau_{10}=1.25\),
\(\tau_{30}=2.0\). Labels at the tail (no \(\mathrm{close}_{t+h}\)) and during
ATR warmup are NaN.

Eval extra (not a train label): on true-flat rows, score residual \(\mathrm{sign}(z)\)
vs model tilt \(P(\mathrm{up})>P(\mathrm{down})\) → flat micro-direction hit-rate.

Path metrics (MFE/MAE) and “invalidation machine” ideas are **out of** this
label; they may become later layers.

### Predictor

- `fit(X, y)` / `predict_proba(X)` → ternary proba `(n, 3)` = `[down, flat, up]`
- **One model per horizon** (`y_h3`, `y_h10`, `y_h30`)
- **v1 backend:** `HistGradientBoostingClassifier` (`pm.model.hgb`), `class_weight=balanced`
- **Later:** LightGBM on a server (optional extra; needs libomp on macOS)
- Train: time holdout + `purge=max(h)` (`pm.train`); metrics log-loss / Brier / accuracy

### Eval

- Time-ordered holdout (and later purge/embargo for overlapping labels)
- Metrics after state features exist

## Work order

1. **Structure** — layers + empty registries ✅
2. **Horizons** — ternary displacement 3/10/30 ✅
3. **Market constants** — v1 state catalog (~30) ✅
4. **Train** — HGB per horizon ✅; LGBM on server later

## Explicitly out of scope (for now)

- Indicator / Pine pattern ports
- Quanqora fill contracts, passports, live wiring
- Training loops, SHAP, stop/TP backtests
- Path-based labels / invalidation-time models
- Committing secrets or market data dumps
