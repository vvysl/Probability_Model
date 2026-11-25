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
| F alphabet wave-1 | `vwap_dist`, `session_range_pos`, `dist_session_extreme_atr`, `gap_atr`, `mins_in_session`, `mins_to_session_end`, `session_vol_frac`, `gk_rv_ratio`, `range_compression`, `vol_pct`, `ret_skew`, `ret_accel`, `sma_open_approach` | Quanqora Core A subset; Warsaw session canon; no Quanqora import |

Default matrix = v1 + wave-1 (**43** ids). ATR denominators default to ATR(14)
unless the feature *is* a vol scale/ratio.

### TargetSpec

**Primary (current):** `atr_barrier_first` — continuous probabilistic system,
not a trade strategy. On each bar \(t\), with \(n=30\), \(\tau=2\cdot\mathrm{ATR}_{14}\):

\[
y_t \in \{\text{up first},\ \text{down first},\ \text{none}\}
\]

Path on \((t+1..t+n)\): which barrier is touched **first**. Same-bar double
hit → `none`. Model outputs

\[
P(\text{up first}),\ P(\text{down first}),\ P(\text{none})
\]

Comparing \(P(\text{up first})>P(\text{down first})\) is a thin consumption layer,
not baked into training as a trade rule.

**Also registered:** `atr_barrier_touch` (independent hits); `ternary_displacement`
(close-to-close legacy).

### Predictor (legacy checkpoint)

- First-barrier: **one ternary HGB** → proba `[down_first, none, up_first]`
- Backend: `HistGradientBoostingClassifier`, `class_weight=balanced`
- Train: time holdout + `purge=n`; log-loss / Brier / first-side hit-rate

This line is **frozen**. Next backend = **LightGBM on this Mac**; server is for
order execution only.

### Eval

- Time-ordered holdout + purge/embargo for overlapping labels

## Work order

1. **Structure** — layers + empty registries ✅
2. **Target** — first ATR barrier \(n=30,\tau=2\) ✅ (legacy)
3. **Market constants** — v1 + alphabet wave-1 (43) ✅ (legacy)
4. **Train** — ternary HGB for first-touch probs ✅ (legacy; edge ~coin)
5. **Rebuild** — one horizon; three coupled models (price / position /
   entry-exit); LightGBM local; non-obvious features — **next**

## Explicitly out of scope

- Quanqora fill contracts, passports, live wiring (in this repo)
- Committing secrets or market data dumps
- Treating TV-top-1 indicator soup as the main feature bet
