# Validation results — first real-data run (2026-11-20)

Command: `quantai validate --tf 1d` over the 21-instrument universe. Costs included: half-spread
per fill (configured assumptions; Kraken base-tier taker fee 0.40% for crypto), slippage
0.05×ATR on market/stop fills, holding costs, pessimistic intrabar ordering. Risk 0.25%/trade
with all account-level limits active.

**Bottom line: no strategy shows a demonstrated edge. None is promoted to paper trading.**
This is the system working as designed — rejecting unproven strategies — not a failure to hide.

## Data actually used

| Group | Source | History used | Quality |
|---|---|---|---|
| 8 FX pairs | Yahoo 60m → 1d (22:00 UTC) | 2024-02 → 2026-11 (~775 bars; Yahoo's 730-day hourly limit) | ok |
| NDX, SPX, DJI, DAX, CAC40, FTSE | Yahoo 1d | 2014-11 → 2026-11 | ok / degraded (suspected bad ticks flagged) |
| Gold, silver, Brent | Yahoo front-month continuous futures (roll gaps **not** adjusted) | 2014-11 → 2026-11 | ok / degraded |
| WTI | Yahoo `CL=F` | — | **FAILED** (negative prices, April 2020) → excluded automatically |
| BTC, ETH, SOL | Kraken | 2024-11 → 2026-11 (Kraken's 720-bar limit) | ok |

Walk-forward: 4 rolling folds, 60% of the timeline for training, out-of-sample (OOS) test windows
covering 2022-02 → 2026-11. Parameters picked on each training window by
`mean(R) − 0.5·stderr(R)`, frozen, then tested on the next unseen window. Because FX/crypto
history is short, early OOS folds are dominated by indices and commodities.

## Results (out-of-sample, after costs)

| Strategy | OOS trades | Expectancy | 95% bootstrap CI | Profit factor | MC p95 drawdown | Gate |
|---|---|---|---|---|---|---|
| trend_pullback | 245 | **+0.002 R** | −0.108 … +0.119 | 1.08 | 7.5% | research |
| compression_breakout | 222 | **−0.183 R** | −0.332 … −0.036 | 0.75 | 16.5% | research |
| range_mean_reversion | 6 | −0.577 R | −1.183 … +0.193 | 0.36 | 1.7% | research |

Per-fold OOS expectancy:

* trend_pullback: −0.235, +0.232, +0.040, −0.031 R — unstable sign across periods; `rr` chosen 3.0 / 1.5 / 3.0 / 2.0 (parameter instability).
* compression_breakout: +0.180, +0.149, −0.260, −0.287 R — deteriorating; the CI excludes zero on the **negative** side, i.e. evidence the rule loses money after costs on this universe.
* range_mean_reversion: 6 trades in 4.8 years — the entry conditions (range regime + ADX < 18-22 + z ≤ −1.8 + confirmation) are too restrictive on daily bars to evaluate. Needs a different timeframe or redesign, not parameter tuning.

Earlier exploratory runs show how fragile in-sample numbers are: on a 9-instrument subset the same
trend_pullback walk-forward gave +0.10 R (CI −0.04 … +0.26). Adding 12 instruments took it to
+0.00 R. The bootstrap-CI gate rejected both.

## Regime classifier calibration (`quantai regime-eval`, 10-bar forward returns)

| | SPX | Gold | EUR/USD |
|---|---|---|---|
| strong_uptrend: mean fwd return / hit-rate | +0.36% / 66% | +0.43% / 54% | +0.01% / 36% |
| strong_downtrend: mean fwd return / hit-rate | **+1.59%** / 69% | +0.53% / 54% | −0.26% / 36% |
| strong_downtrend: mean \|fwd return\| | **4.48%** | 2.49% | 0.93% |
| strong_uptrend: mean \|fwd return\| | 1.78% | 2.69% | 1.09% |
| unclassifiable share | 18% | 21% | 39% |

Findings:
1. Trend labels **do not** forecast direction (SPX rebounds after strong downtrends). They should not be used as directional signals.
2. Labels carry **volatility** information (SPX forward absolute move 2.5× larger in strong downtrends) — appropriate for strategy gating and risk scaling.
3. The unclassifiable share is high (18-39%); thresholds need calibration per asset class.
Overlapping windows inflate t-statistics; treat them as indicative.

## Live scan (2026-11-21 00:05 UTC)

21 instruments scanned on real data → **NO TRADE — NO VALID SETUP**. Candidates seen:
GBPJPY trend_pullback short (score 66.7 < 80), DJI trend_pullback (rejected: expansion regime),
FTSE breakout short (rejected: relative volume filter). WTI blocked by data quality.

## What should happen next (evidence-driven, not curve-fitting)

* Do **not** tune these three rules on the same data until they pass — that would be overfitting the OOS period.
* Get longer, licensed FX/crypto history so OOS folds are not dominated by indices.
* Test hypotheses that the calibration supports: volatility-regime-conditioned position sizing; trend following on longer horizons with roll-adjusted futures; mean reversion on intraday bars with real bid/ask.
* Each new hypothesis = new strategy version, fingerprinted and journaled, through the same gate.

---

# Phase 2 validation (2026-11-21)

Same universe, folds, costs and gate as above. Both runs: `quantai validate` and
`quantai validate --vol-scaling`.

**Bottom line: still no strategy promoted.** The new `tsmom` is the best so far, but its
confidence interval includes zero and it is sensitive to its parameters.

## Out-of-sample results

| Strategy | OOS trades | Expectancy | 95% CI | PF | Max DD | MC p95 DD | Gate |
|---|---|---|---|---|---|---|---|
| trend_pullback | 245 | +0.002 R | −0.108 … +0.119 | 1.08 | 3.7% | 7.5% | research |
| compression_breakout | 222 | −0.183 R | −0.332 … −0.036 | 0.75 | 10.3% | 16.5% | research (retire) |
| range_mean_reversion | 6 | −0.577 R | — | 0.36 | 0.7% | 1.7% | research (redesign) |
| **tsmom (new)** | 214 | **+0.048 R** | −0.087 … +0.193 | 1.10 | 2.1% | 5.7% | research |

`tsmom` per fold: +0.015, +0.070, +0.134, −0.022 R (3 of 4 positive). Walk-forward picked
`long_lb=150` in every fold (stable), `trail_atr` 2.5 → 3.5. Failed gates: profit factor
(1.10 < 1.15), expectancy (0.048 < 0.05), bootstrap lower bound ≤ 0, and parameter sensitivity:
some ±20% changes removed most of the edge (degradation up to 1.0). It is a candidate for more
data, not for paper trading.

## Volatility-regime risk scaling (risk ×0.5 in high-volatility regimes)

| Strategy | Expectancy off → on | Max DD off → on | MC p95 DD off → on |
|---|---|---|---|
| trend_pullback | +0.002 → +0.040 R | 3.7% → 2.7% | 7.5% → 6.1% |
| tsmom | +0.048 → +0.043 R | 2.1% → 2.1% | 5.7% → 5.9% |
| compression_breakout | −0.183 → −0.184 R | 10.3% → 9.7% | 16.5% → 16.5% |

Reading: R is normalised per trade, so scaling mostly changes *which* trades fit inside the
account risk limits, not per-trade outcomes. It lowered drawdown for trend_pullback and was
neutral elsewhere. The effect is small and not significant, so **`vol_scaling` stays off**.

## Score calibration (realised R by score bucket, OOS)

| Strategy | < 60 | 60–70 | 70–80 | Monotonic? |
|---|---|---|---|---|
| trend_pullback | +0.12 (n=37) | +0.01 (n=170) | −0.16 (n=38) | **no (inverted)** |
| compression_breakout | −0.09 (n=107) | −0.26 (n=114) | (n=1) | no |
| tsmom | — | +0.02 (n=124) | +0.09 (n=90) | yes |

In backtests the score lacks multi-timeframe, macro and news inputs (they count as neutral),
and **no OOS trade scored ≥ 80**. So the live threshold of 80 is not calibrated: it works as a
strict filter, not as a validated quality measure. For trend_pullback the score ranks trades
backwards. Before score-based gating can be trusted, its weights need re-fitting on training
folds only and then checking out of sample.

## Crypto cross-provider check

Kraken vs Yahoo aggregate daily closes over 720 days: median absolute gap 0.017% (BTC),
0.021% (ETH), 0.040% (SOL); p95 0.08–0.18%; 1–2 days above 0.5%. A latest-bar gap above 0.5%
now blocks signals; an unavailable second source only degrades.
