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
