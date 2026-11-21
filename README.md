# QUANT AI — Market Intelligence, Research & Paper-Trading System

A modular, risk-first trading research platform: real market data with quality validation,
macro/news intelligence, regime detection, rule-based strategies, an independent risk engine,
a conservative execution simulator shared by backtesting and paper trading, walk-forward
validation with promotion gates, an append-only journal, alerts and a dashboard.

> **Status: research & paper trading only. No live execution exists.** No strategy in this
> repository has passed the promotion gate; nothing here is evidence of a profitable edge.
> This software does not give financial advice and does not guarantee returns.

## What works today (v0.1)

| Area | Status |
|---|---|
| Market data | Kraken (crypto, real-time, keyless), Yahoo chart (FX/indices/commodities; unofficial, delay unverified), FRED (macro), CSV import. Every bar set carries source, timestamps (UTC), frequency, retrieval time, delay status and a quality report. |
| Data validation | Missing values, duplicates, timestamp order/tz/future bars, OHLC consistency, non-positive prices, reverting spikes (bad ticks), gaps (24h markets), staleness, API failure, cross-provider discrepancy (crypto: Kraken checked live against Yahoo's aggregate). FAILED ⇒ signals blocked. |
| Intelligence | Macro regime from 14 FRED series with evidence; news RSS/Atom ingestion, classification, 0-100 event score, rumor/confirmation handling; calendar blackouts; post-event market-reaction measurement. |
| Analysis | Causal indicators (EMA, ATR, RSI, MACD, ADX, Bollinger, Donchian, VWAP, realised vol, efficiency ratio), regime classifier + calibration report, multi-timeframe technical report with key levels. |
| Strategies | `trend_pullback`, `compression_breakout`, `range_mean_reversion`, `tsmom` (time-series momentum with trailing stop) — explicit entry/stop/target/time-stop/trailing-stop, regime/volatility/liquidity/time filters, versioned by parameter fingerprint. |
| Selection & ranking | Conflict rejection, validated-expectancy preference (shrunk by sample size), explainable score, calibration table. |
| Risk engine | Sizing from stop distance incl. costs and FX; optional volatility-regime risk reduction (off by default); daily/weekly loss, drawdown, loss-streak cooldown, max positions, total & correlated exposure, leverage, stop sanity, spread, data quality, event blackout, kill switch. |
| Backtesting | Event-driven multi-instrument engine, next-bar fills, spread/slippage/commission/holding costs, walk-forward, bootstrap CI, Monte Carlo drawdown/ruin, parameter sensitivity, promotion gate. |
| Paper trading | Live data + simulated orders via the *same* decision pipeline and simulator; persistent state; one decision per closed bar. |
| Research | `quantai research` turns validation/journal evidence into journaled proposals (retire / redesign / filter / hypothesis); never auto-activated. Strategy versions fingerprinted and journaled with promotion decisions. |
| Journal | Every signal (approved or rejected) with reasons; fills, exits, outcomes, lessons; append-only with hash chain; pattern analysis. |
| Ops | Alerts (cooldown, webhook), kill switch, FastAPI + dashboard, JSON logs with secret redaction, Docker/compose, CI. |
| Tests | 106 offline tests + 2 live-endpoint tests (`-m network`). |

## Install & run

```bash
python -m pip install -e ".[dev]"        # Python >= 3.11
cp .env.example .env                     # optional: admin token, webhook, paid keys
pytest -q                                # offline test suite
pytest -q -m network                     # live endpoint checks (needs network access)

quantai health --probe BTCUSD EURUSD     # provider + data-quality probe
quantai macro                            # FRED macro regime with evidence
quantai calendar                         # scheduled events from config/calendar.yaml
quantai news                             # configured RSS feeds (none by default)
quantai scan                             # full-universe scan -> var/latest_scan.json
quantai backtest --symbols BTCUSD SPX GOLD --tf 1d
quantai validate --tf 1d                 # walk-forward + Monte Carlo + sensitivity + score calibration + promotion gate
quantai validate --vol-scaling --out latest_backtest_volscaled.json   # same, with vol-regime risk scaling
quantai regime-eval SPX                  # regime classifier calibration on history
quantai paper --cycles 1                 # one paper-trading cycle (use --cycles N --interval S)
quantai killswitch on --reason "manual"  # stop new trades, cancel pending simulated orders
quantai report                           # daily target feasibility + journal analysis
quantai research                         # evidence-based proposals (journaled, never auto-activated)
quantai journal verify                   # tamper check
quantai serve                            # dashboard at http://127.0.0.1:8000  (API docs at /docs)
```

Docker: `docker compose up -d --build` (api on 127.0.0.1:8000, paper worker every 15 min, scanner hourly).

## Configuration

* `config/settings.yaml` — mode, universe, timeframe, risk limits, target, ranking thresholds, data checks, execution assumptions, promotion gates, alerts, news feeds. Risk ceilings are enforced at load (risk/trade > 0.5% needs `exceptional_justification`; > 1% impossible; daily loss ceiling ≤ 2%).
* `config/instruments.yaml` — provider mapping and contract/cost assumptions. **Spreads and contract sizes are conservative assumptions (`spec_source: assumption`)**; replace with your broker's specifications before trusting position sizes.
* `config/calendar.yaml` — high-impact events (UTC) that create risk blackouts.
* Environment (`.env`): `QUANTAI_DATA_DIR`, `QUANTAI_ADMIN_TOKEN` (required for `POST /api/killswitch`), `QUANTAI_ALERT_WEBHOOK_URL`, paid provider keys. Secrets are never read from files in the repo and are redacted from logs.

## To activate more capability

| Capability | Requirement |
|---|---|
| Licensed FX/index/stock data with bid/ask | Subscription (e.g. OANDA account, Polygon) + implement the adapter behind `PaidProviderStub`; key in `QUANTAI_<NAME>_API_KEY` |
| Live news | Add feed URLs to `news.feeds` and allow their hosts in the environment's network policy |
| Economic calendar API | Licensed provider; adapter next to `intel/calendar.py` |
| Live execution | **Not implemented.** Requires explicit authorisation, a separate config and code path, broker validation, trade-only keys (no withdrawal), and the live-readiness review in `docs/LIVE_READINESS.md` |

## Documentation

* `docs/ARCHITECTURE.md` — audit, technology decisions, module map, invariants
* `docs/DATA_PROVIDERS.md` — provider evaluation and recommendations
* `docs/VALIDATION.md` — real-data validation results for v0.1 and Phase 2 (no edge demonstrated yet)
* `docs/LIVE_READINESS.md` — readiness review (verdict: not ready)
