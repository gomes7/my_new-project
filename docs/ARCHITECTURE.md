# QUANT AI — Technical Architecture (v0.1)

## 1. Environment audit (2026-11-20)

| Item | Finding |
|---|---|
| Runtime | Python 3.13, pandas 3.0, NumPy 2.5, SciPy 1.18, FastAPI 0.165, pydantic 2.13, pytest 9 |
| Tooling | Docker, Node 22 (not needed for v1) |
| Network | Limited egress. Package registries allowed. Market-data hosts were blocked by default; `api.kraken.com`, `query1.finance.yahoo.com` and `fred.stlouisfed.org` were added to this project's environment. Central-bank / news RSS hosts remain blocked. |
| Credentials | None. No broker, no paid data vendor, no news API. |
| Consequence | Real data integration verified for crypto (Kraken), FX/indices/commodities (Yahoo, unofficial) and macro (FRED). News ingestion is implemented and tested against fixtures, but **no live feed is configured**. No live execution exists. |

## 2. Technology decisions

| Need | Choice | Why / alternatives considered |
|---|---|---|
| Language | Python | Quant ecosystem; one language for research, backtest and serving. |
| Dataframes | pandas + NumPy | Sufficient at current scale (≈60k bars per run). Polars was **not** adopted: the per-bar loop, not columnar maths, was the bottleneck (profiling showed ~85% of time in pandas row access) — fixed with `FeatureFrame` (7× faster, identical results). |
| TA library | In-house causal indicators (`analysis/indicators.py`) | TA-Lib needs a C build; pandas-ta maintenance is uncertain. ~150 lines, every indicator tested for causality. |
| Storage | SQLite (WAL) | Single-host, single-writer workload; zero ops; append-only triggers + hash chain for the journal. **Move to PostgreSQL** when several writers/hosts appear; TimescaleDB only if tick/1-minute history across many symbols is stored. |
| Cache | None (no Redis) | Not justified: one process per role, data refresh is minutes-scale. |
| API | FastAPI | Typed, OpenAPI docs at `/docs`, trivial TestClient testing. |
| Dashboard | Single static HTML + vanilla JS + inline SVG | No build chain, no CDN dependency, works offline. React/Next.js deferred until the UI needs interactivity beyond read-only monitoring. |
| Deploy | Docker / docker-compose (api, paper worker, scanner) | Local first; API bound to 127.0.0.1. |
| Tests | pytest (offline by default; `-m network` for live endpoints) | |

## 3. Module map

```
config/                settings.yaml, instruments.yaml, calendar.yaml (versioned parameters)
quantai/
  config.py            pydantic settings; risk ceilings enforced at load; secrets only from env
  logging_setup.py     JSON logs; secret redaction
  models.py            Bar metadata, quality reports, signals, positions, trades
  data/                MODULE A — providers (Kraken, Yahoo, FRED, CSV, paid stubs), validation, bar cache, HTTP retry/rate-limit
  intel/               MODULES B/C/D — news ingestion+scoring, macro regime (FRED), calendar blackouts, event-reaction analysis
  analysis/            MODULES E/G — causal indicators, regime detection (+ calibration), multi-timeframe technical report
  strategies/          MODULE H — framework + trend_pullback, compression_breakout, range_mean_reversion
  selection.py         MODULE I — regime → applicable strategies → validated expectancy → conflict rejection
  ranking.py           MODULE J — explainable 0-100 score + calibration table
  risk/engine.py       MODULE K — independent approval/rejection and position sizing
  targets.py           MODULE L — daily target feasibility (never changes risk)
  execution/           MODULES M/N — conservative bar simulator (shared by backtest & paper), FX conversion
  decision.py          single per-bar pipeline used by backtest, paper and scanner
  backtest/            MODULE O — event-driven engine, metrics, walk-forward, Monte Carlo, sensitivity, promotion gate
  paper.py             MODULE N — live data + simulated orders, persisted state
  scanner.py           MODULE F — multi-asset scan
  monitoring.py        MODULE P — performance / data alerts
  journal.py           MODULE Q — append-only hash-chained journal + pattern analysis
  research.py          MODULE R — evidence -> proposals (stage 'proposed'); promotion only via the gate
  alerts.py            MODULE T — categories, cooldown, webhook
  killswitch.py        MODULE U — kill switch
  api/                 MODULE S — FastAPI + dashboard
  cli.py               operator interface
```

## 4. Data flow (one closed bar)

```
provider ──► drop forming bar ──► validate (FAILED ⇒ no signals) ──► causal features ──► regime label
                                                                            │
     macro regime (FRED) ─┐   MTF technical report ─┐   news/event risk ─┐  │
                          ▼                          ▼                   ▼  ▼
                         strategies (signal at bar close) ──► filters ──► selection (conflict ⇒ reject)
                                                                            │
                                                     score 0-100 (explainable, not a probability)
                                                                            │
                                         RISK ENGINE (independent) ── approve / reject + size
                                                                            │
                         journal (every candidate, incl. rejected) ◄────────┤
                                                                            ▼
                                         SimBroker: fill at NEXT bar open (+½ spread + slippage)
```

## 5. Key invariants (all covered by tests)

* **No look-ahead**: features are causal (test appends future bars and checks earlier values unchanged); signals at bar *i* are identical whether bars > *i* exist; fills happen at the next bar's open; FX conversion uses only closed FX bars; 4H aggregation is anchored so no source bar straddles a bucket close.
* **No silent stale data**: on provider failure the runtime returns cached bars *flagged FAILED* (`api_failure`), which blocks signals.
* **Risk engine cannot be bypassed**: `SimBroker.submit` refuses unapproved decisions; pending orders count toward open risk and correlated exposure.
* **Pessimistic fills**: stop assumed before target inside one bar; target on the entry bar not credited; gaps through stops fill at the open; entries gapping through the stop are cancelled.
* **Append-only journal**: UPDATE/DELETE blocked by triggers; SHA-256 chain detects edits made by bypassing triggers.
* **Paper == backtest**: same `decide()` and `SimBroker`; the paper trader makes exactly one decision per closed bar.

## 6. Scoring, regimes and what they are *not*

* The opportunity score is a weighted average of documented components (regime fit, MTF alignment, setup quality, R:R, validated expectancy, execution cost, volatility fit, macro, event risk, data quality, momentum). Missing inputs are neutral (0.5) and listed as `unavailable`. Weights are initial priors; `ranking.calibrate()` produces realised-R by score bucket, which must be monotonic before the score is trusted.
* Macro→asset bias (`MacroRegime.bias`) is an economic **prior**, unvalidated, with small weight.
* Regime calibration on real data (see docs/VALIDATION.md) shows trend labels carry **volatility** information but **not** reliable forward direction. They are used to gate strategies, not to predict.
