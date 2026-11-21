# Data providers and trading infrastructure — evaluation

Capabilities below are those documented by each vendor as of this project's knowledge; **verify current
pricing and terms before subscribing** (they change often). Only rows marked *verified here* were
exercised from this environment.

## Market data

| Provider | Coverage | Real-time? | Cost model | Notes | Status in QUANT AI |
|---|---|---|---|---|---|
| **Kraken public REST** | Kraken spot crypto | Yes (Kraken's own book) | Free, no key | `OHLC` returns at most 720 most-recent bars per interval (≈2 y daily, 120 d 4H, 30 d 1H). Ticker gives best bid/ask → real spread. Longer history via Kraken's downloadable OHLCVT files. | **Implemented, verified here** |
| **Yahoo Finance chart endpoint** | FX, indices, futures, stocks | Delay not guaranteed | Free, unofficial | Undocumented, can change or rate-limit; terms restrict non-personal use. Found here: `range=max` silently returns monthly bars; daily FX bars fail OHLC consistency (90/3121 EURUSD, 193 USDJPY) → we build FX daily from hourly; appends a partial bar with non-zero seconds; continuous CL=F contains negative 2020 prints. | **Implemented, verified here; research only** |
| **FRED** (St. Louis Fed) | US macro, rates, spreads, VIX | Official release cadence | Free (CSV without key; JSON API with free key) | Authoritative for macro; release lags apply (monthly series ~2-6 weeks). | **Implemented, verified here** |
| CSV import | Any licensed export | n/a | n/a | For vendor history (Kraken OHLCVT files, Dukascopy, broker exports). | **Implemented** |
| Polygon.io | US stocks/options/indices, FX, crypto | Yes on paid tiers | Subscription | Good REST + websocket APIs, clean US equity data. | Interface stub (`PaidProviderStub`) |
| OANDA v20 | FX/CFD quotes from a broker | Yes (account required) | Free with account | Bid/ask streaming, also an execution API (future broker adapter). | Interface stub |
| Twelve Data / Alpha Vantage | Multi-asset | Delayed on free tiers | Freemium | Strict free-tier rate limits. | Not implemented |
| Interactive Brokers TWS/Gateway | Global multi-asset | Yes (market-data subscriptions) | Account + per-exchange fees | Best breadth; operationally heavier (gateway process). | Not implemented |
| Binance / Coinbase public APIs | Crypto | Yes | Free | Blocked by this environment's network policy; would add a second crypto source for cross-provider checks. | Not implemented |

## News / calendar

| Source | Notes | Status |
|---|---|---|
| Central-bank RSS (Fed, ECB, BoE, BoJ) | Authoritative, free; hosts must be allowed by the network policy | Parser + scoring implemented and tested on fixtures; **no live feed configured** |
| Commercial news APIs (e.g. Benzinga, Reuters/LSEG, Bloomberg) | Licensed, low latency, entity-tagged | Not implemented (needs subscription) |
| Economic calendar | No free, reliable, machine-readable calendar was reachable | `config/calendar.yaml` (operator-maintained) drives risk blackouts; adapter interface for a licensed calendar |

## Execution (future, not enabled)

Broker adapters must implement the same order semantics as `SimBroker` and pass broker-specific
validation (contract specs, min size, margin, trading hours). Requirements already encoded:
credentials from environment only, keys without withdrawal permission, explicit
`QUANTAI_LIVE_TRADING_AUTHORIZED=yes` **plus** a separate live configuration and code path (which does not exist yet), kill switch.

## Recommendation

1. Keep Kraken + FRED (free, reliable) for crypto and macro.
2. Replace Yahoo with a licensed FX/CFD feed that provides **bid/ask** (OANDA v20 if you open an account; Polygon for US equities/indices) before trusting any FX/index result.
3. Add a second crypto source (Coinbase) to activate cross-provider discrepancy checks (`compare_providers`).
4. Add central-bank RSS hosts to the network allowlist and configure `news.feeds`.
