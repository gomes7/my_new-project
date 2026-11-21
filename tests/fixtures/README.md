# Test fixtures — provenance

Real market data captured on 2026-11-20 (UTC) for reproducible offline tests:

| File | Source | Notes |
|---|---|---|
| `BTCUSD_1d.csv` | Kraken public REST `OHLC` (XBTUSD, 1440) | last 700 closed daily bars |
| `EURUSD_1d.csv` | Yahoo chart `EURUSD=X` 60m, aggregated to 1d (22:00 UTC boundary) | vendor daily FX bars fail validation |
| `SPX_1d.csv` | Yahoo chart `^GSPC` 1d | cash index |
| `GOLD_1d.csv` | Yahoo chart `GC=F` 1d | front-month continuous future, unadjusted |
| `kraken_ohlc_sample.json` | raw Kraken response (trimmed to 30 rows) | parser test |
| `yahoo_chart_sample.json` | raw Yahoo chart response, GC=F 60m 5d | parser test |
| `sample_feed.xml` | hand-written RSS for parser tests | fictional headlines, not real news |

Yahoo's endpoint is unofficial; these files are for software testing only.
Synthetic series generated inside tests are labelled as such and never used for performance claims.
