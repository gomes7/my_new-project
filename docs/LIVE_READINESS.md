# Live-readiness review — v0.1

**Verdict: NOT READY. Live trading must not be enabled.** It is not implemented, and the
prerequisites below are unmet.

| Area | State | Blocking? |
|---|---|---|
| Strategy performance | No strategy passed the walk-forward promotion gate (see VALIDATION.md). Best candidate `tsmom`: +0.048 R OOS, CI includes zero. Opportunity score not calibrated. | **Yes** |
| Paper-trading record | None yet. Gate requires ≥ 30 paper trades over ≥ 30 days consistent with validated expectations. | **Yes** |
| Drawdown | Monte Carlo drawdown estimates exist only for unvalidated strategies. | Yes |
| Execution assumptions | Conservative bar-based fills (pessimistic intrabar ordering, half-spread per fill, ATR slippage, commissions, holding costs). Spreads for Yahoo-sourced instruments are **assumed**, not observed. Crypto spread observed live from Kraken. | Yes, for FX/indices |
| Risk controls | Implemented and tested: sizing, daily/weekly/drawdown limits, loss-streak cooldown, exposure & correlation limits, leverage, stop sanity, spread, data quality, blackouts, kill switch. | No |
| Data reliability | Kraken and FRED reliable. Yahoo is unofficial with documented defects (monthly-for-daily, inconsistent FX daily bars, partial bars, negative WTI prints). Needs a licensed bid/ask feed. Crypto is now cross-checked live (Kraken vs Yahoo aggregate); FX/indices/commodities still have a single source. | **Yes** |
| Security | Secrets via env only, log redaction, admin token for mutations, API bound to localhost, append-only hash-chained journal. Not done: authentication for read endpoints, TLS, secrets manager, backups schedule, external error monitoring. | Yes, before any remote deployment |
| Operational stability | Single-host processes; no supervisor beyond docker restart; no exchange-session calendars for indices/commodities. | Yes |
| Broker integration | None. Needs an adapter matching `SimBroker` semantics, contract-spec validation, reconciliation, idempotent order IDs, and trade-only credentials without withdrawal permission. | **Yes** |

## Minimum path to a live pilot (each step gated, explicit approval at the end)

1. Licensed data with bid/ask for the instruments to be traded; re-run validation on it.
2. A strategy passing every promotion gate out-of-sample on that data.
3. ≥ 30 days / ≥ 30 trades of paper trading within the validated expectancy CI.
4. Broker adapter in a *separate* module and config, tested against the broker's sandbox.
5. Security hardening (auth, TLS, secrets manager, backups, monitoring).
6. Written approval from the account owner, small fixed size, kill switch drill.
