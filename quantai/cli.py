"""Command-line interface: `python -m quantai <command>` (or `quantai <command>`)."""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict

import numpy as np
import pandas as pd

from quantai import __version__
from quantai.config import get_instruments, get_settings
from quantai.logging_setup import setup_logging
from quantai.models import utcnow


def _rt():
    from quantai.runtime import Runtime
    return Runtime()


def _strategies(names):
    import quantai.strategies as S
    return [S.registry.create(n) for n in names]


def _print(obj) -> None:
    print(json.dumps(obj, default=str, indent=2))


def _load_history(rt, symbols, tf):
    ins = get_instruments()
    data, notes = {}, {}
    for s in symbols:
        md = rt.fetch(ins[s], tf, live=False)
        if md.df.empty:
            notes[s] = "no data"
            continue
        data[s] = md.df
        notes[s] = f"{len(md.df)} bars {md.df.index[0].date()}..{md.df.index[-1].date()} [{md.meta.source}, {md.quality.status.value}]"
    return data, notes


def cmd_health(a):
    rt = _rt()
    ok, bad = rt.journal.verify()
    ins = get_instruments()
    probes = {}
    for sym in a.probe:
        md = rt.fetch(ins[sym], "1d")
        probes[sym] = {"bars": len(md.df), "last_bar": str(md.df.index[-1]) if len(md.df) else None,
                       "quality": md.quality.status.value, "issues": [i.code for i in md.quality.issues],
                       "delay": md.meta.delay_status.value}
    _print({"version": __version__, "mode": rt.settings.mode, "data_dir": str(rt.data_dir), "journal_ok": ok,
            "journal_first_bad": bad, "kill_switch": rt.kill.status(), "providers": rt.api_status, "probes": probes,
            "live_trading": "not implemented"})


def cmd_fetch(a):
    rt = _rt()
    md = rt.fetch(get_instruments()[a.symbol], a.tf)
    print(md.df.tail(a.n).to_string())
    _print({"source": md.meta.source, "provider_symbol": md.meta.provider_symbol, "retrieved_at": md.meta.retrieved_at,
            "delay_status": md.meta.delay_status.value, "timezone": md.meta.timezone, "notes": md.meta.notes,
            "quality": md.quality.to_dict()})


def cmd_macro(a):
    from quantai.intel.macro import classify_macro, fetch_macro
    rt = _rt()
    m = classify_macro(fetch_macro())
    rt.write_json("latest_macro.json", m.to_dict())
    _print(m.to_dict())


def cmd_news(a):
    from quantai.intel.news import fetch_feed, score_events
    rt = _rt()
    items, errors = [], {}
    for f in rt.settings.news.feeds:
        try:
            items += fetch_feed(f.url, f.name, f.credibility)
        except Exception as e:  # noqa: BLE001
            errors[f.name] = str(e)
    ev = score_events(items, set(rt.settings.universe))
    out = {"generated_at": utcnow().isoformat(), "feeds": len(rt.settings.news.feeds), "items": len(items),
           "errors": errors, "events": [e.to_dict() for e in ev[:100]]}
    if not rt.settings.news.feeds:
        out["note"] = "no feeds configured (settings.yaml news.feeds); hosts must also be allowed by the network policy"
    rt.write_json("latest_events.json", out)
    _print({k: v for k, v in out.items() if k != "events"} | {"top": [(e.score, e.title) for e in ev[:10]]})


def cmd_calendar(a):
    from quantai.config import CONFIG_DIR
    from quantai.intel.calendar import YamlCalendar
    rt = _rt()
    up = YamlCalendar(CONFIG_DIR / "calendar.yaml").upcoming(hours=a.hours)
    out = {"generated_at": utcnow().isoformat(), "events": [{**asdict(e), "time": e.time.isoformat()} for e in up]}
    rt.write_json("latest_calendar.json", out)
    _print(out)


def cmd_scan(a):
    from quantai.config import CONFIG_DIR
    from quantai.intel.calendar import YamlCalendar
    from quantai.intel.macro import MacroRegime
    from quantai.intel.news import ScoredEvent
    from quantai.paper import PaperTrader
    from quantai.scanner import scan
    rt = _rt()
    st = rt.settings
    symbols = a.symbols or st.universe
    mj = rt.read_json("latest_macro.json")
    macro = None
    if mj:
        from quantai.intel.macro import Evidence
        macro = MacroRegime(**{**mj, "evidence": [Evidence(**e) for e in mj["evidence"]]})
    ej = rt.read_json("latest_events.json")
    events = [ScoredEvent(**e) for e in ej["events"]] if ej and ej.get("events") is not None and st.news.feeds else None
    acct = PaperTrader(rt, symbols, [], st.primary_timeframe).state.acct  # scan against the paper account's risk state
    res = scan(rt, symbols, _strategies(st.strategies), acct, primary_tf=a.tf or st.primary_timeframe, macro=macro,
               events=events, calendar=YamlCalendar(CONFIG_DIR / "calendar.yaml").events())
    out = asdict(res)
    out["api_status"] = rt.api_status
    rt.write_json("latest_scan.json", out)
    print(res.message)
    rows = [(m["symbol"], m.get("status"), m.get("regime"), m.get("bias"), m.get("quality")) for m in res.markets]
    print(pd.DataFrame(rows, columns=["symbol", "status", "regime", "mtf_bias", "quality"]).to_string(index=False))
    for o in res.opportunities + res.rejected:
        print(f"{o['decision']:>8} {o['symbol']:<7} {o['strategy']:<22} {o['direction']:<5} score={o['score']} "
              f"entry~{o['entry_ref']:.5g} stop={o['stop']:.5g} tgt={o['target']:.5g} rr={o['reward_risk']:.2f} {o['reasons']}")


def cmd_backtest(a):
    from quantai.backtest import metrics
    from quantai.backtest.engine import run_backtest
    rt = _rt()
    st = rt.settings
    data, notes = _load_history(rt, a.symbols or st.universe, a.tf)
    res = run_backtest(data, a.tf, _strategies(a.strategies or st.strategies), get_instruments(), st,
                       min_score=a.min_score)
    s = metrics.summary(res.trades, res.equity, st.account.starting_capital, st.backtest.annualization_days)
    print("DATA:"); [print(f"  {k}: {v}") for k, v in notes.items()]
    if res.skipped_symbols:
        print("SKIPPED (data quality):"); [print(f"  {k}: {v}") for k, v in res.skipped_symbols.items()]
    print("\nIN-SAMPLE SUMMARY (default parameters — NOT evidence of edge):")
    _print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()})
    for by in ("strategy", "symbol", "regime", "session"):
        b = metrics.breakdown(res.trades, by)
        if not b.empty:
            print(f"\nBy {by}:\n{b.round(3).to_string()}")
    rej = pd.Series([r for x in res.signals if x.decision == "rejected" for r in x.reasons]).value_counts().head(10)
    print(f"\nSignals: {len(res.signals)} (approved {sum(x.decision == 'approved' for x in res.signals)}); top rejections:\n{rej.to_string()}")


def cmd_validate(a):
    from quantai.backtest.robustness import monte_carlo, promotion_gate, sensitivity, walk_forward
    from quantai.strategies.base import registry
    rt = _rt()
    st = rt.settings
    data, notes = _load_history(rt, a.symbols or st.universe, a.tf)
    grids = {
        "trend_pullback": {"rr": [1.5, 2.0, 3.0], "touch_atr": [0.3, 0.6]},
        "compression_breakout": {"rr": [1.5, 2.0, 3.0], "stop_atr": [1.0, 1.5]},
        "range_mean_reversion": {"z_entry": [1.8, 2.2], "max_adx": [18.0, 22.0]},
    }
    report = {"generated_at": utcnow().isoformat(), "timeframe": a.tf, "data": notes, "strategies": {}}
    for name in a.strategies or st.strategies:
        t0 = time.time()
        wf = walk_forward(data, a.tf, name, grids.get(name, {}), get_instruments(), st, n_folds=a.folds)
        rs = [t.r_multiple for t in wf.oos_trades]
        mc = monte_carlo(rs, st.risk.risk_per_trade_pct, st.backtest.monte_carlo_runs, st.backtest.seed)
        last_params = wf.folds[-1].best_params if wf.folds else {}
        sens = sensitivity(data, a.tf, name, last_params, get_instruments(), st)
        gate = promotion_gate(wf, mc, sens, st.promotion)
        s = wf.oos_summary
        spec = registry.create(name, **last_params).spec()
        entry = {"oos_trades": s["trades"], "oos_expectancy_r": s["expectancy_r"], "ci95": s.get("expectancy_r_ci95"),
                 "oos_profit_factor": s["profit_factor"], "oos_win_rate": s["win_rate"], "oos_max_dd_pct": s["max_drawdown_pct"],
                 "mc_p95_dd": mc.max_dd_pct_p95, "mc_median_return_pct": mc.final_return_pct_p50,
                 "risk_of_ruin": mc.prob_dd_exceeds, "folds": [asdict(f) for f in wf.folds],
                 "param_stability": wf.param_stability, "sensitivity": sens.replace({np.nan: None}).to_dict("records"),
                 "promote": gate.promote, "stage": gate.stage, "passed": gate.passed, "failed": gate.failed,
                 "spec": spec, "seconds": round(time.time() - t0, 1)}
        report["strategies"][name] = entry
        rt.journal.append("strategy_version", spec, ref_id=spec["fingerprint"], strategy=name)
        rt.journal.append("promotion", {"strategy": name, "fingerprint": spec["fingerprint"], **asdict(gate)},
                          ref_id=spec["fingerprint"], strategy=name)
        print(f"\n== {name} ==  OOS trades {s['trades']}, expectancy {s['expectancy_r']:.3f}R "
              f"(95% CI {s['expectancy_r_ci95'][0]:.3f}..{s['expectancy_r_ci95'][1]:.3f}), PF {s['profit_factor']:.2f}, "
              f"MC p95 DD {mc.max_dd_pct_p95:.2f}%  ->  {gate.stage.upper()}")
        for f in wf.folds:
            print(f"   test {f.test[0][:10]}..{f.test[1][:10]} params={f.best_params} trades={f.test_trades} E={f.test_expectancy_r:.3f}R")
        if gate.failed:
            print("   failed gates:", ", ".join(gate.failed))
    # aggregate for the target engine (validated strategies only)
    passed = [v for v in report["strategies"].values() if v["promote"]]
    if passed:
        n = sum(v["oos_trades"] for v in passed)
        report["walk_forward"] = {"expectancy_r": sum(v["oos_expectancy_r"] * v["oos_trades"] for v in passed) / n}
    rt.write_json("latest_backtest.json", report)


def cmd_regime_eval(a):
    from quantai.analysis.indicators import add_features
    from quantai.analysis.regime import classify, evaluate
    rt = _rt()
    md = rt.fetch(get_instruments()[a.symbol], a.tf, live=False)
    f = add_features(md.df)
    print(f"{a.symbol} {a.tf}: {len(f)} bars; forward horizon {a.horizon} bars (overlapping windows: t-stats indicative only)")
    print(evaluate(f, classify(f), a.horizon).round(3).to_string())


def cmd_paper(a):
    from quantai.paper import PaperTrader
    rt = _rt()
    st = rt.settings
    pt = PaperTrader(rt, a.symbols or st.universe, _strategies(st.strategies), a.tf or st.primary_timeframe)
    for k in range(a.cycles):
        rep = pt.run_once()
        print(json.dumps({k2: v for k2, v in rep.items() if k2 != "events"}, default=str), flush=True)
        for e in rep["events"]:
            print("  event:", e)
        if k < a.cycles - 1:
            time.sleep(a.interval)


def cmd_killswitch(a):
    rt = _rt()
    if a.action == "on":
        _print(rt.kill.activate(a.reason or "manual", "cli"))
        from quantai.paper import PaperTrader
        pt = PaperTrader(rt, rt.settings.universe, [], rt.settings.primary_timeframe)
        n = pt.state.broker.cancel_all_pending("kill_switch", utcnow())
        pt.save()
        rt.write_json("paper_snapshot.json", pt.snapshot())
        print(f"cancelled {n} pending simulated orders; monitoring of open positions continues")
    elif a.action == "off":
        _print(rt.kill.deactivate(a.reason or "manual", "cli"))
    snap = rt.read_json("paper_snapshot.json") or {}
    _print({"kill_switch": rt.kill.status(), "equity": snap.get("equity"), "open_positions": snap.get("open_positions"),
            "pending": snap.get("pending"), "journal_ok": rt.journal.verify()[0]})


def cmd_report(a):
    from quantai.journal import analyze
    from quantai.targets import daily_target
    rt = _rt()
    st = rt.settings
    snap = rt.read_json("paper_snapshot.json") or {}
    scan = rt.read_json("latest_scan.json") or {}
    bt = rt.read_json("latest_backtest.json") or {}
    eq = snap.get("equity", st.account.starting_capital)
    tgt = daily_target(st.account.starting_capital, eq, st.target.daily_target_pct, st.risk.risk_per_trade_pct,
                       st.risk.max_daily_loss_pct, st.target.max_trades_per_day,
                       (bt.get("walk_forward") or {}).get("expectancy_r"), None, len(scan.get("opportunities", [])))
    rep = {"at": utcnow().isoformat(), "equity": eq, "target": tgt.to_dict(), "journal": analyze(rt.journal),
           "kill_switch": rt.kill.status()}
    rt.journal.append("daily_report", rep)
    _print(rep)


def cmd_journal(a):
    rt = _rt()
    if a.action == "verify":
        ok, bad = rt.journal.verify()
        print("journal integrity:", "OK" if ok else f"BROKEN at seq {bad}")
        sys.exit(0 if ok else 2)
    if a.action == "note":
        print(rt.journal.append("lesson" if a.lesson else "note", {"text": a.text}, symbol=a.symbol))
        return
    for e in reversed(rt.journal.entries(a.kind, a.n)):
        print(e["seq"], e["ts"][:19], e["kind"], e["symbol"] or "", json.dumps(e["payload"], default=str)[:160])


def cmd_serve(a):
    import uvicorn
    from quantai.api.app import create_app
    uvicorn.run(create_app(), host=a.host, port=a.port, log_level="info")


def main(argv: list[str] | None = None) -> None:
    setup_logging("WARNING")
    st = get_settings()
    p = argparse.ArgumentParser(prog="quantai", description="QUANT AI research & paper-trading system")
    sub = p.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("health"); sp.add_argument("--probe", nargs="*", default=[]); sp.set_defaults(f=cmd_health)
    sp = sub.add_parser("fetch"); sp.add_argument("symbol"); sp.add_argument("--tf", default="1d"); sp.add_argument("-n", type=int, default=5); sp.set_defaults(f=cmd_fetch)
    sub.add_parser("macro").set_defaults(f=cmd_macro)
    sub.add_parser("news").set_defaults(f=cmd_news)
    sp = sub.add_parser("calendar"); sp.add_argument("--hours", type=int, default=72); sp.set_defaults(f=cmd_calendar)
    sp = sub.add_parser("scan"); sp.add_argument("--symbols", nargs="*"); sp.add_argument("--tf"); sp.set_defaults(f=cmd_scan)
    for name, fn in (("backtest", cmd_backtest), ("validate", cmd_validate)):
        sp = sub.add_parser(name); sp.add_argument("--symbols", nargs="*"); sp.add_argument("--tf", default=st.primary_timeframe)
        sp.add_argument("--strategies", nargs="*")
        if name == "backtest":
            sp.add_argument("--min-score", type=float, default=None)
        else:
            sp.add_argument("--folds", type=int, default=4)
        sp.set_defaults(f=fn)
    sp = sub.add_parser("regime-eval"); sp.add_argument("symbol"); sp.add_argument("--tf", default="1d"); sp.add_argument("--horizon", type=int, default=10); sp.set_defaults(f=cmd_regime_eval)
    sp = sub.add_parser("paper"); sp.add_argument("--symbols", nargs="*"); sp.add_argument("--tf"); sp.add_argument("--cycles", type=int, default=1)
    sp.add_argument("--interval", type=int, default=900); sp.set_defaults(f=cmd_paper)
    sp = sub.add_parser("killswitch"); sp.add_argument("action", choices=["on", "off", "status"]); sp.add_argument("--reason"); sp.set_defaults(f=cmd_killswitch)
    sub.add_parser("report").set_defaults(f=cmd_report)
    sp = sub.add_parser("journal"); sp.add_argument("action", choices=["list", "verify", "note"]); sp.add_argument("--kind"); sp.add_argument("-n", type=int, default=30)
    sp.add_argument("--text"); sp.add_argument("--symbol"); sp.add_argument("--lesson", action="store_true"); sp.set_defaults(f=cmd_journal)
    sp = sub.add_parser("serve"); sp.add_argument("--host", default="127.0.0.1"); sp.add_argument("--port", type=int, default=8000); sp.set_defaults(f=cmd_serve)
    a = p.parse_args(argv)
    a.f(a)


if __name__ == "__main__":
    main()
