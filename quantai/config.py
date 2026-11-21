"""Configuration loading: YAML files for parameters, environment variables for secrets."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = Path(os.environ.get("QUANTAI_CONFIG_DIR", ROOT / "config"))


class AccountCfg(BaseModel):
    currency: str = "USD"
    starting_capital: float = 100_000.0


class RiskCfg(BaseModel):
    risk_per_trade_pct: float = 0.25
    max_risk_per_trade_pct: float = 0.50
    absolute_ceiling_pct: float = 1.0
    exceptional_justification: str = ""
    max_daily_loss_pct: float = 1.0
    max_daily_loss_ceiling_pct: float = 2.0
    max_weekly_loss_pct: float = 2.5
    max_drawdown_pct: float = 8.0
    max_consecutive_losses: int = 4
    consecutive_loss_cooldown_hours: int = 72   # trading pause after the streak; counter resets after
    max_open_positions: int = 4
    max_risk_per_group_pct: float = 0.75
    max_total_open_risk_pct: float = 1.5
    max_leverage: float = 5.0
    min_stop_atr: float = 0.3
    max_stop_atr: float = 6.0
    max_spread_to_atr: float = 0.25
    event_blackout_minutes: int = 30

    @model_validator(mode="after")
    def _check(self) -> "RiskCfg":
        if self.risk_per_trade_pct <= 0:
            raise ValueError("risk_per_trade_pct must be > 0")
        if self.absolute_ceiling_pct > 1.0:
            raise ValueError("absolute_ceiling_pct may not exceed 1% of capital")
        if self.risk_per_trade_pct > self.absolute_ceiling_pct:
            raise ValueError("risk_per_trade_pct exceeds the absolute ceiling")
        if self.risk_per_trade_pct > self.max_risk_per_trade_pct and not self.exceptional_justification.strip():
            raise ValueError("risk above max_risk_per_trade_pct requires exceptional_justification")
        if self.max_daily_loss_pct > self.max_daily_loss_ceiling_pct or self.max_daily_loss_ceiling_pct > 2.0:
            raise ValueError("max_daily_loss_pct must be <= ceiling, and ceiling <= 2%")
        return self


class TargetCfg(BaseModel):
    daily_target_pct: float = 0.30
    max_trades_per_day: int = 3


class RankingCfg(BaseModel):
    thresholds: dict[str, float] = Field(
        default_factory=lambda: {"exceptional": 90, "high": 80, "watchlist": 70, "insufficient": 60})
    min_score_to_trade: float = 80


class DataCfg(BaseModel):
    stale_multiplier: float = 3.0
    jump_sigma: float = 8.0
    max_missing_ratio: float = 0.05
    provider_discrepancy_pct: float = 0.5


class ExecutionCfg(BaseModel):
    slippage_atr_fraction: float = 0.05
    same_bar_sl_tp_policy: Literal["stop_first"] = "stop_first"
    partial_exit_r: float = 1.0
    partial_exit_fraction: float = 0.5


class BacktestCfg(BaseModel):
    annualization_days: int = 252
    monte_carlo_runs: int = 2000
    seed: int = 7


class PromotionCfg(BaseModel):
    min_oos_trades: int = 60
    min_oos_profit_factor: float = 1.15
    min_oos_expectancy_r: float = 0.05
    require_bootstrap_lower_bound_positive: bool = True
    max_mc_p95_drawdown_pct: float = 15.0
    max_sensitivity_degradation: float = 0.6
    min_paper_trades: int = 30
    min_paper_days: int = 30


class AlertsCfg(BaseModel):
    cooldown_seconds: int = 900
    enabled_categories: list[str] = Field(default_factory=list)


class FeedCfg(BaseModel):
    name: str
    url: str
    credibility: float = 0.7


class NewsCfg(BaseModel):
    feeds: list[FeedCfg] = Field(default_factory=list)


class Settings(BaseModel):
    account: AccountCfg = AccountCfg()
    mode: Literal["research", "paper"] = "research"
    risk: RiskCfg = RiskCfg()
    target: TargetCfg = TargetCfg()
    ranking: RankingCfg = RankingCfg()
    data: DataCfg = DataCfg()
    execution: ExecutionCfg = ExecutionCfg()
    backtest: BacktestCfg = BacktestCfg()
    promotion: PromotionCfg = PromotionCfg()
    alerts: AlertsCfg = AlertsCfg()
    news: NewsCfg = NewsCfg()
    universe: list[str] = Field(default_factory=lambda: ["EURUSD", "GBPUSD", "USDJPY", "SPX", "NDX", "GOLD", "BTCUSD", "ETHUSD"])
    primary_timeframe: Literal["1d", "4h", "1h", "15m", "5m"] = "1d"
    strategies: list[str] = Field(default_factory=lambda: ["trend_pullback", "compression_breakout", "range_mean_reversion"])

    @property
    def data_dir(self) -> Path:
        p = Path(os.environ.get("QUANTAI_DATA_DIR", ROOT / "var"))
        p.mkdir(parents=True, exist_ok=True)
        return p


class InstrumentSpec(BaseModel):
    symbol: str
    asset_class: Literal["forex", "index", "commodity", "crypto", "stock"]
    provider: str
    provider_symbol: str
    quote_currency: str
    contract_size: float = 1.0
    qty_step: float
    min_qty: float
    spread: float | None = None          # absolute price units
    spread_bps: float | None = None      # alternative: basis points of price
    commission_bps: float = 0.0
    risk_group: str = "default"
    annualization_days: int = 252
    spec_source: str = "assumption"
    holding_cost_bps_per_day: float = 0.0  # swap / CFD financing / funding, charged per UTC day held
    daily_from_intraday: bool = False    # build 1d bars from 1h (used where vendor daily bars fail validation)

    def spread_at(self, price: float) -> float:
        if self.spread is not None:
            return self.spread
        if self.spread_bps is not None:
            return price * self.spread_bps / 10_000
        raise ValueError(f"{self.symbol}: no spread assumption configured")


class Secrets:
    """Read-only accessor for secrets from the environment. Never logged, never persisted."""

    @staticmethod
    def get(name: str) -> str | None:
        v = os.environ.get(f"QUANTAI_{name}")
        return v if v else None

    @staticmethod
    def live_trading_authorized() -> bool:
        return os.environ.get("QUANTAI_LIVE_TRADING_AUTHORIZED", "no").strip().lower() == "yes"


def _load_yaml(name: str) -> dict:
    path = CONFIG_DIR / name
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(**_load_yaml("settings.yaml"))


@lru_cache(maxsize=1)
def get_instruments() -> dict[str, InstrumentSpec]:
    raw = _load_yaml("instruments.yaml").get("instruments", {})
    return {sym: InstrumentSpec(symbol=sym, **spec) for sym, spec in raw.items()}
