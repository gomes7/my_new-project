"""Market reaction analysis: what price actually did around an event (vs the headline)."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime

import numpy as np
import pandas as pd


@dataclass
class Reaction:
    symbol: str
    event_time: str
    pre_vol: float
    ret_post: float
    z_move: float
    range_expansion: float
    volume_ratio: float | None
    reaction: str              # up | down | none
    significant: bool
    agrees_with_hypothesis: bool | None

    def to_dict(self) -> dict:
        return asdict(self)


def measure_reaction(df: pd.DataFrame, symbol: str, event_time: datetime, hypothesis_sign: int | None = None,
                     pre_bars: int = 48, post_bars: int = 4, z_threshold: float = 2.0) -> Reaction | None:
    """Compare the post-event move to pre-event volatility. `hypothesis_sign`: +1 expected up, -1 down."""
    t = pd.Timestamp(event_time)
    pre = df[df.index < t].tail(pre_bars)
    post = df[df.index >= t].head(post_bars)
    if len(pre) < 10 or post.empty:
        return None
    r = np.log(pre["close"]).diff().dropna()
    vol = float(r.std())
    ret = float(np.log(post["close"].iloc[-1] / pre["close"].iloc[-1]))
    z = ret / (vol * np.sqrt(len(post))) if vol > 0 else 0.0
    rng_pre = float((pre["high"] - pre["low"]).mean())
    rng_post = float((post["high"] - post["low"]).mean())
    vr = None
    if pre["volume"].sum() > 0:
        vr = float(post["volume"].mean() / max(pre["volume"].mean(), 1e-12))
    sig = abs(z) >= z_threshold
    direction = "up" if sig and z > 0 else "down" if sig and z < 0 else "none"
    agrees = None if hypothesis_sign is None or not sig else (np.sign(z) == hypothesis_sign)
    return Reaction(symbol, t.isoformat(), vol, ret, float(z), rng_post / rng_pre if rng_pre > 0 else 0.0, vr,
                    direction, sig, None if agrees is None else bool(agrees))
