# models.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

@dataclass
class TickerSignal:
    ticker: str
    score_lin: float
    mom_12_1: Optional[float]    # kann None/NaN sein, wenn 12-1 nicht berechenbar
    slope: float
    r2: float
    volatility: Optional[float]  # annualisierte Volatilität
    stop_loss_pct: Optional[float]
    # optional – nur falls du es eingebaut hast:
    sma100_val: Optional[float] = None
    sma100_dist_pct: Optional[float] = None


# Optional – nur falls du irgendwo ein stark typisiertes Portfolio-Objekt brauchst:
@dataclass
class PortfolioPosition:
    as_of: str
    ticker: str
    rank: int
    score: float
    volatility: Optional[float]
    stop_loss_pct: Optional[float]
    allocation_pct: Optional[float] = None
