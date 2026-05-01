from __future__ import annotations

import sys
from argparse import Namespace

import pandas as pd

from aktien_oop.backtest import _build_cfg_from_config_and_cli
from aktien_oop.config import Config
from aktien_oop.core_calc import CalcParams, apply_filters


def _bt_args(config_path: str) -> Namespace:
    return Namespace(
        config=config_path,
        tickers=None,
        sector_meta=None,
        save_dir=None,
        start=None,
        end=None,
        frequency=None,
        as_of=None,
        top_k=None,
        buffer_k=None,
        use_sector_limits=None,
        max_per_sector=None,
        gap_filter=None,
        cost_bps=None,
        slippage_bps=None,
        min_history_days=None,
        use_equal_weight=None,
        friction_eps=None,
        friction_eps_pct=None,
        weight_round_step=None,
        max_turnover_cap=None,
        rebalance_every_n=None,
        benchmark=None,
        benchmark_ticker=None,
        dual_benchmark=None,
        benchmark2=None,
        dump_decisions=None,
        dump_selection=None,
        dump_weights=None,
        decisions_dir=None,
        min_position_weight=None,
        max_active_names=None,
        score_days=None,
        vol_days=None,
        regime_exposure_low=None,
        vol_target_ann=None,
        vol_lookback_days=None,
        include_cash=None,
        cash_yield_annual=None,
        verbose=None,
    )


def test_backtest_config_reads_gap_filter_from_limits(tmp_path):
    config_path = tmp_path / "backtest_config.toml"
    config_path.write_text(
        """
[limits]
gap_filter = 0.15
""".strip(),
        encoding="utf-8",
    )

    cfg = _build_cfg_from_config_and_cli(_bt_args(str(config_path)))

    assert cfg.gap_filter == 0.15


def test_runner_config_reads_gap_filter_from_limits(tmp_path):
    config_path = tmp_path / "runner_config.toml"
    config_path.write_text(
        """
[limits]
gap_filter = 0.15
""".strip(),
        encoding="utf-8",
    )

    old_argv = sys.argv[:]
    try:
        sys.argv = ["runner", "--config", str(config_path)]
        cfg = Config.from_cli()
    finally:
        sys.argv = old_argv

    assert cfg.gap_filter == 0.15


def test_core_gap_filter_disabled_keeps_gap_ticker():
    scores = pd.Series({"AAA": 1.0, "BBB": 0.9})
    prices = pd.DataFrame(
        {"AAA": [100.0, 125.0], "BBB": [100.0, 101.0]},
        index=pd.to_datetime(["2025-01-30", "2025-01-31"]),
    )
    params = CalcParams(
        as_of="2025-01-31",
        period="10d",
        adjusted=True,
        score_days=2,
        vol_days=2,
        gap_filter=0.0,
    )

    keep = apply_filters(scores, prices, params)

    assert list(keep) == ["AAA", "BBB"]


def test_core_gap_filter_enabled_removes_gap_ticker():
    scores = pd.Series({"AAA": 1.0, "BBB": 0.9})
    prices = pd.DataFrame(
        {"AAA": [100.0, 125.0], "BBB": [100.0, 101.0]},
        index=pd.to_datetime(["2025-01-30", "2025-01-31"]),
    )
    params = CalcParams(
        as_of="2025-01-31",
        period="10d",
        adjusted=True,
        score_days=2,
        vol_days=2,
        gap_filter=0.10,
    )

    keep = apply_filters(scores, prices, params)

    assert list(keep) == ["BBB"]
