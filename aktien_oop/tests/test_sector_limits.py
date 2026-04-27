from __future__ import annotations

import sys
from argparse import Namespace

import pandas as pd

from aktien_oop.backtest import _build_cfg_from_config_and_cli
from aktien_oop.config import Config
from aktien_oop.core_calc import CalcParams, select_topk_buffer


def _calc_params(use_sector_limits: bool, *, tag: str) -> CalcParams:
    return CalcParams(
        as_of="2025-01-31",
        period="800d",
        adjusted=True,
        score_days=200,
        vol_days=63,
        use_sector_limits=use_sector_limits,
        max_per_sector=1,
        top_k=3,
        buffer_k=0,
        dump_tag=tag,
    )


def _sample_selection(params: CalcParams) -> list[str]:
    scores = pd.Series(
        {
            "AAA": 10.0,
            "AAB": 9.0,
            "AAC": 8.0,
            "BBA": 7.0,
        }
    )
    sectors = {
        "AAA": "TECH",
        "AAB": "TECH",
        "AAC": "TECH",
        "BBA": "HEALTH",
    }
    return list(
        select_topk_buffer(
            scores,
            scores.index,
            sectors,
            params,
            prev_holdings=[],
        )
    )


def test_sector_limits_enabled_enforces_max_per_sector_for_bt_and_run():
    bt_selection = _sample_selection(_calc_params(True, tag="BT"))
    run_selection = _sample_selection(_calc_params(True, tag="RUN"))

    assert bt_selection == ["AAA", "BBA"]
    assert run_selection == bt_selection


def test_sector_limits_disabled_allows_exceeding_sector_cap_for_bt_and_run():
    bt_selection = _sample_selection(_calc_params(False, tag="BT"))
    run_selection = _sample_selection(_calc_params(False, tag="RUN"))

    assert bt_selection == ["AAA", "AAB", "AAC"]
    assert run_selection == bt_selection


def test_backtest_config_uses_explicit_use_sector_limits_false(tmp_path):
    config_path = tmp_path / "backtest_config.toml"
    config_path.write_text(
        """
top_k = 3
buffer_k = 0

[limits]
use_sector_limits = false
max_per_sector = 1
""".strip(),
        encoding="utf-8",
    )

    args = Namespace(
        config=str(config_path),
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

    cfg = _build_cfg_from_config_and_cli(args)

    assert cfg.use_sector_limits is False
    assert cfg.max_per_sector == 1


def test_runner_config_uses_limits_section_for_sector_limits(tmp_path):
    config_path = tmp_path / "runner_config.toml"
    config_path.write_text(
        """
top_k = 3
buffer_k = 0

[limits]
use_sector_limits = false
max_per_sector = 1
""".strip(),
        encoding="utf-8",
    )

    old_argv = sys.argv[:]
    try:
        sys.argv = ["runner", "--config", str(config_path)]
        cfg = Config.from_cli()
    finally:
        sys.argv = old_argv

    assert cfg.use_sector_limits is False
    assert cfg.max_per_sector == 1
