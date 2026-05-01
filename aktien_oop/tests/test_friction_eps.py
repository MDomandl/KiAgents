from __future__ import annotations

import sys
from argparse import Namespace

from aktien_oop.backtest import _build_cfg_from_config_and_cli
from aktien_oop.config import Config
from aktien_oop.core_calc import CalcParams, _apply_friction_eps


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


def test_backtest_config_reads_friction_eps_from_limits(tmp_path):
    config_path = tmp_path / "backtest_config.toml"
    config_path.write_text(
        """
[limits]
friction_eps = 0.02
""".strip(),
        encoding="utf-8",
    )

    cfg = _build_cfg_from_config_and_cli(_bt_args(str(config_path)))

    assert cfg.friction_eps == 0.02


def test_runner_config_reads_friction_eps_from_limits(tmp_path):
    config_path = tmp_path / "runner_config.toml"
    config_path.write_text(
        """
[limits]
friction_eps = 0.02
""".strip(),
        encoding="utf-8",
    )

    old_argv = sys.argv[:]
    try:
        sys.argv = ["runner", "--config", str(config_path)]
        cfg = Config.from_cli()
    finally:
        sys.argv = old_argv

    assert cfg.friction_eps == 0.02


def test_universe_section_configures_alternative_tickers_file(tmp_path):
    tickers_path = tmp_path / "custom_tickers.txt"
    meta_path = tmp_path / "custom_meta.csv"
    config_path = tmp_path / "config.toml"
    tickers_path.write_text("AAA\nBBB\n", encoding="utf-8")
    meta_path.write_text("ticker,sector\nAAA,Tech\nBBB,Health\n", encoding="utf-8")
    config_path.write_text(
        f"""
[universe]
name = "custom"
tickers_file = "{tickers_path.as_posix()}"
meta_file = "{meta_path.as_posix()}"
""".strip(),
        encoding="utf-8",
    )

    bt_cfg = _build_cfg_from_config_and_cli(_bt_args(str(config_path)))

    old_argv = sys.argv[:]
    try:
        sys.argv = ["runner", "--config", str(config_path)]
        runner_cfg = Config.from_cli()
    finally:
        sys.argv = old_argv

    assert bt_cfg.universe_name == "custom"
    assert bt_cfg.tickers_file == tickers_path.as_posix()
    assert bt_cfg.sector_meta == meta_path.as_posix()
    assert runner_cfg.universe.name == "custom"
    assert runner_cfg.universe.tickers_file == tickers_path.resolve()
    assert runner_cfg.universe.meta_file == meta_path.resolve()


def test_friction_eps_zero_has_no_suppression():
    params = CalcParams(
        as_of="2025-01-31",
        period="10d",
        adjusted=True,
        score_days=2,
        vol_days=2,
        friction_eps=0.0,
    )

    result = _apply_friction_eps(
        {"AAA": 0.5, "BBB": 0.5},
        {"AAA": 0.49, "BBB": 0.51},
        params,
    )

    assert result == {"AAA": 0.5, "BBB": 0.5}


def test_friction_eps_positive_suppresses_small_changes_symmetrically():
    bt_params = CalcParams(
        as_of="2025-01-31",
        period="10d",
        adjusted=True,
        score_days=2,
        vol_days=2,
        friction_eps=0.02,
        dump_tag="BT",
    )
    run_params = CalcParams(
        as_of="2025-01-31",
        period="10d",
        adjusted=True,
        score_days=2,
        vol_days=2,
        friction_eps=0.02,
        dump_tag="RUN",
    )
    prev = {"AAA": 0.49, "BBB": 0.51}
    target = {"AAA": 0.5, "BBB": 0.5}

    bt_result = _apply_friction_eps(target, prev, bt_params)
    run_result = _apply_friction_eps(target, prev, run_params)

    assert bt_result == {"AAA": 0.49, "BBB": 0.51}
    assert run_result == bt_result


def test_runner_config_reads_dump_friction_debug_flag(tmp_path):
    config_path = tmp_path / "runner_config.toml"
    config_path.write_text(
        'dump_friction_debug = true\n',
        encoding="utf-8",
    )

    old_argv = sys.argv[:]
    try:
        sys.argv = ["runner", "--config", str(config_path)]
        cfg = Config.from_cli()
    finally:
        sys.argv = old_argv

    assert cfg.dump_friction_debug is True


def test_runner_friction_debug_dump_contains_intermediate_fields(tmp_path):
    from pathlib import Path
    from aktien_oop.runner import _write_runner_friction_debug

    old_cwd = Path.cwd()
    try:
        import os
        os.chdir(tmp_path)
        dumps_dir = Path("aktien_oop/dumps")
        debug_dir = Path("aktien_oop/debug")
        dumps_dir.mkdir(parents=True)
        debug_dir.mkdir(parents=True)

        (dumps_dir / "weights_RUN_2025-10-08.csv").write_text(
            "\n".join([
                "ticker,weight_raw,weight_after_round,weight_final,cash_weight",
                "AVGO,0.08333333333333333,0.08333333333333333,0.11111111111111113,0.0",
                "CVS,0.08333333333333333,0.08333333333333333,0.11111111111111113,0.0",
            ]) + "\n",
            encoding="utf-8",
        )

        _write_runner_friction_debug(
            debug_dir=debug_dir,
            as_of_str="2025-10-08",
            prev_weights={"AVGO": 0.1111111111111111},
            final_weights={"AVGO": 0.11111111111111113, "CVS": 0.11111111111111113},
            friction_eps=0.0015,
        )

        content = (debug_dir / "RUN_friction_2025-10-08.csv").read_text(encoding="utf-8")
        assert "target_weight_before_friction" in content
        assert "weight_after_friction" in content
        assert "friction_action" in content
        assert "enter_applied_exact" in content
    finally:
        os.chdir(old_cwd)
