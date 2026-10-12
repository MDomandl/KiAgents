from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from aktien_oop.runner import Runner, _is_first_run
from aktien_oop.store import PortfolioStore


def test_known_empty_snapshot_is_not_a_first_run(tmp_path):
    store = PortfolioStore(tmp_path)
    store.save_positions(
        pd.DataFrame(columns=["as_of", "ticker", "allocation_pct"]),
        snapshot_as_of="2025-10-31",
    )
    empty_snapshot = store.load_positions_snapshot_before("2025-11-01")

    assert not _is_first_run(empty_snapshot)
    assert _is_first_run(None)


def test_cadence_skip_reports_latest_known_empty_snapshot_as_flat(
    tmp_path,
    monkeypatch,
    capsys,
):
    store = PortfolioStore(tmp_path)
    current_as_of = pd.Timestamp.now().normalize()
    previous_as_of = current_as_of - pd.DateOffset(months=1)
    store.save_positions(pd.DataFrame([
        {"as_of": previous_as_of, "ticker": "OLD", "allocation_pct": 100.0},
    ]))
    store.save_positions(
        pd.DataFrame(columns=["as_of", "ticker", "allocation_pct"]),
        snapshot_as_of=current_as_of,
    )

    runner = Runner.__new__(Runner)
    runner.cfg = SimpleNamespace(
        as_of=current_as_of,
        top_k=1,
        buffer_k=1,
        score_days=1,
        vol_days=1,
        use_sector_limits=False,
        max_per_sector=None,
        force_rebalance=False,
        rebalance_frequency="monthly",
        verbose=False,
        lib_debug=False,
        save_dir=tmp_path,
    )
    runner.data = SimpleNamespace()
    runner.engine = SimpleNamespace()
    runner.store = store
    monkeypatch.setattr(
        runner,
        "_normalize_cfg",
        lambda **kwargs: {
            "period": "1d",
            "as_of": current_as_of,
            "score_days": 1,
            "vol_days": 1,
            "use_sector_limits": False,
            "max_per_sector": None,
        },
    )
    monkeypatch.setattr(runner, "_apply_norm_to_cfg", lambda *args, **kwargs: None)

    runner.run()

    output = capsys.readouterr().out
    assert "flat" in output
    assert "OLD" not in output
