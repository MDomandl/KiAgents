from __future__ import annotations

import pandas as pd

from aktien_oop.store import PortfolioStore


def test_save_positions_replaces_complete_snapshot_for_same_as_of(tmp_path):
    store = PortfolioStore(tmp_path)

    store.save_positions(pd.DataFrame([
        {"as_of": "2025-09-30", "ticker": "AAA", "allocation_pct": 50.0},
        {"as_of": "2025-09-30", "ticker": "BBB", "allocation_pct": 50.0},
        {"as_of": "2025-08-29", "ticker": "OLD", "allocation_pct": 100.0},
    ]))

    store.save_positions(pd.DataFrame([
        {"as_of": "2025-09-30", "ticker": "AAA", "allocation_pct": 100.0},
    ]))

    saved = pd.read_csv(store.positions_path)
    sept = saved[saved["as_of"].astype(str).str.startswith("2025-09-30")]
    aug = saved[saved["as_of"].astype(str).str.startswith("2025-08-29")]

    assert sept["ticker"].tolist() == ["AAA"]
    assert float(sept.iloc[0]["allocation_pct"]) == 100.0
    assert aug["ticker"].tolist() == ["OLD"]


def test_load_positions_before_uses_latest_replaced_snapshot(tmp_path):
    store = PortfolioStore(tmp_path)

    store.save_positions(pd.DataFrame([
        {"as_of": "2025-09-30", "ticker": "AAA", "allocation_pct": 50.0},
        {"as_of": "2025-09-30", "ticker": "STALE", "allocation_pct": 50.0},
    ]))
    store.save_positions(pd.DataFrame([
        {"as_of": "2025-09-30", "ticker": "AAA", "allocation_pct": 100.0},
    ]))

    prev = store.load_positions_before("2025-10-08")

    assert prev is not None
    assert prev["ticker"].tolist() == ["AAA"]
