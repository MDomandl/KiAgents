from __future__ import annotations

import pandas as pd

from aktien_oop.runner import _is_first_run
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
