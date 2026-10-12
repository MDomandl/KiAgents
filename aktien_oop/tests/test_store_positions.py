from __future__ import annotations

import pandas as pd
import pytest

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


def test_save_positions_replaces_snapshot_with_empty_snapshot_for_same_as_of(tmp_path):
    store = PortfolioStore(tmp_path)

    store.save_positions(pd.DataFrame([
        {"as_of": "2025-09-30", "ticker": "AAA", "allocation_pct": 100.0},
        {"as_of": "2025-10-31", "ticker": "BBB", "allocation_pct": 100.0},
    ]))

    empty_snapshot = pd.DataFrame(columns=["as_of", "ticker", "allocation_pct"])
    store.save_positions(empty_snapshot, snapshot_as_of="2025-09-30")

    saved = pd.read_csv(store.positions_path)
    assert not saved["as_of"].astype(str).str.startswith("2025-09-30").any()
    assert saved["as_of"].astype(str).str.startswith("2025-10-31").all()
    assert saved["ticker"].tolist() == ["BBB"]

    snapshots = pd.read_csv(store.snapshots_path)
    assert snapshots["as_of"].astype(str).str.startswith("2025-09-30").any()


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


def test_load_positions_snapshot_before_distinguishes_missing_empty_and_non_empty(tmp_path):
    store = PortfolioStore(tmp_path)

    assert store.load_positions_snapshot_before("2025-09-01") is None

    store.save_positions(
        pd.DataFrame(columns=["as_of", "ticker", "allocation_pct"]),
        snapshot_as_of="2025-09-30",
    )
    empty = store.load_positions_snapshot_before("2025-10-01")

    assert empty is not None
    assert empty.as_of == pd.Timestamp("2025-09-30")
    assert empty.positions.empty

    store.save_positions(pd.DataFrame([
        {"as_of": "2025-10-31", "ticker": "AAA", "allocation_pct": 100.0},
    ]))
    populated = store.load_positions_snapshot_before("2025-11-01")

    assert populated is not None
    assert populated.as_of == pd.Timestamp("2025-10-31")
    assert populated.positions["ticker"].tolist() == ["AAA"]


def test_load_positions_before_does_not_fall_back_past_empty_snapshot(tmp_path):
    store = PortfolioStore(tmp_path)
    store.save_positions(pd.DataFrame([
        {"as_of": "2025-09-30", "ticker": "OLD", "allocation_pct": 100.0},
    ]))
    store.save_positions(
        pd.DataFrame(columns=["as_of", "ticker", "allocation_pct"]),
        snapshot_as_of="2025-10-31",
    )

    prev = store.load_positions_before("2025-11-15")

    assert prev is not None
    assert prev.empty


@pytest.mark.parametrize("interrupted_source", ["positions", "snapshots"])
def test_interrupted_empty_snapshot_write_remains_authoritatively_flat(
    tmp_path,
    monkeypatch,
    interrupted_source,
):
    store = PortfolioStore(tmp_path)
    store.save_positions(pd.DataFrame([
        {"as_of": "2025-09-30", "ticker": "OLD", "allocation_pct": 100.0},
    ]))
    replace_file = store._replace_file

    def interrupt_during_commit(source, target):
        interrupted_path = (
            store.positions_pending_path
            if interrupted_source == "positions"
            else store.snapshots_pending_path
        )
        if source == interrupted_path:
            raise OSError("simulated interruption")
        replace_file(source, target)

    monkeypatch.setattr(store, "_replace_file", interrupt_during_commit)

    with pytest.raises(OSError, match="simulated interruption"):
        store.save_positions(
            pd.DataFrame(columns=["as_of", "ticker", "allocation_pct"]),
            snapshot_as_of="2025-10-31",
        )

    prev = store.load_positions_before("2025-11-15")

    assert store.snapshots_pending_path.exists()
    assert prev is not None
    assert prev.empty

    monkeypatch.setattr(store, "_replace_file", replace_file)
    store.save_positions(pd.DataFrame([
        {"as_of": "2025-11-30", "ticker": "NEW", "allocation_pct": 100.0},
    ]))

    recovered_empty = store.load_positions_before("2025-11-15")
    recovered_non_empty = store.load_positions_before("2025-12-15")
    assert not store.snapshots_pending_path.exists()
    assert recovered_empty is not None and recovered_empty.empty
    assert recovered_non_empty is not None
    assert recovered_non_empty["ticker"].tolist() == ["NEW"]


def test_snapshot_history_supports_non_empty_empty_non_empty_transition(tmp_path):
    store = PortfolioStore(tmp_path)
    store.save_positions(pd.DataFrame([
        {"as_of": "2025-09-30", "ticker": "OLD", "allocation_pct": 100.0},
    ]))
    store.save_positions(
        pd.DataFrame(columns=["as_of", "ticker", "allocation_pct"]),
        snapshot_as_of="2025-10-31",
    )
    store.save_positions(pd.DataFrame([
        {"as_of": "2025-11-30", "ticker": "NEW", "allocation_pct": 100.0},
    ]))

    september = store.load_positions_before("2025-10-15")
    october = store.load_positions_before("2025-11-15")
    november = store.load_positions_before("2025-12-15")

    assert september is not None and september["ticker"].tolist() == ["OLD"]
    assert october is not None and october.empty
    assert november is not None and november["ticker"].tolist() == ["NEW"]


def test_load_positions_before_supports_legacy_positions_csv_without_snapshot_index(tmp_path):
    store = PortfolioStore(tmp_path)
    pd.DataFrame([
        {"as_of": "2025-09-30", "ticker": "AAA", "allocation_pct": 100.0},
    ]).to_csv(store.positions_path, index=False)

    prev = store.load_positions_before("2025-10-01")

    assert not store.snapshots_path.exists()
    assert prev is not None
    assert prev["ticker"].tolist() == ["AAA"]
