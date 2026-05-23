from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd

SCHEMA_VERSION = 1


def default_cache_dir() -> Path:
    override = os.environ.get("AKTIEN_OOP_MARKET_CACHE_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().parent / "data_cache" / "market_data"


def _norm_tickers(tickers: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted(str(t).strip().upper() for t in tickers if str(t).strip()))


def _norm_date(value) -> str | None:
    if value is None or value == "":
        return None
    return pd.Timestamp(value).tz_localize(None).normalize().strftime("%Y-%m-%d")


@dataclass(frozen=True)
class MarketDataRequest:
    data_kind: str
    tickers: tuple[str, ...]
    start: str | None = None
    end: str | None = None
    period: str | None = None
    as_of: str | None = None
    adjusted: bool = True
    benchmark_symbol: str | None = None
    allow_missing_tickers: bool = False

    @classmethod
    def build(cls, *, data_kind: str, tickers: Iterable[str], start=None, end=None,
              period: str | None = None, as_of=None, adjusted: bool = True,
              benchmark_symbol: str | None = None,
              allow_missing_tickers: bool = False) -> "MarketDataRequest":
        benchmark = str(benchmark_symbol).strip().upper() if benchmark_symbol else None
        return cls(
            data_kind=str(data_kind).strip().lower(),
            tickers=_norm_tickers(tickers),
            start=_norm_date(start),
            end=_norm_date(end),
            period=str(period) if period is not None else None,
            as_of=_norm_date(as_of),
            adjusted=bool(adjusted),
            benchmark_symbol=benchmark,
            allow_missing_tickers=bool(allow_missing_tickers),
        )

    def key_payload(self) -> dict:
        universe_hash = hashlib.sha1("|".join(self.tickers).encode("utf-8")).hexdigest()
        return {
            "schema_version": SCHEMA_VERSION,
            "data_kind": self.data_kind,
            "tickers": list(self.tickers),
            "universe_hash": universe_hash,
            "start": self.start,
            "end": self.end,
            "period": self.period,
            "as_of": self.as_of,
            "adjusted": self.adjusted,
            "benchmark_symbol": self.benchmark_symbol,
            "allow_missing_tickers": self.allow_missing_tickers,
        }


def build_cache_key(request: MarketDataRequest) -> str:
    payload = json.dumps(request.key_payload(), sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
    return f"{request.data_kind}_{request.key_payload()['universe_hash'][:12]}_{digest}"


class MarketDataCache:
    def __init__(self, cache_dir: str | Path | None = None):
        self.cache_dir = Path(cache_dir) if cache_dir is not None else default_cache_dir()

    def cache_path(self, request: MarketDataRequest) -> Path:
        return self.cache_dir / f"{build_cache_key(request)}.pkl"

    def get_or_load_frame(
        self,
        request: MarketDataRequest,
        loader: Callable[[], pd.DataFrame | None],
    ) -> pd.DataFrame | None:
        path = self.cache_path(request)
        cached = self._read_valid(path, request)
        if cached is not None:
            self._log("HIT", request, path)
            return cached.copy()

        self._log("MISS", request, path)
        frame = loader()
        ok, reason = self._validate_frame(frame, request)
        if not ok:
            self._warn(f"download result not cached: {reason}; key={build_cache_key(request)}")
            return frame

        self._write(path, request, frame)
        self._log("STORE", request, path)
        return frame.copy()

    def _read_valid(self, path: Path, request: MarketDataRequest) -> pd.DataFrame | None:
        if not path.exists():
            return None
        try:
            with path.open("rb") as fh:
                payload = pickle.load(fh)
            metadata = payload.get("metadata", {})
            frame = payload.get("frame")
        except Exception as exc:
            self._warn(f"invalid file ignored: {path} ({exc})")
            return None

        ok, reason = self._validate_metadata(metadata, request)
        if not ok:
            self._warn(f"metadata mismatch ignored: {reason}; path={path}")
            return None
        ok, reason = self._validate_frame(frame, request)
        if not ok:
            self._warn(f"incomplete/invalid data ignored: {reason}; path={path}")
            return None
        return frame

    def _write(self, path: Path, request: MarketDataRequest, frame: pd.DataFrame) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "metadata": {
                **request.key_payload(),
                "cache_key": build_cache_key(request),
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
            "frame": frame,
        }
        fd, tmp_name = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=str(path.parent))
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "wb") as fh:
                pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
            tmp_path.replace(path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

    def _validate_metadata(self, metadata: dict, request: MarketDataRequest) -> tuple[bool, str]:
        expected = request.key_payload()
        fields = (
            "schema_version", "data_kind", "tickers", "start", "end",
            "period", "as_of", "adjusted", "benchmark_symbol", "allow_missing_tickers",
        )
        for field in fields:
            if metadata.get(field) != expected.get(field):
                return False, f"{field} expected={expected.get(field)!r} got={metadata.get(field)!r}"
        return True, ""

    def _validate_frame(self, frame, request: MarketDataRequest) -> tuple[bool, str]:
        if frame is None or not isinstance(frame, pd.DataFrame):
            return False, "not a DataFrame"
        if frame.empty or frame.dropna(how="all").empty:
            return False, "empty DataFrame"

        columns = _flatten_columns(frame.columns)
        if request.data_kind in {"close", "benchmark"}:
            missing = [ticker for ticker in request.tickers if ticker not in columns]
            if missing:
                message = "missing ticker columns: " + ",".join(missing)
                if request.allow_missing_tickers:
                    self._warn(message)
                else:
                    return False, message
        if request.data_kind == "ohlc" and not {"Close", "High", "Low"}.issubset(columns):
            return False, "missing OHLC columns"

        try:
            idx = pd.to_datetime(frame.index).tz_localize(None)
        except Exception:
            return False, "index is not datetime-like"
        idx = idx[~pd.isna(idx)]
        if len(idx) == 0:
            return False, "no valid dates"
        first = idx.min().normalize()
        last = idx.max().normalize()
        if request.start:
            start = pd.Timestamp(request.start)
            if first > start + pd.Timedelta(days=7):
                return False, f"starts too late: {first.date()} > {start.date()}"
        if request.end:
            end = pd.Timestamp(request.end) - pd.Timedelta(days=1)
            if last < end - pd.Timedelta(days=7):
                return False, f"ends too early: {last.date()} < {end.date()}"
        if request.as_of:
            as_of = pd.Timestamp(request.as_of)
            if last > as_of:
                return False, f"contains dates after as_of: {last.date()} > {as_of.date()}"
            if last < as_of - pd.Timedelta(days=7):
                return False, f"does not cover as_of window: {last.date()} < {as_of.date()}"
        return True, ""

    @staticmethod
    def _log(event: str, request: MarketDataRequest, path: Path) -> None:
        msg = (
            f"[market-cache] {event} kind={request.data_kind} "
            f"tickers={len(request.tickers)} key={build_cache_key(request)} path={path}"
        )
        logging.info(msg)
        print(msg, flush=True)

    @staticmethod
    def _warn(message: str) -> None:
        logging.warning("[market-cache] %s", message)
        print(f"[market-cache] WARN {message}", flush=True)


def _flatten_columns(columns) -> set[str]:
    values: set[str] = set()
    if isinstance(columns, pd.MultiIndex):
        for entry in columns.to_flat_index():
            parts = entry if isinstance(entry, tuple) else (entry,)
            for part in parts:
                if str(part).strip():
                    values.add(str(part).strip())
                    values.add(str(part).strip().upper())
    else:
        for column in columns:
            values.add(str(column).strip())
            values.add(str(column).strip().upper())
    return values


def get_default_market_data_cache() -> MarketDataCache:
    return MarketDataCache()
