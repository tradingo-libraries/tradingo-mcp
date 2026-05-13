"""Cached ArcticDB client and research-namespace helpers."""

from __future__ import annotations

import os
import re

import arcticdb as adb
import pandas as pd

_ARCTIC: adb.Arctic | None = None

RESEARCH_PREFIX = "research."


def arctic_client() -> adb.Arctic:
    global _ARCTIC
    if _ARCTIC is None:
        _ARCTIC = adb.Arctic(os.environ["TP_ARCTIC_URI"])
    return _ARCTIC


def research_symbols(run_id: str) -> dict[str, list[str]]:
    """Return {library: [symbols]} for a given run_id across all libraries."""
    ac = arctic_client()
    pattern = re.compile(rf"^{re.escape(RESEARCH_PREFIX)}{re.escape(run_id)}\.")
    result: dict[str, list[str]] = {}
    for lib_name in ac.list_libraries():
        lib = ac.get_library(lib_name)
        matches = [s for s in lib.list_symbols() if pattern.match(s)]
        if matches:
            result[lib_name] = matches
    return result


def delete_research_symbols(run_id: str) -> int:
    """Delete all ArcticDB symbols for run_id. Returns count deleted."""
    ac = arctic_client()
    pattern = re.compile(rf"^{re.escape(RESEARCH_PREFIX)}{re.escape(run_id)}\.")
    deleted = 0
    for lib_name in ac.list_libraries():
        lib = ac.get_library(lib_name)
        for sym in lib.list_symbols():
            if pattern.match(sym):
                lib.delete(sym)
                deleted += 1
    return deleted


def list_symbols(library: str, regex: str = "") -> list[str]:
    ac = arctic_client()
    if library not in ac.list_libraries():
        return []
    lib = ac.get_library(library)
    return lib.list_symbols(regex=regex) if regex else lib.list_symbols()


def describe_symbol(library: str, symbol: str) -> dict:
    ac = arctic_client()
    lib = ac.get_library(library)
    desc = lib.get_description(symbol)
    return {
        "library": library,
        "symbol": symbol,
        "date_range": [
            str(desc.date_range[0]) if desc.date_range else None,
            str(desc.date_range[1]) if desc.date_range else None,
        ],
        "row_count": desc.row_count,
        "columns": [c.name for c in desc.columns] if desc.columns else [],
        "last_update": str(desc.last_update_time),
    }


def fetch_bars(
    library: str,
    symbol: str,
    start: str,
    end: str,
    columns: list[str] | None = None,
    max_rows: int = 10_000,
) -> tuple[pd.DataFrame, bool]:
    """Read a time slice. Returns (df, resampled) where resampled=True if downsampled."""
    ac = arctic_client()
    lib = ac.get_library(library)
    date_range = (pd.Timestamp(start), pd.Timestamp(end))
    item = lib.read(symbol, date_range=date_range, columns=columns, lazy=False)
    assert isinstance(item, adb.VersionedItem)
    df = item.data
    resampled = False
    if len(df) > max_rows:
        step = max(1, len(df) // max_rows)
        df = df.iloc[::step]
        resampled = True
    return df, resampled
