"""The literature_* tools are thin wrappers over research_catalog.tools; check wiring only."""

from __future__ import annotations

import asyncio

import pytest

from tradingo_mcp import server

TOOLS = {"literature_search", "literature_paper", "literature_stats"}


def test_tools_registered():
    names = {t.name for t in asyncio.run(server.mcp.list_tools())}
    assert TOOLS <= names


def test_search_passes_filters_through(monkeypatch):
    seen = {}

    def fake(query, **kw):
        seen.update(query=query, **kw)
        return {"total": 0, "items": []}

    monkeypatch.setattr(server.literature, "literature_search", fake)
    out = server.literature_search(
        "trend", mechanism=["momentum"], date_from="2020-01-01", limit=5
    )
    assert out == {"total": 0, "items": []}
    assert seen["query"] == "trend"
    assert seen["mechanism"] == ["momentum"]
    assert seen["date_from"] == "2020-01-01"
    assert seen["limit"] == 5
    assert seen["full_text_only"] is False


def test_paper_and_stats_delegate(monkeypatch):
    monkeypatch.setattr(server.literature, "literature_paper", lambda ref: {"ref": ref})
    monkeypatch.setattr(server.literature, "literature_stats", lambda: {"papers": 1})
    assert server.literature_paper("doi:10.1/x") == {"ref": "doi:10.1/x"}
    assert server.literature_stats() == {"papers": 1}


def test_bad_input_surfaces_as_error():
    # No RC_DB_URL / DB in the test env: a bad date is rejected before any DB access.
    with pytest.raises(ValueError, match="date_from"):
        server.literature_search("x", date_from="not-a-date")
