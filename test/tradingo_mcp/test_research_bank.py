"""Tests for tradingo_mcp.research_bank (read-only bank queries).

``data/bank.sqlite`` is built by tradingo-plat's indexer from the synthetic bank in
``tradingo-plat/tests/test_research_bank.py::make_bank_root``. Regenerate it there if the schema changes.
"""

from __future__ import annotations

import pathlib

import pytest

from tradingo_mcp import research_bank as rb

FIXTURE = pathlib.Path(__file__).parent / "data" / "bank.sqlite"
R1, R2 = "20261001T100000", "20261008T100000"
CB, US = f"{R1}/cb-shock-5m", f"{R2}/us-0830-shock-5m"


@pytest.fixture(autouse=True)
def _bank(monkeypatch):
    monkeypatch.setenv("TP_RESEARCH_BANK", str(FIXTURE))


def test_missing_bank(monkeypatch, tmp_path):
    monkeypatch.setenv("TP_RESEARCH_BANK", str(tmp_path / "nope.sqlite"))
    with pytest.raises(rb.BankUnavailable):
        rb.search("x")


def test_search_text_and_filters():
    assert [r["key"] for r in rb.search("dies")] == [CB]
    keys = {r["key"] for r in rb.search(about=["cause:below-cost-floor"])}
    assert keys == {CB, US}
    assert [r["key"] for r in rb.search(about=["EURUSD"])] == [
        CB
    ]  # bare value resolves
    assert [r["key"] for r in rb.search(decision="continue")] == [CB]
    # All-terms match fails, falls back to any-term.
    assert rb.search("dies zebra")[0]["key"] == CB


def test_similar():
    hits = rb.similar("central bank decision shock continuation 5m")
    assert hits[0]["key"] == CB
    assert all(h["key"] != CB for h in rb.similar("shock continuation", exclude=CB))


def test_get_item():
    item = rb.get_item(CB)
    assert item["metrics_quality"] == "v1" and "dies" in item["findings"]
    assert {"type": "continued_from", "src": f"item:{US}"} in [
        {k: link[k] for k in ("type", "src")} for link in item["links_in"]
    ]
    assert rb.get_item("us-0830-shock-5m")["key"] == US  # unique item_id works


def test_lessons():
    active = rb.lessons()
    assert [les["id"] for les in active] == [
        f"L:{R2}:cost-floor-v2",
        f"L:{R1}:duka-gaps",
    ]
    data = rb.lessons(about=["dukascopy-5min"], type="data")
    assert [les["id"] for les in data] == [f"L:{R1}:duka-gaps"]
    # Lessons reachable through their evidence items' links.
    assert f"L:{R2}:cost-floor-v2" in [les["id"] for les in rb.lessons(about=["US500"])]
    assert len(rb.lessons(status="any")) == 3
    assert rb.lessons("gaps")[0]["id"] == f"L:{R1}:duka-gaps"


def test_neighbours():
    g = rb.neighbours("cause:below-cost-floor", direction="in")
    assert set(g["items"]) == {CB, US}
    g2 = rb.neighbours("EURUSD", depth=2)
    assert (
        f"item:{CB}" in g2["nodes"]
        and "theme:event-shock-continuation-5m" in g2["nodes"]
    )


def test_lineage():
    chain = rb.lineage(US)["chain"]
    assert [c["key"] for c in chain] == [CB, US]
    assert [c["key"] for c in rb.lineage(CB)["chain"]] == [CB, US]


def test_theme():
    themes = rb.theme()
    assert themes[0]["slug"] == "event-shock-continuation-5m"
    th = rb.theme("event-shock-continuation-5m")
    assert th["n_trials"] == 13 and len(th["items"]) == 2
    assert th["note"].startswith("Two triggers")
    # Scoped to the theme, or evidenced by one of its items (the data caveat came from cb-shock-5m).
    assert {les["id"] for les in th["lessons"]} == {
        f"L:{R2}:cost-floor-v2",
        f"L:{R1}:duka-gaps",
    }


def test_untested():
    gaps = rb.untested("EURUSD", "mechanism")
    assert gaps["tested"] == {"event-drift": 1}
    assert "momentum" in gaps["untested"] and "event-drift" not in gaps["untested"]
    with pytest.raises(ValueError):
        rb.untested("EURUSD", "colour")


def test_resolve_errors():
    with pytest.raises(ValueError):
        rb.neighbours("no-such-thing")


def test_contradictions_empty():
    out = rb.contradictions()
    assert out == {
        "lesson_vs_lesson": [],
        "lessons_with_counter_evidence": [],
        "item_vs_item": [],
    }
