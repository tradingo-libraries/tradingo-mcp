from __future__ import annotations

import pytest

from tradingo_mcp.results import open_in_monitor


def test_open_in_monitor_builds_shareable_view(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TP_MONITOR_URL", raising=False)
    assert (
        open_in_monitor("abc123")
        == "http://monitor:8082/?s=research.abc123~0&tf=ALL&f=auto"
    )


def test_open_in_monitor_base_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TP_MONITOR_URL", "https://monitor.example/")
    assert open_in_monitor("x").startswith("https://monitor.example/?s=research.x~0")
