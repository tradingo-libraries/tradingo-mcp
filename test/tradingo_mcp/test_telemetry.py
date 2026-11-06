from collections.abc import Iterator

import arcticdb as adb
import pandas as pd
import pytest

pytest.importorskip("opentelemetry.sdk")

from opentelemetry.sdk.trace import TracerProvider  # noqa: E402
from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (  # noqa: E402
    InMemorySpanExporter,
)

from tradingo_mcp import arctic, server, telemetry  # noqa: E402


@pytest.fixture
def spans(monkeypatch: pytest.MonkeyPatch) -> Iterator[InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(telemetry, "_tracer", provider.get_tracer("test"))
    yield exporter
    provider.shutdown()


def test_traced_tool_records_name_and_arg_names_not_values(
    spans: InMemorySpanExporter,
) -> None:
    @telemetry.traced_tool
    def my_tool(query: str, limit: int = 5) -> str:
        return query

    assert my_tool(query="secret", limit=3) == "secret"
    (s,) = spans.get_finished_spans()
    assert s.name == "mcp.tool my_tool"
    assert s.attributes is not None
    assert s.attributes["rpc.system"] == "mcp"
    assert s.attributes["mcp.tool.args"] == "limit,query"
    assert "secret" not in str(s.attributes)


def test_traced_tool_records_exceptions(spans: InMemorySpanExporter) -> None:
    @telemetry.traced_tool
    def broken() -> None:
        raise RuntimeError("nope")

    with pytest.raises(RuntimeError):
        broken()
    (s,) = spans.get_finished_spans()
    assert s.status.status_code.name == "ERROR"


def test_registered_tools_are_wrapped() -> None:
    tool = server.mcp._tool_manager.get_tool("list_libraries")
    assert tool is not None
    assert getattr(tool.fn, "__wrapped__", None) is not None


def test_fetch_bars_emits_arcticdb_read_span(
    spans: InMemorySpanExporter, monkeypatch: pytest.MonkeyPatch
) -> None:
    ac = adb.Arctic("mem://mcp-telemetry")
    lib = ac.get_library("prices", create_if_missing=True)
    lib.write(
        "X",
        pd.DataFrame(
            {"close": range(10)}, index=pd.date_range("2025-01-01", periods=10)
        ),
    )
    monkeypatch.setattr(arctic, "_ARCTIC", ac)

    df, _ = arctic.fetch_bars("prices", "X", "2025-01-01", "2025-01-10")

    (s,) = spans.get_finished_spans()
    assert s.name == "arcticdb.read"
    assert s.attributes is not None
    assert s.attributes["arcticdb.library"] == "prices"
    assert s.attributes["arcticdb.rows"] == len(df)
