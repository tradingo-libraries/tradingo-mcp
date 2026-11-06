"""Optional OpenTelemetry spans for MCP tool calls and ArcticDB access.

No-ops unless ``opentelemetry-api`` is importable; a tracer provider is configured by
``opentelemetry-instrument`` (see the ``mcp`` image CMD in tradingo-plat). Deliberately does not
depend on ``tradingo.telemetry`` so this server works with any deployed ``tradingo`` version.
"""

from __future__ import annotations

import contextlib
import functools
from collections.abc import Callable, Iterator
from typing import Any

try:
    from opentelemetry import trace as _trace

    _tracer: Any = _trace.get_tracer("tradingo_mcp")
except ImportError:  # pragma: no cover
    _tracer = None

_PRIMITIVES = (str, bool, int, float)


@contextlib.contextmanager
def span(name: str, **attributes: Any) -> Iterator[Any]:
    """Child span of the current context; yields the span (``None`` without OTel)."""
    if _tracer is None:
        yield None
        return
    clean = {
        k: v if isinstance(v, _PRIMITIVES) else str(v)
        for k, v in attributes.items()
        if v is not None
    }
    with _tracer.start_as_current_span(name, attributes=clean) as s:
        yield s


def traced_tool(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap an MCP tool function so each call is a ``mcp.tool <name>`` span.

    Only argument *names* are recorded: values can be large (configs, code) or sensitive.
    """

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        with span(
            f"mcp.tool {fn.__name__}",
            **{
                "rpc.system": "mcp",
                "rpc.method": fn.__name__,
                "mcp.tool.args": ",".join(sorted(kwargs)),
            },
        ):
            return fn(*args, **kwargs)

    return wrapper
