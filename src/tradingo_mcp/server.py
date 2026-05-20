"""FastMCP app, tool registration, response-size cap."""

from __future__ import annotations

import functools
import json
import os
import textwrap
from typing import Any

import arcticdb as adb
from mcp.server.fastmcp import FastMCP

from tradingo_mcp import analytics as _analytics
from tradingo_mcp import arctic as _arctic
from tradingo_mcp import config_io, news, notifications, results, risk, runner

MAX_BYTES = int(os.environ.get("TP_MCP_MAX_BYTES", str(2 * 1024 * 1024)))


class McpResponseTooLarge(Exception):
    pass


def _capped(fn):  # type: ignore[no-untyped-def]
    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        result = fn(*args, **kwargs)
        encoded = json.dumps(result, default=str)
        if len(encoded.encode("utf-8")) > MAX_BYTES:
            raise McpResponseTooLarge(
                f"{fn.__name__} returned {len(encoded)} bytes "
                f"(cap={MAX_BYTES}). Narrow your query and retry."
            )
        return result

    return wrapper


_host = os.environ.get("MCP_HOST", "0.0.0.0")
_port = int(os.environ.get("MCP_PORT", "8765"))

mcp = FastMCP(
    name="tradingo-research",
    host=_host,
    port=_port,
    instructions=textwrap.dedent("""
        You are a quantimental/systematic trading research agent inside the Tradingo
        platform. You author Tradingo *task-graph* configs. 
        Your output is a backtested config that a human can review and promote.

        RESEARCH LOOP
        1. news.fetch / news.calendar      → form a thesis
        2. data.list_universes             → see what instrument universes exist
        3. data.describe_symbol(...)       → confirm data availability for the window
        4. config.list_templates           → pick a starter template
        5. config.render_template(...)     → returns (run_id, yaml). The
                                             yaml is already namespaced; you
                                             only need to edit `params`.
        6. config.write_config(run_id, yaml) → validates + writes under research_configs/
        7. run.execute(run_id, ...)        → dispatches to Celery (the cluster)
        8. results.get_metrics(run_id)     → Sharpe, drawdown, hit rate, turnover
        9. iterate; results.compare(run_ids) when you have multiple
    """),
)


# ---------------------------------------------------------------------------
# data tools
# ---------------------------------------------------------------------------


@mcp.tool()
@_capped
def list_libraries() -> list[str]:
    """List ArcticDB libraries available on this cluster."""
    return _arctic.arctic_client().list_libraries()


@mcp.tool()
@_capped
def list_universes() -> list[dict]:
    """List configured instrument universes from config/tradingo/universes/."""
    import pathlib

    import yaml

    config_home = pathlib.Path(
        os.environ.get("TP_CONFIG_HOME", "/opt/airflow/config/tradingo")
    )
    universes_dir = config_home / "universes"
    result: list[dict] = []
    if not universes_dir.exists():
        return result
    for p in sorted(universes_dir.glob("*.yaml")):
        try:
            data = yaml.safe_load(p.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                continue
            instruments: list = data.get("epics") or data.get("tickers") or []
            result.append(
                {
                    "name": data.get("universe_name", p.stem),
                    "interval": data.get("interval"),
                    "n_instruments": len(instruments),
                    "sample_epics": instruments[:5],
                }
            )
        except Exception:
            continue
    return result


@mcp.tool()
@_capped
def describe_symbol(library: str, symbol: str) -> dict:
    """SymbolDescription: date range, row count, columns, last update."""
    return _arctic.describe_symbol(library, symbol)


@mcp.tool()
@_capped
def list_symbols(library: str, regex: str = "") -> list[str]:
    """List symbols in a library, optionally filtered by regex."""
    return _arctic.list_symbols(library, regex)


@mcp.tool()
@_capped
def fetch_bars(
    library: str,
    symbol: str,
    start: str,
    end: str,
    columns: list[str] | None = None,
    max_rows: int = 10_000,
    frequency: str | None = None,
) -> str:
    """Read a time slice. Returns split-orient JSON. Downsamples if > max_rows.

    frequency: optional resample rule applied before the row cap.
               "D" = daily last, "W" = weekly last, "ME" = month-end last.
    """
    df, resampled = _arctic.fetch_bars(
        library, symbol, start, end, columns, max_rows, frequency
    )
    payload = json.loads(df.to_json(orient="split", date_format="iso"))
    if resampled:
        payload["_resampled_to"] = max_rows
    if frequency:
        payload["_frequency"] = frequency
    return json.dumps(payload)


@mcp.tool()
@_capped
def fetch_analytics(
    library: str,
    symbol: str,
    start: str,
    end: str,
    kind: str = "vol",
    method: str | None = None,
    halflife: int | None = None,
    window: int | None = None,
    period: int = 1,
    annualisation: int | None = None,
    frequency: str | None = None,
    columns: list[str] | None = None,
) -> str:
    """Compute a returns or volatility time series from price data.

    kind:          "vol" (default) | "returns"
    method:        None = outright scalar, "ewm" = exponentially weighted,
                   "rolling" = rolling window, "expanding" = expanding window.
    halflife:      periods; required when method="ewm".
    window:        periods; required when method="rolling".
    period:        look-back periods for returns computation (default=1).
    annualisation: scaling factor — 252 (daily→annual), 52 (weekly→annual), 1 (none).
                   Inferred from frequency if not supplied.
    frequency:     optional resample of prices before computing ("D", "W", "ME").

    Returns split-orient JSON of the resulting time series.
    """
    df = _analytics.fetch_analytics(
        library,
        symbol,
        start,
        end,
        kind=kind,
        method=method,
        halflife=halflife,
        window=window,
        period=period,
        annualisation=annualisation,
        frequency=frequency,
        columns=columns,
    )
    return df.to_json(orient="split", date_format="iso")


@mcp.tool()
@_capped
def fetch_covariance(
    library: str,
    symbol: str,
    start: str,
    end: str,
    annualisation: int | None = None,
    frequency: str | None = "D",
    columns: list[str] | None = None,
) -> dict:
    """Compute an insample annualised covariance matrix over the given period.

    Returns a dict-of-dicts {instrument: {instrument: covariance}}.
    frequency: resample prices before computing returns ("D"=daily, "W"=weekly).
               Defaults to "D". Annualisation inferred from frequency if not supplied.
    columns: optional list of instrument names to limit the matrix size.
    """
    return _analytics.fetch_covariance(
        library,
        symbol,
        start,
        end,
        annualisation=annualisation,
        frequency=frequency,
        columns=columns,
    )


@mcp.tool()
@_capped
def describe_timeseries(
    library: str,
    symbol: str,
    end: str,
    lookback_days: int = 365,
    factor_library: str | None = None,
    factor_symbols: list[str] | None = None,
    columns: list[str] | None = None,
) -> dict:
    """Summary statistics for a time series, resampled to daily.

    Returns:
      current_price, 52wk_high, 52wk_low, distance from those levels.
      Period returns: 1d, 5d, 20d, 60d, 252d.
      annualised_vol_ewm36: EWM volatility (halflife=36 days), annualised.
      sharpe_ratio: annualised Sharpe over the lookback window.
      max_drawdown: peak-to-trough drawdown.
      betas: per-factor insample OLS beta, if factor_library and factor_symbols given.

    columns: optional filter — pass instrument names to limit output size.
    """
    return _analytics.describe_timeseries(
        library,
        symbol,
        end,
        lookback_days=lookback_days,
        factor_library=factor_library,
        factor_symbols=factor_symbols,
        columns=columns,
    )


@mcp.tool()
@_capped
def head_signal(symbol: str, n: int = 20) -> str:
    """Quick peek at a recently-written research signal output."""
    ac = _arctic.arctic_client()
    lib = ac.get_library("signals")
    item = lib.head(symbol, n, lazy=False)
    assert isinstance(item, adb.VersionedItem)
    return item.data.to_json(orient="split", date_format="iso")


# ---------------------------------------------------------------------------
# config tools
# ---------------------------------------------------------------------------


@mcp.tool()
@_capped
def get_schema() -> str:
    """Return the annotated task-graph schema."""
    return textwrap.dedent("""
        # Tradingo research config schema
        # Each top-level key with `depends_on` or `stage` is a task.
        <task.name>:
          function: "<dotted.python.path>"
          depends_on: ["<other.task.name>"]
          symbols_in:
            <kwarg>: "<library>/<dotted.symbol>"
          symbols_out:
            - "<library>/<dotted.symbol>"
          publish_args:
            symbol_prefix: "<prefix.>"    # MUST be "research.<run_id>."
          params: {...}
          enabled: true
    """)


@mcp.tool()
@_capped
def list_templates() -> list[dict]:
    """List available Jinja2 starter templates."""
    return config_io.list_templates()


@mcp.tool()
@_capped
def render_template(name: str, variables: dict, run_id: str | None = None) -> dict:
    """Render a starter template. Returns {run_id, prefix, yaml}.
    The server binds run_id and prefix — the agent's variables cannot override them.
    """
    return config_io.render_template(name, variables, run_id=run_id)


@mcp.tool()
@_capped
def list_research_configs() -> list[dict]:
    """List YAML configs under research_configs/."""
    return config_io.list_configs()


@mcp.tool()
@_capped
def read_research_config(run_id: str) -> str:
    """Read a research config YAML."""
    return config_io.read_config(run_id)


@mcp.tool()
@_capped
def write_research_config(run_id: str, yaml_text: str) -> dict:
    """Validate and write a YAML to research_configs/<run_id>.yaml."""
    return config_io.write_config(run_id, yaml_text)


@mcp.tool()
@_capped
def validate_research_config(run_id: str, yaml_text: str) -> dict:
    """Validate without persisting."""
    return config_io.validate_config(run_id, yaml_text)


@mcp.tool()
@_capped
def delete_research_config(run_id: str, also_delete_symbols: bool = True) -> dict:
    """Remove the YAML and optionally all ArcticDB symbols for this run_id."""
    deleted = config_io.delete_config(run_id)
    deleted_symbols = 0
    if also_delete_symbols:
        deleted_symbols = _arctic.delete_research_symbols(run_id)
    return {"config_deleted": deleted, "deleted_symbols": deleted_symbols}


# ---------------------------------------------------------------------------
# run tools
# ---------------------------------------------------------------------------


@mcp.tool()
@_capped
def execute(
    config_name: str,
    task_name: str,
    start_date: str,
    end_date: str,
    with_deps: bool = True,
    batch_interval: str | None = None,
    batch_mode: str = "stepped",
    executor: str = "celery",
    wait: bool = True,
    timeout_s: int = 3600,
) -> dict:
    """Run a research task via tradingo-cli subprocess dispatched to Celery."""
    return runner.execute(
        config_name=config_name,
        task_name=task_name,
        start_date=start_date,
        end_date=end_date,
        with_deps=with_deps,
        batch_interval=batch_interval,
        batch_mode=batch_mode,
        executor=executor,
        wait=wait,
        timeout_s=timeout_s,
    )


@mcp.tool()
@_capped
def list_runs(config_name: str | None = None, limit: int = 20) -> list[dict]:
    """List runs from research_runs/*.json."""
    return runner.list_runs(config_name=config_name, limit=limit)


@mcp.tool()
@_capped
def get_run_status(run_id: str) -> dict:
    """Get status of a run from its manifest."""
    return runner.get_run_status(run_id)


# ---------------------------------------------------------------------------
# results tools
# ---------------------------------------------------------------------------


@mcp.tool()
@_capped
def get_metrics(run_id: str) -> dict:
    """Read backtest results and return canonical metric dict."""
    return results.get_metrics(run_id)


@mcp.tool()
@_capped
def get_portfolio_summary(run_id: str) -> str:
    """Return backtest portfolio DataFrame as JSON for plotting."""
    return results.get_portfolio_summary(run_id)


@mcp.tool()
@_capped
def get_instrument_pnl(run_id: str, field: str = "total_pnl") -> str:
    """Read per-instrument backtest field."""
    return results.get_instrument_pnl(run_id, field)


@mcp.tool()
@_capped
def compare(run_ids: list[str], metrics: list[str] | None = None) -> str:
    """Side-by-side markdown table of metrics across runs."""
    return results.compare(run_ids, metrics=metrics)


@mcp.tool()
@_capped
def open_in_monitor(run_id: str) -> str:
    """Return Monitor URL pre-filtered to this portfolio."""
    return results.open_in_monitor(run_id)


# ---------------------------------------------------------------------------
# news tools
# ---------------------------------------------------------------------------


@mcp.tool()
@_capped
def fetch_news(query: str, lookback_days: int = 7, limit: int = 20) -> list[dict]:
    """Search Miniflux entries matching query, published within lookback_days."""
    return news.fetch(query, lookback_days=lookback_days, limit=limit)


@mcp.tool()
@_capped
def list_feeds() -> list[dict]:
    """List all RSS feeds configured in Miniflux."""
    return news.list_feeds()


@mcp.tool()
@_capped
def add_feed(feed_url: str, category_id: int | None = None) -> dict:
    """Add a new RSS feed to Miniflux."""
    return news.add_feed(feed_url, category_id=category_id)


@mcp.tool()
@_capped
def remove_feed(feed_id: int) -> dict:
    """Remove a feed from Miniflux by ID."""
    return news.remove_feed(feed_id)


@mcp.tool()
@_capped
def refresh_feeds(feed_id: int | None = None) -> dict:
    """Trigger a Miniflux feed refresh. Refreshes all feeds if feed_id is omitted."""
    return news.refresh_feeds(feed_id=feed_id)


@mcp.tool()
@_capped
def economic_calendar(
    lookback_days: int = 3,
    lookahead_days: int = 7,
    currencies: list[str] | None = None,
) -> list[dict]:
    """Fetch economic calendar events."""
    return news.calendar(
        lookback_days=lookback_days,
        lookahead_days=lookahead_days,
        currencies=currencies,
    )


# ---------------------------------------------------------------------------
# risk tools
# ---------------------------------------------------------------------------


@mcp.tool()
@_capped
def vol_target_size(
    universe: str,
    capital: float,
    annual_vol_target: float = 0.20,
    halflife: int = 36,
    lookback_days: int = 120,
) -> dict:
    """Compute per-instrument notional based on vol targeting."""
    return risk.vol_target_size(
        universe=universe,
        capital=capital,
        annual_vol_target=annual_vol_target,
        halflife=halflife,
        lookback_days=lookback_days,
    )


@mcp.tool()
@_capped
def live_portfolio_summary() -> dict:
    """Aggregate latest net/gross exposure across live portfolios."""
    return risk.portfolio_summary()


# ---------------------------------------------------------------------------
# notification tools
# ---------------------------------------------------------------------------


@mcp.tool()
def send_email(subject: str, body: str) -> dict:
    """Send an email to the configured recipients (TP_MCP_EMAIL_RECIPIENTS, semicolon-separated)."""
    return notifications.send_email(subject=subject, body=body)


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
