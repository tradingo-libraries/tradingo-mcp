"""Results layer over tradingo_quant.analytics.summary."""

from __future__ import annotations

from typing import Any

from tradingo_mcp.arctic import arctic_client

_DEFAULT_COMPARE_METRICS = ("sharpe_ann", "max_drawdown", "total_pnl", "win_rate")

BACKTEST_FIELDS = (
    "unrealised_pnl",
    "realised_pnl",
    "total_pnl",
    "net_investment",
    "net_position",
    "net_exposure",
    "avg_open_price",
    "stop_trade",
)


def get_metrics(run_id: str) -> dict[str, Any]:
    from tradingo_quant.analytics.summary import summarize_backtest

    ac = arctic_client()
    lib = ac.get_library("backtest")
    pf = lib.read(f"research.{run_id}.portfolio", lazy=False).data
    try:
        pos = lib.read(f"research.{run_id}.instrument.net_position", lazy=False).data
    except Exception:
        pos = None
    return summarize_backtest(pf, net_position=pos)


def get_portfolio_summary(run_id: str, max_rows: int = 10_000) -> str:
    ac = arctic_client()
    lib = ac.get_library("backtest")
    df = lib.read(f"research.{run_id}.portfolio", lazy=False).data
    if len(df) > max_rows:
        step = max(1, len(df) // max_rows)
        df = df.iloc[::step]
    return df.to_json(orient="split", date_format="iso")


def get_instrument_pnl(run_id: str, field: str = "total_pnl") -> str:
    if field not in BACKTEST_FIELDS:
        raise ValueError(f"field must be one of {BACKTEST_FIELDS}")
    ac = arctic_client()
    lib = ac.get_library("backtest")
    df = lib.read(f"research.{run_id}.instrument.{field}", lazy=False).data
    return df.to_json(orient="split", date_format="iso")


def compare(run_ids: list[str], metrics: list[str] | None = None) -> str:
    cols = list(metrics or _DEFAULT_COMPARE_METRICS)
    rows = []
    for rid in run_ids:
        try:
            m = get_metrics(rid)
            rows.append({**{"run_id": rid}, **{c: m.get(c, "N/A") for c in cols}})
        except Exception as e:
            rows.append({"run_id": rid, **{c: f"error: {e}" for c in cols}})

    header = "| run_id | " + " | ".join(cols) + " |"
    sep = "|---|" + "|---|" * len(cols)
    lines = [header, sep]
    for row in rows:
        vals = [
            str(
                round(row.get(c, "N/A"), 4)
                if isinstance(row.get(c), float)
                else row.get(c, "N/A")
            )
            for c in cols
        ]
        lines.append(f"| {row['run_id']} | " + " | ".join(vals) + " |")
    return "\n".join(lines)


def open_in_monitor(run_id: str) -> str:
    monitor_host = "monitor"
    return f"http://{monitor_host}:8082/?portfolio=research.{run_id}"
