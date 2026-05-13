"""Volatility sizing helpers."""

from __future__ import annotations

from typing import Any

from tradingo_mcp.arctic import arctic_client


def vol_target_size(
    universe: str,
    capital: float,
    annual_vol_target: float = 0.20,
    halflife: int = 36,
    lookback_days: int = 120,
) -> dict[str, Any]:
    """Compute per-instrument notional size based on vol targeting."""
    try:
        from tradingo_quant.analytics.returns import returns as _returns
        from tradingo_quant.analytics.vol import vol as _vol
    except ImportError as e:
        return {"error": f"tradingo_quant not available: {e}"}

    ac = arctic_client()
    lib = ac.get_library("prices")
    close = lib.read(f"{universe}.mid.close", lazy=False).data.tail(lookback_days)

    rets = _returns(close)
    vols = _vol(rets, halflife=halflife)
    latest_vol = vols.iloc[-1]
    latest_price = close.iloc[-1]

    result: dict[str, Any] = {}
    for instrument in latest_vol.index:
        iv = latest_vol[instrument]
        price = latest_price[instrument]
        if iv > 0 and price > 0:
            notional = (capital * annual_vol_target) / (iv * price)
            result[instrument] = round(notional, 4)
        else:
            result[instrument] = None

    return {"capital": capital, "annual_vol_target": annual_vol_target, "sizes": result}


def portfolio_summary() -> dict[str, Any]:
    """Aggregate latest net/gross exposure across all live portfolios."""
    ac = arctic_client()
    lib = ac.get_library("portfolio")
    portfolios = lib.list_symbols(regex=r".+\.unlimited$")

    result = []
    for sym in portfolios:
        try:
            df = lib.read(sym, lazy=False).data.tail(1)
            result.append(
                {
                    "symbol": sym,
                    "net_exposure": (
                        float(df["net_exposure"].iloc[0])
                        if "net_exposure" in df.columns
                        else None
                    ),
                    "gross_exposure": (
                        float(df["gross_exposure"].iloc[0])
                        if "gross_exposure" in df.columns
                        else None
                    ),
                }
            )
        except Exception:
            continue

    return {"portfolios": result}
