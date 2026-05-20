"""Analytics helpers for MCP tools: returns, volatility, and summary statistics."""

from __future__ import annotations

from typing import Any

import pandas as pd

from tradingo_mcp.arctic import arctic_client

# Default annualisation factors by resample frequency
_FREQ_ANN: dict[str, int] = {
    "D": 252,
    "B": 252,
    "W": 52,
    "W-FRI": 52,
    "ME": 12,
    "M": 12,
    "QE": 4,
    "Q": 4,
    "YE": 1,
    "Y": 1,
    "A": 1,
}


def _load(
    library: str,
    symbol: str,
    start: str,
    end: str,
    frequency: str | None = None,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Read from ArcticDB with no row cap, optionally resampled."""
    client = arctic_client()
    lib = client.get_library(library)
    date_range = (pd.Timestamp(start), pd.Timestamp(end))
    item = lib.read(symbol, date_range=date_range, columns=columns, lazy=False)
    df: pd.DataFrame = item.data
    if frequency:
        df = df.resample(frequency).last().dropna(how="all")
    return df


def _window_kwargs(method: str, halflife: int | None, window: int | None) -> dict:
    if method == "ewm":
        if halflife is None:
            raise ValueError("halflife is required when method='ewm'")
        return {"halflife": halflife}
    if method == "rolling":
        if window is None:
            raise ValueError("window is required when method='rolling'")
        return {"window": window}
    return {}


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
) -> pd.DataFrame:
    """Compute a returns or volatility time series.

    kind="returns": period pct returns, or their EWM/rolling mean if method given.
    kind="vol":     rolling/EWM/expanding annualised vol; method=None → scalar std.

    annualisation defaults to the frequency-implied factor (252 for daily, 52 for weekly).
    """
    from tradingo_quant.analytics.returns import returns as _returns
    from tradingo_quant.analytics.vol import vol as _vol

    df = _load(library, symbol, start, end, frequency=frequency, columns=columns)
    ann = (
        annualisation
        if annualisation is not None
        else _FREQ_ANN.get(frequency or "", 1)
    )
    rets = _returns(df, period=period, kind="pct")

    if kind == "returns":
        if method is None:
            return rets
        kw = _window_kwargs(method, halflife, window)
        return getattr(rets, method)(**kw).mean() * ann

    if kind == "vol":
        if method is None:
            # Outright std over the full window, scaled to the annualisation factor
            return (rets.std() * (ann**0.5)).to_frame("vol")
        kw = _window_kwargs(method, halflife, window)
        return _vol(rets, annualisation=ann, how=method, **kw)

    raise ValueError(f"kind must be 'vol' or 'returns', got {kind!r}")


def describe_timeseries(
    library: str,
    symbol: str,
    end: str,
    lookback_days: int = 365,
    factor_library: str | None = None,
    factor_symbols: list[str] | None = None,
    columns: list[str] | None = None,
) -> dict[str, Any]:
    """Summary statistics for a price time series, resampled to daily.

    Returns:
        52-week high/low and distance from current price.
        Period returns: 1d, 5d, 20d, 60d, 252d.
        EWM vol (halflife=36 days, annualised).
        Annualised Sharpe ratio.
        Max drawdown.
        Beta to each factor symbol (insample OLS), if supplied.
    """
    from tradingo_quant.analytics.returns import returns as _returns
    from tradingo_quant.analytics.vol import vol as _vol

    end_ts = pd.Timestamp(end)
    start_ts = end_ts - pd.Timedelta(days=max(lookback_days, 365) + 30)

    raw = _load(library, symbol, str(start_ts), end, columns=columns)
    if raw.empty:
        return {"error": "no data"}

    # Resample to daily for interpretable N-day statistics
    df = raw.resample("D").last().dropna(how="all")
    if df.empty:
        return {"error": "no data after daily resample"}

    current = df.iloc[-1]
    window_252 = df.tail(252)
    high_52 = window_252.max()
    low_52 = window_252.min()

    rets = _returns(df, period=1, kind="pct").dropna(how="all")

    # N-day total returns
    period_returns: dict[str, dict] = {}
    for n in (1, 5, 20, 60, 252):
        if len(df) > n:
            r = (df.iloc[-1] / df.iloc[-(n + 1)] - 1).round(6)
            period_returns[f"{n}d"] = r.to_dict()

    # EWM vol (halflife 36 days), annualised to daily
    ewm_vol = _vol(rets, annualisation=252, how="ewm", halflife=36).iloc[-1].round(6)

    # Annualised Sharpe
    mean_ann = rets.mean() * 252
    std_ann = rets.std() * (252**0.5)
    sharpe = (mean_ann / std_ann.replace(0.0, float("nan"))).round(4)

    # Max drawdown
    cum = (1 + rets).cumprod()
    max_dd = ((cum - cum.cummax()) / cum.cummax()).min().round(6)

    result: dict[str, Any] = {
        "current_price": current.round(6).to_dict(),
        "52wk_high": high_52.round(6).to_dict(),
        "52wk_low": low_52.round(6).to_dict(),
        "pct_from_52wk_high": ((current - high_52) / high_52).round(6).to_dict(),
        "pct_from_52wk_low": ((current - low_52) / low_52).round(6).to_dict(),
        "returns": period_returns,
        "annualised_vol_ewm36": ewm_vol.to_dict(),
        "sharpe_ratio": sharpe.to_dict(),
        "max_drawdown": max_dd.to_dict(),
    }

    if factor_library and factor_symbols:
        result["betas"] = _compute_betas(
            rets, factor_library, factor_symbols, str(start_ts), end
        )

    return result


def fetch_covariance(
    library: str,
    symbol: str,
    start: str,
    end: str,
    annualisation: int | None = None,
    frequency: str | None = "D",
    columns: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    """Compute an insample covariance matrix over the given period.

    Returns a dict-of-dicts {instrument: {instrument: covariance}} suitable
    for direct JSON serialisation.  Prices are resampled to `frequency`
    (default "D") before computing returns, so the matrix represents
    annualised covariances at daily granularity unless overridden.

    annualisation defaults to the frequency-implied factor (252 for daily,
    52 for weekly) if not supplied.
    """
    from tradingo_quant.analytics.cov import cov as _cov
    from tradingo_quant.analytics.returns import returns as _returns

    df = _load(library, symbol, start, end, frequency=frequency, columns=columns)
    ann = (
        annualisation
        if annualisation is not None
        else _FREQ_ANN.get(frequency or "", 1)
    )
    rets = _returns(df, period=1, kind="pct").dropna(how="all")

    cov_matrix = _cov(rets, annualisation=ann, how="insample")
    return {
        str(row): {str(col): round(float(v), 8) for col, v in row_data.items()}
        for row, row_data in cov_matrix.iterrows()
    }


def _compute_betas(
    asset_rets: pd.DataFrame,
    factor_library: str,
    factor_symbols: list[str],
    start: str,
    end: str,
) -> dict[str, Any]:
    """Insample OLS beta per factor: cov(asset, factor) / var(factor)."""
    from tradingo_quant.analytics.returns import returns as _returns

    result: dict[str, Any] = {}
    for fsym in factor_symbols:
        try:
            factor_raw = _load(factor_library, fsym, start, end)
            factor_daily = factor_raw.resample("D").last().dropna(how="all")
            factor_rets = _returns(factor_daily, period=1, kind="pct")

            # Align on common index; take first factor column if multicolumn
            f_series = (
                factor_rets.iloc[:, 0]
                if factor_rets.shape[1] > 1
                else factor_rets.squeeze()
            )
            combined = pd.concat(
                [asset_rets, f_series.rename("__factor")], axis=1
            ).dropna()

            fvar = combined["__factor"].var()
            if fvar == 0:
                result[fsym] = {"error": "zero variance factor"}
                continue

            asset_cols = [c for c in combined.columns if c != "__factor"]
            result[fsym] = {
                col: round(float(combined[[col, "__factor"]].cov().iloc[0, 1]) / fvar, 4)  # type: ignore[arg-type]
                for col in asset_cols
            }
        except Exception as exc:
            result[fsym] = {"error": str(exc)}

    return result
