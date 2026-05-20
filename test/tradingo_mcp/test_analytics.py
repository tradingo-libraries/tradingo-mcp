"""Tests for tradingo_mcp.analytics using in-memory ArcticDB."""

from __future__ import annotations

from unittest.mock import patch

import arcticdb as adb
import numpy as np
import pandas as pd
import pytest

from tradingo_mcp import analytics

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_INSTRUMENTS = ["EUR/USD", "GBP/USD", "XAU/USD"]
_LIBRARY = "prices"
_SYMBOL = "test.mid.close"
_FACTOR_SYMBOL = "test.factor"

_START = "2024-01-01"
_END = "2025-06-01"


def _make_prices(
    instruments: list[str] = _INSTRUMENTS,
    start: str = _START,
    end: str = _END,
    freq: str = "D",
    seed: int = 42,
) -> pd.DataFrame:
    """Random-walk price series with a known seed."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, end, freq=freq, tz="UTC")
    n = len(idx)
    returns = rng.normal(0.0001, 0.005, size=(n, len(instruments)))
    prices = np.exp(np.cumsum(returns, axis=0))
    prices[0] = 1.0
    return pd.DataFrame(prices, index=idx, columns=instruments)


@pytest.fixture()
def mem_arctic():
    """In-memory ArcticDB with pre-populated price and factor data."""
    ac = adb.Arctic("mem://")
    lib = ac.create_library(_LIBRARY)
    lib.write(_SYMBOL, _make_prices())
    # factor: single-column series correlated with EUR/USD
    factor_df = _make_prices(["factor"], seed=99)
    lib.write(_FACTOR_SYMBOL, factor_df)
    return ac


@pytest.fixture(autouse=True)
def patch_arctic(mem_arctic):
    """Replace both call sites of arctic_client() with the in-memory instance."""
    with (
        patch("tradingo_mcp.analytics.arctic_client", return_value=mem_arctic),
        patch("tradingo_mcp.arctic.arctic_client", return_value=mem_arctic),
    ):
        yield


# ---------------------------------------------------------------------------
# _load
# ---------------------------------------------------------------------------


def test_load_returns_dataframe():
    df = analytics._load(_LIBRARY, _SYMBOL, _START, _END)
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == _INSTRUMENTS


def test_load_column_filter():
    df = analytics._load(_LIBRARY, _SYMBOL, _START, _END, columns=["EUR/USD"])
    assert list(df.columns) == ["EUR/USD"]


def test_load_frequency_daily_reduces_rows():
    # source data is already daily; resampling to weekly should cut rows ~7x
    df_daily = analytics._load(_LIBRARY, _SYMBOL, _START, _END)
    df_weekly = analytics._load(_LIBRARY, _SYMBOL, _START, _END, frequency="W")
    assert len(df_weekly) < len(df_daily) / 5


def test_load_frequency_preserves_last_value():
    # last value of weekly resample = last value of that week in the daily data
    df_daily = analytics._load(_LIBRARY, _SYMBOL, _START, _END)
    df_weekly = analytics._load(_LIBRARY, _SYMBOL, _START, _END, frequency="W")
    # the last row of both should agree (same final week)
    assert df_weekly.iloc[-1]["EUR/USD"] == pytest.approx(
        df_daily.iloc[-1]["EUR/USD"], rel=1e-6
    )


# ---------------------------------------------------------------------------
# fetch_analytics — returns
# ---------------------------------------------------------------------------


def test_fetch_analytics_returns_outright():
    df = analytics.fetch_analytics(_LIBRARY, _SYMBOL, _START, _END, kind="returns")
    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) == set(_INSTRUMENTS)
    assert not df.empty
    # pct returns should be small (our vol is ~0.5% per day)
    assert df["EUR/USD"].abs().mean() < 0.05


def test_fetch_analytics_returns_period_2():
    df1 = analytics.fetch_analytics(
        _LIBRARY, _SYMBOL, _START, _END, kind="returns", period=1
    )
    df5 = analytics.fetch_analytics(
        _LIBRARY, _SYMBOL, _START, _END, kind="returns", period=5
    )
    # 5-period returns should have higher absolute magnitude than 1-period
    assert df5["EUR/USD"].abs().mean() > df1["EUR/USD"].abs().mean()


def test_fetch_analytics_returns_ewm():
    df = analytics.fetch_analytics(
        _LIBRARY, _SYMBOL, _START, _END, kind="returns", method="ewm", halflife=20
    )
    assert isinstance(df, pd.DataFrame)
    # EWM mean of returns should be close to zero (centred random walk)
    assert df["EUR/USD"].abs().mean() < 0.01


def test_fetch_analytics_returns_rolling():
    df = analytics.fetch_analytics(
        _LIBRARY, _SYMBOL, _START, _END, kind="returns", method="rolling", window=20
    )
    assert isinstance(df, pd.DataFrame)
    assert not df.dropna().empty


def test_fetch_analytics_returns_with_frequency():
    df = analytics.fetch_analytics(
        _LIBRARY, _SYMBOL, _START, _END, kind="returns", frequency="W"
    )
    assert isinstance(df, pd.DataFrame)
    # weekly data → fewer rows than daily
    df_daily = analytics.fetch_analytics(
        _LIBRARY, _SYMBOL, _START, _END, kind="returns"
    )
    assert len(df) < len(df_daily) / 4


# ---------------------------------------------------------------------------
# fetch_analytics — vol
# ---------------------------------------------------------------------------


def test_fetch_analytics_vol_outright_scalar():
    df = analytics.fetch_analytics(_LIBRARY, _SYMBOL, _START, _END, kind="vol")
    # outright with method=None → instrument-indexed DataFrame with one "vol" column
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["vol"]
    assert set(df.index) == set(_INSTRUMENTS)
    assert (df["vol"] > 0).all()


def test_fetch_analytics_vol_outright_annualised():
    df = analytics.fetch_analytics(
        _LIBRARY, _SYMBOL, _START, _END, kind="vol", annualisation=252
    )
    assert list(df.columns) == ["vol"]
    # daily vol ~0.5% → annualised ~8%; allow generous range
    eur_vol = df.loc["EUR/USD", "vol"]
    assert 0.03 < eur_vol < 0.30


def test_fetch_analytics_vol_ewm_series():
    df = analytics.fetch_analytics(
        _LIBRARY,
        _SYMBOL,
        _START,
        _END,
        kind="vol",
        method="ewm",
        halflife=36,
        annualisation=252,
    )
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 100  # time series, not scalar
    # first row can be 0.0 (EWM initialised from a NaN return); skip it
    assert (df["EUR/USD"].iloc[1:] > 0).all()


def test_fetch_analytics_vol_rolling_series():
    df = analytics.fetch_analytics(
        _LIBRARY,
        _SYMBOL,
        _START,
        _END,
        kind="vol",
        method="rolling",
        window=20,
        annualisation=252,
    )
    assert isinstance(df, pd.DataFrame)
    assert len(df.dropna()) > 50
    assert (df["EUR/USD"].dropna() > 0).all()


def test_fetch_analytics_vol_expanding():
    df = analytics.fetch_analytics(
        _LIBRARY,
        _SYMBOL,
        _START,
        _END,
        kind="vol",
        method="expanding",
        annualisation=252,
    )
    assert isinstance(df, pd.DataFrame)
    assert not df.dropna().empty


def test_fetch_analytics_vol_frequency_weekly():
    df = analytics.fetch_analytics(
        _LIBRARY,
        _SYMBOL,
        _START,
        _END,
        kind="vol",
        method="ewm",
        halflife=12,
        frequency="W",
    )
    # annualisation inferred as 52 for weekly
    assert isinstance(df, pd.DataFrame)
    assert not df.dropna().empty


def test_fetch_analytics_vol_ewm_requires_halflife():
    with pytest.raises(ValueError, match="halflife"):
        analytics.fetch_analytics(
            _LIBRARY, _SYMBOL, _START, _END, kind="vol", method="ewm"
        )


def test_fetch_analytics_rolling_requires_window():
    with pytest.raises(ValueError, match="window"):
        analytics.fetch_analytics(
            _LIBRARY, _SYMBOL, _START, _END, kind="vol", method="rolling"
        )


def test_fetch_analytics_invalid_kind():
    with pytest.raises(ValueError, match="kind"):
        analytics.fetch_analytics(_LIBRARY, _SYMBOL, _START, _END, kind="covariance")


# ---------------------------------------------------------------------------
# describe_timeseries
# ---------------------------------------------------------------------------


def test_describe_timeseries_keys():
    result = analytics.describe_timeseries(_LIBRARY, _SYMBOL, _END)
    expected_keys = {
        "current_price",
        "52wk_high",
        "52wk_low",
        "pct_from_52wk_high",
        "pct_from_52wk_low",
        "returns",
        "annualised_vol_ewm36",
        "sharpe_ratio",
        "max_drawdown",
    }
    assert expected_keys.issubset(result.keys())


def test_describe_timeseries_instruments():
    result = analytics.describe_timeseries(_LIBRARY, _SYMBOL, _END)
    assert set(result["current_price"].keys()) == set(_INSTRUMENTS)


def test_describe_timeseries_column_filter():
    result = analytics.describe_timeseries(_LIBRARY, _SYMBOL, _END, columns=["EUR/USD"])
    assert list(result["current_price"].keys()) == ["EUR/USD"]


def test_describe_timeseries_52wk_range():
    result = analytics.describe_timeseries(_LIBRARY, _SYMBOL, _END)
    for instr in _INSTRUMENTS:
        assert result["52wk_high"][instr] >= result["current_price"][instr]
        assert result["52wk_low"][instr] <= result["current_price"][instr]


def test_describe_timeseries_pct_from_high_non_positive():
    result = analytics.describe_timeseries(_LIBRARY, _SYMBOL, _END)
    for instr in _INSTRUMENTS:
        assert result["pct_from_52wk_high"][instr] <= 0.0


def test_describe_timeseries_pct_from_low_non_negative():
    result = analytics.describe_timeseries(_LIBRARY, _SYMBOL, _END)
    for instr in _INSTRUMENTS:
        assert result["pct_from_52wk_low"][instr] >= 0.0


def test_describe_timeseries_returns_periods():
    result = analytics.describe_timeseries(_LIBRARY, _SYMBOL, _END)
    assert set(result["returns"].keys()) == {"1d", "5d", "20d", "60d", "252d"}
    for period_key, period_vals in result["returns"].items():
        assert set(period_vals.keys()) == set(_INSTRUMENTS)


def test_describe_timeseries_vol_positive():
    result = analytics.describe_timeseries(_LIBRARY, _SYMBOL, _END)
    for instr in _INSTRUMENTS:
        assert result["annualised_vol_ewm36"][instr] > 0.0


def test_describe_timeseries_max_drawdown_non_positive():
    result = analytics.describe_timeseries(_LIBRARY, _SYMBOL, _END)
    for instr in _INSTRUMENTS:
        assert result["max_drawdown"][instr] <= 0.0


def test_describe_timeseries_no_data_returns_error():
    result = analytics.describe_timeseries(
        _LIBRARY, _SYMBOL, "2020-01-01", lookback_days=30
    )
    assert "error" in result


# ---------------------------------------------------------------------------
# describe_timeseries — betas
# ---------------------------------------------------------------------------


def test_describe_timeseries_betas_keys():
    result = analytics.describe_timeseries(
        _LIBRARY,
        _SYMBOL,
        _END,
        factor_library=_LIBRARY,
        factor_symbols=[_FACTOR_SYMBOL],
    )
    assert "betas" in result
    assert _FACTOR_SYMBOL in result["betas"]
    assert set(result["betas"][_FACTOR_SYMBOL].keys()) == set(_INSTRUMENTS)


def test_describe_timeseries_betas_are_finite():
    result = analytics.describe_timeseries(
        _LIBRARY,
        _SYMBOL,
        _END,
        factor_library=_LIBRARY,
        factor_symbols=[_FACTOR_SYMBOL],
        columns=["EUR/USD"],
    )
    beta = result["betas"][_FACTOR_SYMBOL]["EUR/USD"]
    assert isinstance(beta, float)
    assert np.isfinite(beta)


def test_describe_timeseries_betas_missing_factor_graceful():
    result = analytics.describe_timeseries(
        _LIBRARY,
        _SYMBOL,
        _END,
        factor_library=_LIBRARY,
        factor_symbols=["nonexistent.symbol"],
    )
    assert "betas" in result
    assert "error" in result["betas"]["nonexistent.symbol"]


# ---------------------------------------------------------------------------
# fetch_bars frequency (via arctic.fetch_bars)
# ---------------------------------------------------------------------------


def test_fetch_bars_frequency_weekly(mem_arctic):
    from tradingo_mcp.arctic import fetch_bars

    df_raw, _ = fetch_bars(_LIBRARY, _SYMBOL, _START, _END)
    df_weekly, _ = fetch_bars(_LIBRARY, _SYMBOL, _START, _END, frequency="W")

    assert len(df_weekly) < len(df_raw) / 4
    assert isinstance(df_weekly.index, pd.DatetimeIndex)


def test_fetch_bars_frequency_monthly(mem_arctic):
    from tradingo_mcp.arctic import fetch_bars

    df_monthly, _ = fetch_bars(_LIBRARY, _SYMBOL, _START, _END, frequency="ME")
    # ~17 months of data
    assert 10 < len(df_monthly) < 25


def test_fetch_bars_no_frequency_unchanged(mem_arctic):
    from tradingo_mcp.arctic import fetch_bars

    df, resampled = fetch_bars(_LIBRARY, _SYMBOL, _START, _END)
    assert not resampled
    assert len(df) > 200
