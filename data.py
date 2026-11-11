"""
Fetch and clean historical Forex OHLCV data.

Replace `fetch_forex_data` with your own broker API or database query while
keeping the same return shape for use in scan.py / backtesting loops.
"""

from __future__ import annotations

import pandas as pd

try:
    import yfinance as yf
except ImportError as exc:
    yf = None  # type: ignore
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

OHLCV_COLUMNS = ("Open", "High", "Low", "Close", "Volume")


def _clean_ohlcv(raw: pd.DataFrame) -> pd.DataFrame:
    if raw is None or raw.empty:
        raise ValueError("No data returned for the requested symbol.")

    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.columns = [str(c).strip().title() for c in df.columns]
    missing = [c for c in OHLCV_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Expected OHLCV columns {OHLCV_COLUMNS}; missing {missing}, got {list(df.columns)}"
        )

    df = df.loc[:, list(OHLCV_COLUMNS)]
    for col in OHLCV_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(how="any")
    if df.empty:
        raise ValueError("Dataframe is empty after cleaning.")

    df.index = pd.to_datetime(df.index)
    return df.sort_index()


def fetch_forex_data(
    symbol: str = "EURUSD=X",
    *,
    period: str = "2y",
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Download candlestick history from Yahoo Finance via yfinance.

    symbol: e.g. 'EURUSD=X', 'GBPUSD=X'
    period:   history window (use >= 1y daily for 200-period SMA)
    interval: bar size (default daily '1d')
    """
    if not symbol or not str(symbol).strip():
        raise ValueError("symbol must be a non-empty ticker string.")

    if yf is None:
        raise ImportError("yfinance is not installed. Run: pip install yfinance") from _IMPORT_ERROR

    try:
        raw = yf.download(
            symbol,
            period=period,
            interval=interval,
            auto_adjust=False,
            progress=False,
            threads=False,
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to download data for {symbol!r}: {exc}") from exc

    return _clean_ohlcv(raw)
