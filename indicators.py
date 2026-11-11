"""
Technical indicator calculations on OHLCV DataFrames.

Supply / demand zones: a short consolidation base followed by an impulsive
departure (rally for demand, drop for supply). Zone bounds come from the base
candles' range; tune base_bars and impulse_atr_mult for your timeframe.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SupplyDemandZone:
    """Price band where institutional interest is inferred from structure."""

    kind: Literal["demand", "supply"]
    bottom: float
    top: float
    formed_at: pd.Timestamp
    impulse_at: pd.Timestamp
    active: bool = True

    @property
    def mid(self) -> float:
        return (self.bottom + self.top) / 2.0


def calculate_sma(
    df: pd.DataFrame,
    column: str = "Close",
    periods: tuple[int, ...] = (50, 200),
) -> pd.DataFrame:
    """Add Simple Moving Average columns (e.g. SMA_50, SMA_200)."""
    out = df.copy()
    if column not in out.columns:
        raise KeyError(f"Column {column!r} not found in dataframe.")

    for period in periods:
        if period < 1:
            raise ValueError(f"SMA period must be >= 1, got {period}")
        out[f"SMA_{period}"] = out[column].rolling(window=period, min_periods=period).mean()
    return out


def calculate_rsi(df: pd.DataFrame, column: str = "Close", period: int = 14) -> pd.DataFrame:
    """Add RSI column using Wilder smoothing (standard 14-period default)."""
    out = df.copy()
    if column not in out.columns:
        raise KeyError(f"Column {column!r} not found in dataframe.")
    if period < 1:
        raise ValueError(f"RSI period must be >= 1, got {period}")

    delta = out[column].diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)

    gain_series = pd.Series(gain, index=out.index)
    loss_series = pd.Series(loss, index=out.index)

    avg_gain = gain_series.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss_series.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    out[f"RSI_{period}"] = 100 - (100 / (1 + rs))
    return out


def enrich_with_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Apply SMA 50/200 and RSI(14) used by signal rules."""
    enriched = calculate_sma(df, periods=(50, 200))
    enriched = calculate_rsi(enriched, period=14)
    return enriched


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range for measuring impulsive candle size."""
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window=period, min_periods=period).mean()


def _base_slice(df: pd.DataFrame, impulse_index: int, base_bars: int) -> pd.DataFrame:
    start = impulse_index - base_bars
    if start < 0:
        raise IndexError("Not enough history for base candles before impulse.")
    return df.iloc[start:impulse_index]


def _zone_broken(zone: SupplyDemandZone, df: pd.DataFrame) -> bool:
    """Zone fails after a close decisively through the far edge."""
    after = df.loc[df.index > zone.impulse_at]
    if after.empty:
        return False
    if zone.kind == "demand":
        return bool((after["Close"] < zone.bottom).any())
    return bool((after["Close"] > zone.top).any())


def identify_demand_zones(
    df: pd.DataFrame,
    *,
    base_bars: int = 3,
    impulse_atr_mult: float = 0.85,
    atr_period: int = 14,
) -> list[SupplyDemandZone]:
    """
    Demand zone (support): consolidation base then aggressive bullish departure.

    Base = last `base_bars` candles before the impulse; zone spans base Low–High.
    Impulse = bullish candle range >= impulse_atr_mult * ATR.
    """
    if base_bars < 1:
        raise ValueError("base_bars must be >= 1")

    atr = calculate_atr(df, period=atr_period)
    zones: list[SupplyDemandZone] = []

    for i in range(base_bars, len(df)):
        if pd.isna(atr.iloc[i]):
            continue
        row = df.iloc[i]
        open_, close = float(row["Open"]), float(row["Close"])
        if close <= open_:
            continue
        candle_range = float(row["High"] - row["Low"])
        threshold = impulse_atr_mult * float(atr.iloc[i])
        if candle_range < threshold:
            continue

        base = _base_slice(df, i, base_bars)
        bottom = float(base["Low"].min())
        top = float(base["High"].max())
        if top <= bottom:
            continue

        impulse_time = pd.Timestamp(df.index[i])
        zone = SupplyDemandZone(
            kind="demand",
            bottom=bottom,
            top=top,
            formed_at=pd.Timestamp(base.index[-1]),
            impulse_at=impulse_time,
        )
        if not _zone_broken(zone, df):
            zones.append(zone)

    return _dedupe_zones(zones)


def identify_supply_zones(
    df: pd.DataFrame,
    *,
    base_bars: int = 3,
    impulse_atr_mult: float = 0.85,
    atr_period: int = 14,
) -> list[SupplyDemandZone]:
    """
    Supply zone (resistance): consolidation base then aggressive bearish departure.

    Base = last `base_bars` candles before the impulse; zone spans base Low–High.
    Impulse = bearish candle range >= impulse_atr_mult * ATR.
    """
    if base_bars < 1:
        raise ValueError("base_bars must be >= 1")

    atr = calculate_atr(df, period=atr_period)
    zones: list[SupplyDemandZone] = []

    for i in range(base_bars, len(df)):
        if pd.isna(atr.iloc[i]):
            continue
        row = df.iloc[i]
        open_, close = float(row["Open"]), float(row["Close"])
        if close >= open_:
            continue
        candle_range = float(row["High"] - row["Low"])
        threshold = impulse_atr_mult * float(atr.iloc[i])
        if candle_range < threshold:
            continue

        base = _base_slice(df, i, base_bars)
        bottom = float(base["Low"].min())
        top = float(base["High"].max())
        if top <= bottom:
            continue

        impulse_time = pd.Timestamp(df.index[i])
        zone = SupplyDemandZone(
            kind="supply",
            bottom=bottom,
            top=top,
            formed_at=pd.Timestamp(base.index[-1]),
            impulse_at=impulse_time,
        )
        if not _zone_broken(zone, df):
            zones.append(zone)

    return _dedupe_zones(zones)


def _dedupe_zones(zones: list[SupplyDemandZone], tolerance: float = 1e-5) -> list[SupplyDemandZone]:
    """Keep the most recent zone when bounds overlap the same kind."""
    if not zones:
        return []

    sorted_zones = sorted(zones, key=lambda z: z.impulse_at)
    merged: list[SupplyDemandZone] = []
    for zone in sorted_zones:
        replaced = False
        for idx, existing in enumerate(merged):
            if existing.kind != zone.kind:
                continue
            overlap = zone.bottom <= existing.top + tolerance and zone.top >= existing.bottom - tolerance
            if overlap:
                merged[idx] = zone
                replaced = True
                break
        if not replaced:
            merged.append(zone)
    return merged


def calculate_supply_demand_zones(
    df: pd.DataFrame,
    *,
    base_bars: int = 3,
    impulse_atr_mult: float = 0.85,
    atr_period: int = 14,
    max_active: int = 8,
) -> list[SupplyDemandZone]:
    """
    Detect active demand and supply zones on OHLC history.

    Returns up to `max_active` most recent non-broken zones (mixed types),
    sorted by impulse time descending (newest first).
    """
    for col in ("Open", "High", "Low", "Close"):
        if col not in df.columns:
            raise KeyError(f"Missing OHLC column {col!r}")

    demand = identify_demand_zones(
        df, base_bars=base_bars, impulse_atr_mult=impulse_atr_mult, atr_period=atr_period
    )
    supply = identify_supply_zones(
        df, base_bars=base_bars, impulse_atr_mult=impulse_atr_mult, atr_period=atr_period
    )
    combined = sorted(demand + supply, key=lambda z: z.impulse_at, reverse=True)
    return combined[:max_active]
