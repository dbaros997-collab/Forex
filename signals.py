"""
Rule-based signal evaluation on the latest complete bar.

Chart, zone, and sizing logic are platform-independent. Native MT5 is optional.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from indicators import SupplyDemandZone

# Optional native MT5 — never required for cloud / mobile SaaS hosts.
MT5_AVAILABLE = False
_MT5_IMPORT_ERROR: str | None = None

try:
    from mt5_connect import MT5_AVAILABLE as _MT5_FLAG

    MT5_AVAILABLE = bool(_MT5_FLAG)
except ImportError as exc:
    _MT5_IMPORT_ERROR = str(exc)
except Exception as exc:  # pragma: no cover - defensive for broken native installs
    _MT5_IMPORT_ERROR = str(exc)


def get_platform_capabilities() -> dict[str, Any]:
    """Report what execution features are available on this runtime."""
    return {
        "chart_analysis": True,
        "zone_detection": True,
        "position_sizing": True,
        "webhook_execution": True,
        "mt5_native": MT5_AVAILABLE,
        "mt5_note": (
            "MetaTrader5 terminal not available on this host."
            if not MT5_AVAILABLE
            else "Local MT5 execution available."
        ),
        "mt5_import_error": _MT5_IMPORT_ERROR,
    }


def evaluate_signals(df: pd.DataFrame, symbol: str) -> dict[str, Any]:
    """
    Check the latest row for Golden Cross and Death Cross conditions.

    Golden Cross: SMA_50 crossed above SMA_200 (prev: SMA_50 <= SMA_200, now: SMA_50 > SMA_200).
    Death Cross:  SMA_50 crossed below SMA_200 (prev: SMA_50 >= SMA_200, now: SMA_50 < SMA_200).

    Returns a summary dictionary suitable for logging, APIs, or backtest hooks.
    """
    required = ("Close", "SMA_50", "SMA_200", "RSI_14")
    for col in required:
        if col not in df.columns:
            raise KeyError(f"Missing column {col!r}; run enrich_with_indicators() first.")

    valid = df.dropna(subset=list(required))
    if len(valid) < 2:
        raise ValueError(
            "Insufficient rows after indicator warm-up (need ~200+ bars for SMA_200)."
        )

    previous = valid.iloc[-2]
    latest = valid.iloc[-1]

    sma50_prev = float(previous["SMA_50"])
    sma200_prev = float(previous["SMA_200"])
    sma50 = float(latest["SMA_50"])
    sma200 = float(latest["SMA_200"])

    golden_cross = sma50_prev <= sma200_prev and sma50 > sma200
    death_cross = sma50_prev >= sma200_prev and sma50 < sma200
    trend_bullish = sma50 > sma200

    as_of = pd.Timestamp(latest.name)

    return {
        "symbol": symbol,
        "as_of": as_of.strftime("%Y-%m-%d"),
        "close": float(latest["Close"]),
        "sma_50": sma50,
        "sma_200": sma200,
        "rsi_14": float(latest["RSI_14"]),
        "sma50_above_sma200": trend_bullish,
        "golden_cross": golden_cross,
        "death_cross": death_cross,
        "golden_cross_status": "met" if golden_cross else "not met",
        "death_cross_status": "met" if death_cross else "not met",
    }


def _candle_intersects_zone(
    low: float,
    high: float,
    close: float,
    zone: SupplyDemandZone,
) -> bool:
    """True if the bar overlaps the zone or closes inside it."""
    overlaps = low <= zone.top and high >= zone.bottom
    close_inside = zone.bottom <= close <= zone.top
    return overlaps or close_inside


def _zone_to_dict(zone: SupplyDemandZone) -> dict[str, Any]:
    return {
        "kind": zone.kind,
        "bottom": zone.bottom,
        "top": zone.top,
        "formed_at": zone.formed_at.strftime("%Y-%m-%d"),
        "impulse_at": zone.impulse_at.strftime("%Y-%m-%d"),
    }


def evaluate_zone_signals(
    df: pd.DataFrame,
    zones: list[SupplyDemandZone],
    symbol: str,
) -> dict[str, Any]:
    """
    Check whether the latest candle is testing active supply or demand.

    Demand test: pullback — price trades into an active demand zone.
    Supply test: rally — price trades into an active supply zone.
    """
    if df.empty:
        raise ValueError("Dataframe is empty.")

    latest = df.iloc[-1]
    as_of = pd.Timestamp(latest.name)
    close = float(latest["Close"])
    low = float(latest["Low"])
    high = float(latest["High"])

    demand_zones = [z for z in zones if z.kind == "demand"]
    supply_zones = [z for z in zones if z.kind == "supply"]

    tested_demand = [
        z for z in demand_zones if _candle_intersects_zone(low, high, close, z)
    ]
    tested_supply = [
        z for z in supply_zones if _candle_intersects_zone(low, high, close, z)
    ]

    demand_test = len(tested_demand) > 0
    supply_test = len(tested_supply) > 0

    nearest_demand = _nearest_zone(close, demand_zones, prefer_below=True)
    nearest_supply = _nearest_zone(close, supply_zones, prefer_below=False)

    return {
        "symbol": symbol,
        "as_of": as_of.strftime("%Y-%m-%d"),
        "close": close,
        "low": low,
        "high": high,
        "active_demand_zones": [_zone_to_dict(z) for z in demand_zones],
        "active_supply_zones": [_zone_to_dict(z) for z in supply_zones],
        "demand_zone_test": demand_test,
        "supply_zone_test": supply_test,
        "demand_zone_status": "testing demand zone" if demand_test else "not in demand zone",
        "supply_zone_status": "testing supply zone" if supply_test else "not in supply zone",
        "tested_demand_zones": [_zone_to_dict(z) for z in tested_demand],
        "tested_supply_zones": [_zone_to_dict(z) for z in tested_supply],
        "nearest_demand": _zone_to_dict(nearest_demand) if nearest_demand else None,
        "nearest_supply": _zone_to_dict(nearest_supply) if nearest_supply else None,
    }


def _nearest_zone(
    price: float,
    zones: list[SupplyDemandZone],
    *,
    prefer_below: bool,
) -> SupplyDemandZone | None:
    if not zones:
        return None

    def distance(z: SupplyDemandZone) -> float:
        if z.bottom <= price <= z.top:
            return 0.0
        if price < z.bottom:
            return z.bottom - price
        return price - z.top

    candidates = zones
    if prefer_below:
        below = [z for z in zones if z.top <= price]
        if below:
            candidates = below

    if not prefer_below:
        above = [z for z in zones if z.bottom >= price]
        if above:
            candidates = above

    return min(candidates, key=distance)


def format_zone_summary(summary: dict[str, Any]) -> str:
    """Single string summary for logging or alerts."""
    parts = [
        f"close={summary['close']:.5f}",
        f"demand={summary['demand_zone_status']}",
        f"supply={summary['supply_zone_status']}",
    ]
    return "; ".join(parts)


# Re-export for backward compatibility (e.g. scan.py imports).
from position_sizing import (  # noqa: E402
    build_trade_recommendation,
    calculate_lot_size,
    pip_size,
    pip_value_per_standard_lot,
)

try:
    from webhook_bridge import (  # noqa: E402
        build_execution_bundle,
        package_trade_alert,
        send_trade_webhook,
        trade_alert_to_json,
    )
except ImportError:
    build_execution_bundle = None  # type: ignore
    package_trade_alert = None  # type: ignore
    send_trade_webhook = None  # type: ignore
    trade_alert_to_json = None  # type: ignore


__all__ = [
    "evaluate_signals",
    "evaluate_zone_signals",
    "format_zone_summary",
    "get_platform_capabilities",
    "MT5_AVAILABLE",
    "build_execution_bundle",
    "package_trade_alert",
    "trade_alert_to_json",
    "send_trade_webhook",
    "build_trade_recommendation",
    "calculate_lot_size",
    "pip_size",
    "pip_value_per_standard_lot",
]
