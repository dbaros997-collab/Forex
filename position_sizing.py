"""
Position sizing and trade recommendations from supply/demand zone tests.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from indicators import SupplyDemandZone


def _is_gold(symbol: str) -> bool:
    return "XAU" in symbol or symbol.startswith("GC")


def pip_size(pair: str) -> float:
    """One pip in price terms (0.01 for gold and JPY quotes, 0.0001 for most majors)."""
    symbol = pair.upper().replace("/", "")
    if _is_gold(symbol) or "JPY" in symbol:
        return 0.01
    return 0.0001


def pip_value_per_standard_lot(pair: str, entry_price: float) -> float:
    """
    USD value of a one-pip move per 1.0 standard lot (100k units).

    Majors quoted vs USD (EURUSD, GBPUSD, AUDUSD) use ~$10/pip.
    USD/JPY-style pairs use 1000 / rate.
    """
    if entry_price <= 0:
        raise ValueError("entry_price must be positive.")

    symbol = pair.upper().replace("/", "")
    if _is_gold(symbol):
        # 100 oz contract: a 0.01 price move is $1 per standard lot.
        return 1.0

    if not symbol.endswith("=X") and len(symbol) >= 6:
        symbol = f"{symbol}=X"

    if "JPY" in symbol:
        return 1000.0 / entry_price

    if symbol.endswith("USD=X") and not symbol.startswith("USD"):
        return 10.0

    return 10.0


def calculate_lot_size(
    account_balance: float,
    risk_pct: float,
    entry_price: float,
    stop_loss_price: float,
    pair: str,
) -> dict[str, Any]:
    """
    Position size from fixed fractional risk.

    stop_loss_pips = |entry - stop_loss| converted to pips.
    risk_amount = account_balance * (risk_pct / 100).
    lot_size = risk_amount / (stop_loss_pips * pip_value_per_lot).
    """
    if account_balance <= 0:
        raise ValueError("account_balance must be positive.")
    if risk_pct <= 0 or risk_pct > 100:
        raise ValueError("risk_pct must be between 0 and 100.")

    pip = pip_size(pair)
    sl_distance = abs(entry_price - stop_loss_price)
    stop_loss_pips = sl_distance / pip
    if stop_loss_pips <= 0:
        raise ValueError("Stop loss must be a different price level than entry.")

    risk_amount_usd = account_balance * (risk_pct / 100.0)
    pip_value = pip_value_per_standard_lot(pair, entry_price)
    lot_size = risk_amount_usd / (stop_loss_pips * pip_value)

    return {
        "pair": pair,
        "entry_price": entry_price,
        "stop_loss_price": stop_loss_price,
        "risk_amount_usd": risk_amount_usd,
        "stop_loss_pips": stop_loss_pips,
        "pip_value_per_lot": pip_value,
        "lot_size": lot_size,
    }


def _zone_buffer(pair: str, zone: SupplyDemandZone) -> float:
    thickness = zone.top - zone.bottom
    return max(thickness * 0.10, pip_size(pair) * 2.0)


def _match_zone(zones: list[SupplyDemandZone], zone_dict: dict[str, Any]) -> SupplyDemandZone | None:
    for zone in zones:
        if (
            zone.kind == zone_dict.get("kind", zone.kind)
            and abs(zone.bottom - float(zone_dict["bottom"])) < 1e-9
            and abs(zone.top - float(zone_dict["top"])) < 1e-9
        ):
            return zone
    return None


def build_trade_recommendation(
    zone_summary: dict[str, Any],
    zones: list[SupplyDemandZone],
    account_balance: float,
    risk_pct: float,
) -> dict[str, Any]:
    """
    When price tests a zone, suggest entry at close and SL just beyond the zone.

    Demand test -> long (SL below zone bottom).
    Supply test -> short (SL above zone top).
    """
    pair = zone_summary["symbol"]
    entry = float(zone_summary["close"])

    tested_demand = zone_summary.get("tested_demand_zones") or []
    tested_supply = zone_summary.get("tested_supply_zones") or []

    if tested_demand:
        zone_dict = tested_demand[0]
        zone = _match_zone(zones, {**zone_dict, "kind": "demand"}) or SupplyDemandZone(
            kind="demand",
            bottom=float(zone_dict["bottom"]),
            top=float(zone_dict["top"]),
            formed_at=pd.Timestamp(zone_dict["formed_at"]),
            impulse_at=pd.Timestamp(zone_dict["impulse_at"]),
        )
        stop_loss = zone.bottom - _zone_buffer(pair, zone)
        side = "long"
        setup = "Demand zone test (buy)"
    elif tested_supply:
        zone_dict = tested_supply[0]
        zone = _match_zone(zones, {**zone_dict, "kind": "supply"}) or SupplyDemandZone(
            kind="supply",
            bottom=float(zone_dict["bottom"]),
            top=float(zone_dict["top"]),
            formed_at=pd.Timestamp(zone_dict["formed_at"]),
            impulse_at=pd.Timestamp(zone_dict["impulse_at"]),
        )
        stop_loss = zone.top + _zone_buffer(pair, zone)
        side = "short"
        setup = "Supply zone test (sell)"
    else:
        return {
            "active": False,
            "message": "No active zone test — position sizing not applicable.",
        }

    sizing = calculate_lot_size(account_balance, risk_pct, entry, stop_loss, pair)
    zone_thickness_pips = (zone.top - zone.bottom) / pip_size(pair)

    return {
        "active": True,
        "setup": setup,
        "side": side,
        "entry_price": entry,
        "stop_loss_price": stop_loss,
        "zone_bottom": zone.bottom,
        "zone_top": zone.top,
        "zone_thickness_pips": zone_thickness_pips,
        "risk_amount_usd": sizing["risk_amount_usd"],
        "stop_loss_pips": sizing["stop_loss_pips"],
        "lot_size": sizing["lot_size"],
        "pip_value_per_lot": sizing["pip_value_per_lot"],
    }
