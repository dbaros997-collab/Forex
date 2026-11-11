"""
Webhook trade-alert packaging and delivery (cloud / mobile execution bridge).

Works on any host — no MetaTrader5 dependency.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

try:
    import requests
except ImportError:
    requests = None  # type: ignore

DEFAULT_SOURCE = "forex-supply-demand-scanner"
_USER_AGENT = "ForexSDScanner/1.0"


def validate_webhook_url(webhook_url: str) -> tuple[bool, str]:
    """Basic URL validation before sending live orders."""
    url = (webhook_url or "").strip()
    if not url:
        return False, "Webhook URL is empty."

    parsed = urlparse(url)
    if parsed.scheme not in ("https", "http"):
        return False, "Webhook URL must start with http:// or https://."
    if not parsed.netloc:
        return False, "Webhook URL is missing a host name."

    return True, url


def resolve_order_legs(trade: dict[str, Any]) -> tuple[str, float, str]:
    """
    Map zone recommendation to order type and take-profit target.

    Returns (order_type, take_profit, side_label).
    """
    if trade.get("side") == "long":
        return "BUY", float(trade["zone_top"]), "long"
    return "SELL", float(trade["zone_bottom"]), "short"


def build_execution_bundle(
    trade: dict[str, Any],
    *,
    symbol: str,
    account_balance: float,
    risk_pct: float,
    broker_symbol: str | None = None,
) -> dict[str, Any]:
    """
    Single bundle for UI + webhook: order legs, sizing metadata, JSON payload.
    """
    order_type, take_profit, _side = resolve_order_legs(trade)
    payload = package_trade_alert(
        trade,
        symbol=symbol,
        take_profit=take_profit,
        broker_symbol=broker_symbol,
        account_balance=account_balance,
        risk_pct=risk_pct,
    )
    return {
        "order_type": order_type,
        "take_profit": take_profit,
        "payload": payload,
        "broker_symbol": broker_symbol or symbol,
    }


def package_trade_alert(
    trade: dict[str, Any],
    *,
    symbol: str,
    take_profit: float | None = None,
    broker_symbol: str | None = None,
    account_balance: float | None = None,
    risk_pct: float | None = None,
) -> dict[str, Any]:
    """
    Build a JSON-serializable trade alert for remote execution listeners.

    Includes Symbol, Order Type, Entry, Stop Loss, Take Profit, Lot Size,
    and risk metadata (e.g. 10% account risk sizing from the dashboard).
    """
    if not trade.get("active"):
        raise ValueError("No active trade recommendation to package.")

    order_type = "BUY" if trade.get("side") == "long" else "SELL"
    resolved_symbol = broker_symbol or symbol
    entry = float(trade["entry_price"])
    stop_loss = float(trade["stop_loss_price"])
    lot_size = round(float(trade["lot_size"]), 2)
    tp = float(take_profit) if take_profit is not None else None

    payload: dict[str, Any] = {
        "Symbol": resolved_symbol,
        "Type": order_type,
        "OrderType": order_type,
        "Entry": entry,
        "EntryPrice": entry,
        "StopLoss": stop_loss,
        "TakeProfit": tp,
        "LotSize": lot_size,
        "RiskUsd": float(trade["risk_amount_usd"]),
        "RiskPercent": float(risk_pct) if risk_pct is not None else None,
        "AccountBalance": float(account_balance) if account_balance is not None else None,
        "StopLossPips": float(trade["stop_loss_pips"]),
        "Setup": trade.get("setup"),
        "Side": trade.get("side"),
        "YahooSymbol": symbol,
        "TimestampUtc": datetime.now(timezone.utc).isoformat(),
        "Source": DEFAULT_SOURCE,
        "SchemaVersion": 1,
    }
    return payload


def trade_alert_to_json(payload: dict[str, Any]) -> str:
    """Pretty JSON string for logging or UI preview."""
    return json.dumps(payload, indent=2, sort_keys=True)


def send_trade_webhook(
    webhook_url: str,
    payload: dict[str, Any],
    *,
    secret: str | None = None,
    timeout_seconds: int = 20,
) -> tuple[bool, str]:
    """
    POST trade alert JSON to an external listener (MetaApi, MT5 Web API, local bot).

    Optional shared secret is sent as header `X-Webhook-Secret` (not embedded in JSON).
    """
    valid, url_or_msg = validate_webhook_url(webhook_url)
    if not valid:
        return False, url_or_msg
    url = url_or_msg

    if requests is None:
        return False, "requests library is not installed."

    headers = {
        "Content-Type": "application/json",
        "User-Agent": _USER_AGENT,
        "X-Source": DEFAULT_SOURCE,
    }
    secret_value = (secret or "").strip()
    if secret_value:
        headers["X-Webhook-Secret"] = secret_value

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=timeout_seconds)
    except requests.RequestException as exc:
        return False, f"Webhook request failed: {exc}"

    if response.ok:
        return True, f"Webhook delivered (HTTP {response.status_code})."

    body = re.sub(r"\s+", " ", (response.text or ""))[:300]
    return False, f"Webhook rejected: HTTP {response.status_code} — {body}"
