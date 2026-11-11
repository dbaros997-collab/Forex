"""
MetaTrader 5 connection, account info, and order execution.

Requires MetaTrader 5 terminal installed (Windows). Replace symbol mapping if your
broker uses suffixes (e.g. EURUSD.a).
"""

from __future__ import annotations

from typing import Any, Literal

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None  # type: ignore

MT5_AVAILABLE = mt5 is not None

OrderSide = Literal["BUY", "SELL"]


def _require_mt5():
    if mt5 is None:
        return False, "MetaTrader5 package not installed. Run: pip install MetaTrader5"
    return True, ""


def yahoo_symbol_to_mt5(symbol: str) -> str:
    """Map Yahoo tickers (EURUSD=X, GC=F) to typical MT5 symbol names (EURUSD, XAUUSD)."""
    s = symbol.upper().strip()
    if s in {"GC=F", "GC"} or "XAU" in s:
        return "XAUUSD"
    if s.endswith("=X"):
        s = s[:-2]
    return s.replace("/", "")


def login_mt5(account: int | str, password: str, server: str) -> tuple[bool, str]:
    """
    Initialize MT5 and log in to the trading account.

    Returns (True, success_message) or (False, error_message).
    """
    ok, msg = _require_mt5()
    if not ok:
        return False, msg

    if not mt5.initialize():
        err = mt5.last_error()
        return False, f"mt5.initialize() failed: {err}"

    try:
        login_id = int(account)
    except (TypeError, ValueError):
        shutdown_mt5()
        return False, "Account number must be numeric."

    if not mt5.login(login=login_id, password=str(password), server=str(server)):
        err = mt5.last_error()
        shutdown_mt5()
        return False, f"mt5.login() failed: {err}"

    return True, "Connected successfully"


def get_mt5_account_info() -> dict[str, Any] | None:
    """Return balance, equity, leverage, and login from the active MT5 session."""
    ok, msg = _require_mt5()
    if not ok or mt5 is None:
        return None

    info = mt5.account_info()
    if info is None:
        return None

    return {
        "login": int(info.login),
        "balance": float(info.balance),
        "equity": float(info.equity),
        "leverage": int(info.leverage),
        "currency": str(info.currency),
        "server": str(info.server),
    }


def shutdown_mt5() -> None:
    """Shut down the MT5 terminal connection."""
    if mt5 is not None:
        mt5.shutdown()


def _normalize_volume(symbol: str, lot_size: float) -> tuple[float | None, str | None]:
    info = mt5.symbol_info(symbol)
    if info is None:
        return None, f"Symbol {symbol!r} not found on this broker."

    if not info.visible:
        if not mt5.symbol_select(symbol, True):
            return None, f"Could not select symbol {symbol!r}."

    volume_min = float(info.volume_min)
    volume_max = float(info.volume_max)
    volume_step = float(info.volume_step)

    lots = max(volume_min, min(volume_max, float(lot_size)))
    if volume_step > 0:
        steps = round(lots / volume_step)
        lots = steps * volume_step

    lots = round(lots, 8)
    if lots < volume_min:
        return None, f"Lot size {lot_size} below minimum {volume_min} for {symbol}."
    return lots, None


def _normalize_price(symbol: str, price: float) -> float:
    info = mt5.symbol_info(symbol)
    digits = int(info.digits) if info else 5
    return round(float(price), digits)


def place_zone_order(
    symbol: str,
    order_type: OrderSide,
    lot_size: float,
    stop_loss: float,
    take_profit: float,
) -> tuple[bool, str]:
    """
    Send a market order with stop loss and take profit.

    order_type: 'BUY' or 'SELL'
    """
    ok, msg = _require_mt5()
    if not ok:
        return False, msg

    mt5_symbol = yahoo_symbol_to_mt5(symbol)

    volume, vol_err = _normalize_volume(mt5_symbol, lot_size)
    if volume is None:
        return False, vol_err or "Invalid volume."

    tick = mt5.symbol_info_tick(mt5_symbol)
    if tick is None:
        return False, f"No tick data for {mt5_symbol}."

    sl = _normalize_price(mt5_symbol, stop_loss)
    tp = _normalize_price(mt5_symbol, take_profit)

    if order_type.upper() == "BUY":
        trade_type = mt5.ORDER_TYPE_BUY
        price = float(tick.ask)
    elif order_type.upper() == "SELL":
        trade_type = mt5.ORDER_TYPE_SELL
        price = float(tick.bid)
    else:
        return False, f"Unsupported order_type: {order_type!r}"

    filling = mt5.ORDER_FILLING_FOK
    sym_info = mt5.symbol_info(mt5_symbol)
    if sym_info is not None:
        filling_mode = sym_info.filling_mode
        if filling_mode & mt5.SYMBOL_FILLING_IOC:
            filling = mt5.ORDER_FILLING_IOC
        elif filling_mode & mt5.SYMBOL_FILLING_FOK:
            filling = mt5.ORDER_FILLING_FOK
        else:
            filling = mt5.ORDER_FILLING_RETURN

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": mt5_symbol,
        "volume": volume,
        "type": trade_type,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": 20,
        "magic": 260313,
        "comment": "Forex SD zone",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": filling,
    }

    result = mt5.order_send(request)
    if result is None:
        return False, f"order_send failed: {mt5.last_error()}"

    if result.retcode != mt5.TRADE_RETCODE_DONE:
        return False, f"Order rejected: retcode={result.retcode}, comment={result.comment}"

    return True, f"Order placed: ticket #{result.order}, {order_type} {volume} {mt5_symbol}"
