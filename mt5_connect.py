"""
MetaTrader 5 connection, account info, and order execution.

Requires MetaTrader 5 terminal installed (Windows). Replace symbol mapping if your
broker uses suffixes (e.g. EURUSD.a).

Credentials and terminal options come from the environment (or from arguments
passed by the dashboard). Account credentials are not hardcoded:

  MT5_ACCOUNT         numeric login
  MT5_PASSWORD        terminal password; never logged or returned
  MT5_SERVER          broker server name
  MT5_TERMINAL_PATH   optional full path to terminal64.exe
  MT5_DEVIATION       max slippage in points for market orders (default 20)
  MT5_MAGIC           magic number stamped on orders from this app (default 260313)

Chart candles elsewhere in the app are cached history. A market order is priced
from the terminal's live bid/ask (see latest_quote), which can differ from that
history.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None  # type: ignore

MT5_AVAILABLE = mt5 is not None

OrderSide = Literal["BUY", "SELL"]

_INIT_TIMEOUT_MS = 15_000
_DEFAULT_DEVIATION = 20
_DEFAULT_MAGIC = 260313


def _env(name: str) -> str:
    return os.environ.get(name, "").strip()


def _parse_bounded_int(name: str, default: int, low: int, high: int) -> tuple[int, str | None]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default, None
    try:
        value = int(raw)
    except ValueError:
        return default, f"{name} must be an integer; using {default}."
    if not low <= value <= high:
        clamped = max(low, min(high, value))
        return clamped, f"{name}={value} is outside {low}..{high}; using {clamped}."
    return value, None


def connection_settings_from_env() -> dict[str, Any]:
    """Non-secret view of MT5 settings. The password is only a yes/no flag."""
    deviation, deviation_error = _parse_bounded_int("MT5_DEVIATION", _DEFAULT_DEVIATION, 0, 1000)
    magic, magic_error = _parse_bounded_int("MT5_MAGIC", _DEFAULT_MAGIC, 1, 2_147_483_647)
    path = _env("MT5_TERMINAL_PATH")
    errors = [msg for msg in (deviation_error, magic_error) if msg]
    if path and not Path(path).is_file():
        errors.append(f"MT5_TERMINAL_PATH is not an existing file: {path}")
    return {
        "account": _env("MT5_ACCOUNT"),
        "server": _env("MT5_SERVER"),
        "terminal_path": path,
        "password_configured": bool(os.environ.get("MT5_PASSWORD", "").strip()),
        "deviation": deviation,
        "magic": magic,
        "config_errors": errors,
    }


def _resolve_credentials(
    account: int | str | None,
    password: str | None,
    server: str | None,
) -> tuple[str, str, str, str]:
    """Form values win. Blank fields fall back to the environment."""
    account_text = "" if account is None else str(account).strip()
    server_text = "" if server is None else str(server).strip()
    if password is None or not str(password).strip():
        password_text = os.environ.get("MT5_PASSWORD", "")
    else:
        password_text = str(password)
    path = _env("MT5_TERMINAL_PATH")
    return account_text or _env("MT5_ACCOUNT"), password_text, server_text or _env("MT5_SERVER"), path


def _validate_credentials(account: str, password: str, server: str, path: str) -> tuple[bool, str]:
    if not account or not password.strip() or not server:
        return (
            False,
            "MT5 account, password, and server are required. "
            "Enter them in the sidebar or set MT5_ACCOUNT, MT5_PASSWORD, and MT5_SERVER.",
        )
    if not account.isdigit():
        return False, "Account number must be numeric."
    if len(server) > 64:
        return False, "Server name is too long."
    if path and not Path(path).is_file():
        return False, f"MT5_TERMINAL_PATH is not an existing file: {path}"
    return True, ""


def _format_last_error() -> str:
    if mt5 is None:
        return "MetaTrader5 package not installed."
    try:
        return str(mt5.last_error())
    except Exception:
        return "unknown terminal error"


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


def _initialize_terminal(path: str) -> str | None:
    """Attach to the local terminal. Returns an error string, or None on success."""
    kwargs: dict[str, Any] = {"timeout": _INIT_TIMEOUT_MS}
    if path:
        kwargs["path"] = path
    try:
        if mt5.initialize(**kwargs):
            return None
        # A leftover session can make the first initialize() fail. Reset once.
        mt5.shutdown()
        if mt5.initialize(**kwargs):
            return None
    except Exception:
        try:
            mt5.shutdown()
        except Exception:
            pass
        return (
            "mt5.initialize() raised an unexpected error. "
            "Check MT5_TERMINAL_PATH and that the terminal is installed."
        )
    return f"mt5.initialize() failed: {_format_last_error()}"


def login_mt5(
    account: int | str | None = None,
    password: str | None = None,
    server: str | None = None,
) -> tuple[bool, str]:
    """
    Initialize MT5 and log in to the trading account.

    Blank arguments fall back to MT5_ACCOUNT, MT5_PASSWORD, and MT5_SERVER.
    The password is never included in the returned message.

    Returns (True, success_message) or (False, error_message).
    """
    ok, msg = _require_mt5()
    if not ok:
        return False, msg

    resolved_account, resolved_password, resolved_server, path = _resolve_credentials(
        account, password, server
    )
    valid, reason = _validate_credentials(
        resolved_account, resolved_password, resolved_server, path
    )
    if not valid:
        return False, reason

    init_error = _initialize_terminal(path)
    if init_error:
        return False, init_error

    try:
        logged_in = mt5.login(
            login=int(resolved_account),
            password=resolved_password,
            server=resolved_server,
        )
    except Exception:
        shutdown_mt5()
        return False, "MT5 login failed because of an unexpected terminal error."

    if not logged_in:
        err = _format_last_error()
        shutdown_mt5()
        return False, f"mt5.login() failed: {err}"

    info = mt5.account_info()
    if info is None:
        shutdown_mt5()
        return False, "Login returned success but no account information is available."

    return True, f"Connected to {info.server} as {info.login}."


def get_mt5_account_info() -> dict[str, Any] | None:
    """Return balance, equity, leverage, and login from the active MT5 session."""
    ok, _msg = _require_mt5()
    if not ok or mt5 is None:
        return None

    try:
        info = mt5.account_info()
    except Exception:
        return None
    if info is None:
        return None

    return {
        "login": int(info.login),
        "balance": float(info.balance),
        "equity": float(info.equity),
        "leverage": int(info.leverage),
        "currency": str(info.currency),
        "server": str(info.server),
        "trade_allowed": bool(info.trade_allowed),
    }


def trading_readiness() -> tuple[bool, str]:
    """
    Confirm the open session can send a live market order.

    Algo trading must be enabled in the terminal, and the account itself must
    allow trading. Call this before place_zone_order; that function checks again.
    """
    ok, msg = _require_mt5()
    if not ok:
        return False, msg

    try:
        terminal = mt5.terminal_info()
        account = mt5.account_info()
    except Exception:
        return False, "Could not read MT5 session status."

    if terminal is None or not terminal.connected or account is None:
        return False, "MT5 session lost. Connect again before sending an order."
    if not terminal.trade_allowed:
        return (
            False,
            "Algo trading is disabled in the MT5 terminal. "
            "Turn on Algo Trading before sending an order.",
        )
    if not account.trade_allowed:
        return False, "Trading is disabled on this account."
    return True, "MT5 session is ready to send orders."


def latest_quote(symbol: str) -> dict[str, Any] | None:
    """
    Current bid/ask from the connected terminal.

    This is the live price a market order will use. Dashboard candles are a
    separate cached history feed and are not this stream.
    """
    ok, _msg = _require_mt5()
    if not ok or mt5 is None:
        return None

    mt5_symbol = yahoo_symbol_to_mt5(symbol)
    try:
        info = mt5.symbol_info(mt5_symbol)
        if info is not None and not info.visible:
            mt5.symbol_select(mt5_symbol, True)
        tick = mt5.symbol_info_tick(mt5_symbol)
    except Exception:
        return None
    if tick is None:
        return None
    return {
        "symbol": mt5_symbol,
        "bid": float(tick.bid),
        "ask": float(tick.ask),
        "time": int(tick.time),
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


def _stops_error(order_type: str, price: float, sl: float, tp: float, info: Any) -> str | None:
    """Reject stops that are on the wrong side of the live price or inside the broker minimum."""
    if sl <= 0 or tp <= 0:
        return "Stop loss and take profit must be positive prices."

    point = float(getattr(info, "point", 0.0) or 0.0)
    min_distance = float(getattr(info, "trade_stops_level", 0) or 0) * point
    if order_type == "BUY":
        if not sl < price < tp:
            return (
                f"BUY stops must sit around the live ask {price}: "
                f"stop loss {sl} below it and take profit {tp} above it. "
                "The chart price can differ from the terminal quote."
            )
        if (price - sl) < min_distance or (tp - price) < min_distance:
            return f"Stops are closer than the broker minimum of {info.trade_stops_level} points."
        return None

    if not tp < price < sl:
        return (
            f"SELL stops must sit around the live bid {price}: "
            f"take profit {tp} below it and stop loss {sl} above it. "
            "The chart price can differ from the terminal quote."
        )
    if (sl - price) < min_distance or (price - tp) < min_distance:
        return f"Stops are closer than the broker minimum of {info.trade_stops_level} points."
    return None


def _trade_mode_error(order_type: str, info: Any) -> str | None:
    mode = int(info.trade_mode)
    if mode == mt5.SYMBOL_TRADE_MODE_DISABLED:
        return f"Trading is disabled for {info.name}."
    if mode == mt5.SYMBOL_TRADE_MODE_CLOSEONLY:
        return f"{info.name} is close-only. New market orders are not accepted."
    if order_type == "BUY" and mode == mt5.SYMBOL_TRADE_MODE_SHORTONLY:
        return f"{info.name} is short-only."
    if order_type == "SELL" and mode == mt5.SYMBOL_TRADE_MODE_LONGONLY:
        return f"{info.name} is long-only."
    return None


def place_zone_order(
    symbol: str,
    order_type: OrderSide,
    lot_size: float,
    stop_loss: float,
    take_profit: float,
) -> tuple[bool, str]:
    """
    Send a market order with stop loss and take profit.

    The fill price is the live ask for BUY and the live bid for SELL, not the
    close on the cached chart. order_type is 'BUY' or 'SELL'.
    """
    ok, msg = _require_mt5()
    if not ok:
        return False, msg

    ready, ready_msg = trading_readiness()
    if not ready:
        return False, ready_msg

    side = order_type.upper()
    if side not in {"BUY", "SELL"}:
        return False, f"Unsupported order_type: {order_type!r}"
    if lot_size <= 0:
        return False, "Lot size must be greater than zero."

    mt5_symbol = yahoo_symbol_to_mt5(symbol)
    settings = connection_settings_from_env()

    try:
        volume, vol_err = _normalize_volume(mt5_symbol, lot_size)
        if volume is None:
            return False, vol_err or "Invalid volume."

        # Live tick from the terminal. Do not substitute the chart's last close.
        tick = mt5.symbol_info_tick(mt5_symbol)
        if tick is None:
            return False, f"No live tick for {mt5_symbol}. Confirm the symbol is in Market Watch."

        sym_info = mt5.symbol_info(mt5_symbol)
        if sym_info is None:
            return False, f"Symbol {mt5_symbol!r} not found on this broker."

        mode_error = _trade_mode_error(side, sym_info)
        if mode_error:
            return False, mode_error

        sl = _normalize_price(mt5_symbol, stop_loss)
        tp = _normalize_price(mt5_symbol, take_profit)
        if side == "BUY":
            trade_type = mt5.ORDER_TYPE_BUY
            price = float(tick.ask)
        else:
            trade_type = mt5.ORDER_TYPE_SELL
            price = float(tick.bid)

        stops_error = _stops_error(side, price, sl, tp, sym_info)
        if stops_error:
            return False, stops_error

        # Prefer the filling mode the symbol advertises, otherwise RETURN.
        filling = mt5.ORDER_FILLING_FOK
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
            "deviation": settings["deviation"],
            "magic": settings["magic"],
            "comment": "Forex SD zone",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": filling,
        }
        result = mt5.order_send(request)
    except Exception:
        return False, "order_send failed because of an unexpected terminal error."

    if result is None:
        return False, f"order_send failed: {_format_last_error()}"

    if result.retcode != mt5.TRADE_RETCODE_DONE:
        return False, f"Order rejected: retcode={result.retcode}, comment={result.comment}"

    return True, (
        f"Order placed: ticket #{result.order}, {side} {volume} {mt5_symbol} "
        f"at {price} (live terminal price)"
    )
