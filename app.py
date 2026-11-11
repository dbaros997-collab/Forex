#!/usr/bin/env python3
"""
Streamlit dashboard for the Forex Supply & Demand scanner.

Run: streamlit run app.py
"""

from __future__ import annotations

import hmac
import os
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Ensure project modules resolve from this folder (not a stale/cached `signals` elsewhere).
_APP_DIR = Path(__file__).resolve().parent
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

from data import fetch_forex_data
from indicators import SupplyDemandZone, calculate_supply_demand_zones, enrich_with_indicators
from position_sizing import build_trade_recommendation
from signals import MT5_AVAILABLE, evaluate_signals, evaluate_zone_signals, get_platform_capabilities
from webhook_bridge import (
    build_execution_bundle,
    send_trade_webhook,
    trade_alert_to_json,
)

if MT5_AVAILABLE:
    from mt5_connect import (
        get_mt5_account_info,
        login_mt5,
        place_zone_order,
        shutdown_mt5,
        yahoo_symbol_to_mt5,
    )
else:

    def yahoo_symbol_to_mt5(symbol: str) -> str:
        s = symbol.upper().strip()
        if s in {"GC=F", "GC"} or "XAU" in s:
            return "XAUUSD"
        if s.endswith("=X"):
            s = s[:-2]
        return s.replace("/", "")

    def login_mt5(*_args: Any, **_kwargs: Any) -> tuple[bool, str]:
        return False, "MetaTrader5 is not available on this host."

    def get_mt5_account_info() -> None:
        return None

    def place_zone_order(*_args: Any, **_kwargs: Any) -> tuple[bool, str]:
        return False, "MetaTrader5 is not available on this host."

    def shutdown_mt5() -> None:
        return None

PAIR_OPTIONS = {
    "XAU/USD": "GC=F",
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "USDJPY=X",
    "AUD/USD": "AUDUSD=X",
    "USD/CHF": "USDCHF=X",
}

MULTI_PAIR_SCAN = [
    ("XAU/USD", "GC=F"),
    ("EUR/USD", "EURUSD=X"),
    ("GBP/USD", "GBPUSD=X"),
    ("USD/JPY", "USDJPY=X"),
    ("GBP/JPY", "GBPJPY=X"),
    ("AUD/USD", "AUDUSD=X"),
]

TIMEFRAME_OPTIONS = {
    "Daily (2 years)": {"interval": "1d", "period": "2y"},
    "Daily (1 year)": {"interval": "1d", "period": "1y"},
    "Weekly (5 years)": {"interval": "1wk", "period": "5y"},
    "1 Hour (60 days)": {"interval": "1h", "period": "60d"},
    "4 Hour (730 days)": {"interval": "1h", "period": "730d"},
}

CANDLE_UP = "#089981"
CANDLE_DOWN = "#F23645"
DEMAND_FILL = "rgba(8, 153, 129, 0.2)"
SUPPLY_FILL = "rgba(242, 54, 69, 0.2)"
DEMAND_LINE = "rgba(8, 153, 129, 0.65)"
SUPPLY_LINE = "rgba(242, 54, 69, 0.65)"


@st.cache_data(ttl=900, show_spinner=False)
def load_market_data(symbol: str, interval: str, period: str) -> pd.DataFrame:
    return fetch_forex_data(symbol, period=period, interval=interval)


@st.cache_data(ttl=900, show_spinner=False)
def load_pair_snapshot(symbol: str, interval: str, period: str) -> dict[str, Any]:
    ohlcv = fetch_forex_data(symbol, period=period, interval=interval)
    zones = calculate_supply_demand_zones(ohlcv)
    zone_summary = evaluate_zone_signals(ohlcv, zones, symbol=symbol)
    ta_summary: dict[str, Any] | None = None
    try:
        enriched = enrich_with_indicators(ohlcv)
        ta_summary = evaluate_signals(enriched, symbol=symbol)
    except Exception:
        ta_summary = None
    return {"ohlcv": ohlcv, "zones": zones, "zone_summary": zone_summary, "ta_summary": ta_summary}


def price_decimals(symbol: str) -> int:
    upper = symbol.upper()
    if "XAU" in upper or upper.startswith("GC"):
        return 2
    if "JPY" in upper:
        return 3
    return 5


def format_price(symbol: str, price: float) -> str:
    return f"{price:.{price_decimals(symbol)}f}"


def _chart_x_end(df: pd.DataFrame) -> pd.Timestamp:
    last = pd.Timestamp(df.index[-1])
    if len(df) < 2:
        return last + timedelta(days=1)
    delta = pd.Timestamp(df.index[-1]) - pd.Timestamp(df.index[-2])
    return last + delta


def build_candlestick_figure(
    df: pd.DataFrame,
    zones: list[SupplyDemandZone],
    symbol: str,
    title: str | None = None,
) -> go.Figure:
    plot_df = df.copy()
    plot_df.index = pd.to_datetime(plot_df.index)
    chart_name = title or symbol
    decimals = price_decimals(symbol)
    hover_tpl = (
        "<b>%{x|%Y-%m-%d %H:%M}</b><br>"
        f"Open: %{{open:.{decimals}f}}<br>"
        f"High: %{{high:.{decimals}f}}<br>"
        f"Low: %{{low:.{decimals}f}}<br>"
        f"Close: %{{close:.{decimals}f}}"
        "<extra></extra>"
    )

    fig = go.Figure(
        data=[
            go.Candlestick(
                x=plot_df.index,
                open=plot_df["Open"],
                high=plot_df["High"],
                low=plot_df["Low"],
                close=plot_df["Close"],
                name=chart_name,
                increasing_line_color=CANDLE_UP,
                increasing_fillcolor=CANDLE_UP,
                decreasing_line_color=CANDLE_DOWN,
                decreasing_fillcolor=CANDLE_DOWN,
                hovertemplate=hover_tpl,
            )
        ]
    )

    x_start = plot_df.index[0]
    x_end = _chart_x_end(plot_df)

    for zone in zones:
        is_demand = zone.kind == "demand"
        fillcolor = DEMAND_FILL if is_demand else SUPPLY_FILL
        line_color = DEMAND_LINE if is_demand else SUPPLY_LINE
        label = "Demand" if is_demand else "Supply"

        fig.add_shape(
            type="rect",
            x0=x_start,
            x1=x_end,
            y0=zone.bottom,
            y1=zone.top,
            xref="x",
            yref="y",
            fillcolor=fillcolor,
            line={"color": line_color, "width": 1},
            layer="below",
        )
        fig.add_annotation(
            x=x_start,
            y=zone.top,
            xref="x",
            yref="y",
            text=f"{label} {zone.bottom:.{decimals}f}-{zone.top:.{decimals}f}",
            showarrow=False,
            xanchor="left",
            yanchor="bottom",
            font={"size": 10, "color": line_color.replace("0.65", "1")},
        )

    fig.update_layout(
        title=f"{chart_name} — Supply & Demand zones",
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_white",
        height=640,
        xaxis_rangeslider_visible=False,
        margin=dict(l=48, r=24, t=72, b=48),
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.06)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.06)", tickformat=f".{decimals}f")
    return fig


def _trend_label(ta: dict[str, Any] | None) -> str:
    if ta is None:
        return "N/A"
    return "Bullish" if ta["sma50_above_sma200"] else "Bearish"


def _zone_status_label(zone_summary: dict[str, Any]) -> str:
    if zone_summary["demand_zone_test"] and zone_summary["supply_zone_test"]:
        return "Testing demand & supply"
    if zone_summary["demand_zone_test"]:
        return "Testing demand"
    if zone_summary["supply_zone_test"]:
        return "Testing supply"
    return "No zone test"


def _render_header_metric(column: Any, label: str, value: str) -> None:
    """Custom metric block — responsive, avoids label/value clipping on mobile."""
    column.markdown(
        f"""
        <div style="
            background: rgba(151, 166, 195, 0.12);
            border-radius: 8px;
            padding: 0.75rem 1rem;
            min-height: 5.25rem;
        ">
            <div style="
                font-size: clamp(0.75rem, 2vw, 0.85rem);
                font-weight: 600;
                color: rgb(49, 51, 63);
                margin-bottom: 0.35rem;
                line-height: 1.35;
                white-space: normal;
                word-wrap: break-word;
                overflow-wrap: anywhere;
            ">{label}</div>
            <div style="
                font-size: clamp(1rem, 3vw, 1.35rem);
                font-weight: 700;
                color: rgb(38, 39, 48);
                line-height: 1.3;
                white-space: normal;
                word-wrap: break-word;
                overflow-wrap: anywhere;
            ">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _expected_credentials() -> tuple[str, str] | None:
    """Read the login from the environment or Streamlit secrets. Never hardcode it."""
    username = os.environ.get("APP_USERNAME", "").strip()
    password = os.environ.get("APP_PASSWORD", "")
    try:
        if not username and "APP_USERNAME" in st.secrets:
            username = str(st.secrets["APP_USERNAME"]).strip()
        if not password and "APP_PASSWORD" in st.secrets:
            password = str(st.secrets["APP_PASSWORD"])
    except Exception:
        pass
    if not username or not password:
        return None
    return username, password


def _credentials_match(username: str, password: str) -> bool:
    expected = _expected_credentials()
    if expected is None:
        return False
    expected_username, expected_password = expected
    return hmac.compare_digest(
        username.strip().encode("utf-8"),
        expected_username.encode("utf-8"),
    ) and hmac.compare_digest(
        password.encode("utf-8"),
        expected_password.encode("utf-8"),
    )


def _render_login_gate() -> None:
    _, center, _ = st.columns([1, 1.15, 1])
    with center:
        st.title("Sign in")
        st.caption("Enter your username and password to open the Forex dashboard.")
        if _expected_credentials() is None:
            st.warning(
                "Login is not configured on this server. "
                "Set APP_USERNAME and APP_PASSWORD before signing in."
            )
        with st.form("login_form", clear_on_submit=False):
            username = st.text_input("Username", autocomplete="username")
            password = st.text_input(
                "Password",
                type="password",
                autocomplete="current-password",
            )
            submitted = st.form_submit_button("Sign in", type="primary", use_container_width=True)
        if submitted:
            if _expected_credentials() is None:
                st.error("Login is not configured on this server.")
            elif _credentials_match(username, password):
                st.session_state.authenticated = True
                st.session_state.auth_username = username.strip()
                st.rerun()
            else:
                st.error("Incorrect username or password.")


def _logout() -> None:
    st.session_state.authenticated = False
    st.session_state.auth_username = ""
    st.rerun()


def _init_app_session_state() -> None:
    defaults = {
        "authenticated": False,
        "auth_username": "",
        "mt5_connected": False,
        "mt5_account_id": None,
        "mt5_status_message": "",
        "mt5_balance": 10000.0,
        "account_balance": 10000.0,
        "webhook_url": "",
        "webhook_secret": "",
        "last_webhook_payload": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _render_platform_banner() -> None:
    caps = get_platform_capabilities()
    if caps.get("mt5_native"):
        st.success("Runtime: cloud dashboard + optional **local MT5** execution (Windows).")
    else:
        st.info(
            "Runtime: **cloud / mobile SaaS mode** — charts, zones, risk sizing, and "
            "**webhook execution** are fully available. Native MT5 is disabled on this host."
        )


def _broker_symbol(symbol: str) -> str:
    return yahoo_symbol_to_mt5(symbol)


def build_multi_pair_table(interval: str, period: str) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for pair_name, symbol in MULTI_PAIR_SCAN:
        try:
            snap = load_pair_snapshot(symbol, interval, period)
            zs = snap["zone_summary"]
            price = format_price(symbol, float(zs["close"]))
            trend = _trend_label(snap["ta_summary"])
            zone_status = _zone_status_label(zs)
        except Exception as exc:
            price = "—"
            trend = "Error"
            zone_status = str(exc)[:80]

        rows.append(
            {
                "Pair": pair_name,
                "Current Price": price,
                "Active Trend": trend,
                "Zone Status": zone_status,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    st.set_page_config(
        page_title="Forex Supply & Demand Scanner",
        page_icon="📊",
        layout="wide",
    )

    _init_app_session_state()
    if not st.session_state.authenticated:
        _render_login_gate()
        return

    st.title("Forex Supply & Demand Scanner")
    st.caption("Cloud-ready supply & demand analysis with remote webhook execution.")

    _render_platform_banner()

    with st.sidebar:
        st.caption(f"Signed in as **{st.session_state.auth_username}**")
        if st.button("Log out", use_container_width=True):
            _logout()
        st.header("Settings")
        pair_label = st.selectbox(
            "Currency pair",
            list(PAIR_OPTIONS.keys()),
            index=0,
            key="currency_pair",
        )
        timeframe_label = st.selectbox("Timeframe", list(TIMEFRAME_OPTIONS.keys()), index=0)
        symbol = PAIR_OPTIONS[pair_label]
        tf = TIMEFRAME_OPTIONS[timeframe_label]

        st.divider()
        st.subheader("Remote execution (webhook)")
        st.session_state.webhook_url = st.text_input(
            "Webhook URL",
            value=st.session_state.webhook_url,
            placeholder="https://your-bridge.example/trade",
            help="Route signals to MetaApi, an MT5 Web API bridge, or your local listener.",
        )
        st.session_state.webhook_secret = st.text_input(
            "Webhook secret (optional)",
            value=st.session_state.webhook_secret,
            type="password",
            help="Sent as X-Webhook-Secret header — not stored in the JSON body.",
        )

        if MT5_AVAILABLE:
            st.divider()
            st.subheader("Local MT5 (Windows PC)")
            st.caption("Local MT5 execution (Windows terminal)")
            mt5_account = st.text_input("Account number", value="", placeholder="12345678")
            mt5_password = st.text_input("Password", type="password")
            mt5_server = st.text_input("Server", value="ICMarkets-Demo", placeholder="Broker-Server")

            connect_col, disconnect_col = st.columns(2)
            with connect_col:
                connect_mt5 = st.button("Connect to MT5", use_container_width=True)
            with disconnect_col:
                disconnect_mt5 = st.button("Disconnect", use_container_width=True)

            if connect_mt5:
                if not mt5_account or not mt5_password or not mt5_server:
                    st.session_state.mt5_connected = False
                    st.session_state.mt5_status_message = "Enter account, password, and server."
                else:
                    success, message = login_mt5(mt5_account, mt5_password, mt5_server)
                    st.session_state.mt5_connected = success
                    st.session_state.mt5_status_message = message
                    if success:
                        info = get_mt5_account_info()
                        if info:
                            st.session_state.mt5_account_id = info["login"]
                            st.session_state.mt5_balance = info["balance"]
                            st.session_state.account_balance = info["balance"]

            if disconnect_mt5:
                shutdown_mt5()
                st.session_state.mt5_connected = False
                st.session_state.mt5_account_id = None
                st.session_state.mt5_status_message = "Disconnected from MT5."

            if st.session_state.mt5_connected:
                info = get_mt5_account_info()
                if info:
                    st.session_state.mt5_balance = info["balance"]
                    st.session_state.account_balance = info["balance"]
                    st.markdown(
                        f"""
                        <div style="
                            background: rgba(8, 153, 129, 0.15);
                            border: 1px solid rgba(8, 153, 129, 0.5);
                            border-radius: 8px;
                            padding: 0.6rem 0.75rem;
                            font-size: 0.9rem;
                        ">
                            <strong style="color: #089981;">MT5 Connected</strong><br>
                            Account: <strong>{info["login"]}</strong><br>
                            Balance: <strong>{info["balance"]:,.2f} {info["currency"]}</strong><br>
                            Equity: {info["equity"]:,.2f}<br>
                            Leverage: 1:{info["leverage"]}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.warning("MT5 session lost. Please connect again.")
                    st.session_state.mt5_connected = False
            elif st.session_state.mt5_status_message:
                st.caption(st.session_state.mt5_status_message)
        else:
            st.info(
                "Local MetaTrader5 is unavailable on this host. "
                "Charts, zones, sizing, and webhook execution still work in the browser."
            )

        st.divider()
        st.subheader("Risk management")
        balance_value = float(st.session_state.account_balance)
        if st.session_state.mt5_connected:
            balance_value = float(st.session_state.mt5_balance)

        account_balance = st.number_input(
            "Account Balance ($)",
            min_value=100.0,
            value=balance_value,
            step=100.0,
            format="%.2f",
            key="account_balance_input",
        )
        st.session_state.account_balance = account_balance

        risk_pct = st.slider(
            "Risk Per Trade (%)",
            min_value=0.5,
            max_value=25.0,
            value=10.0,
            step=0.5,
        )
        refresh = st.button("Refresh data", type="primary", use_container_width=True)

    if refresh:
        load_market_data.clear()
        load_pair_snapshot.clear()

    with st.spinner(f"Loading {pair_label} ({timeframe_label})…"):
        try:
            snap = load_pair_snapshot(symbol, tf["interval"], tf["period"])
        except Exception as exc:
            st.error(f"Could not load market data: {exc}")
            return

    ohlcv = snap["ohlcv"]
    zones = snap["zones"]
    zone_summary = snap["zone_summary"]
    ta_summary = snap["ta_summary"]
    ta_error: str | None = None
    if ta_summary is None:
        ta_error = "Not enough bars for SMA 200 on this timeframe."

    trend = _trend_label(ta_summary)
    signal = _zone_status_label(zone_summary)

    st.markdown("### Overview")
    row_a1, row_a2, row_a3 = st.columns(3, gap="medium")
    row_b1, row_b2 = st.columns(2, gap="medium")
    _render_header_metric(row_a1, "Pair", pair_label)
    _render_header_metric(row_a2, "Price", format_price(symbol, zone_summary["close"]))
    _render_header_metric(row_a3, "Date", zone_summary["as_of"])
    _render_header_metric(row_b1, "Trend", trend)
    _render_header_metric(row_b2, "Signal", signal)

    if ta_error:
        st.info(f"SMA/RSI trend metrics need more history on this timeframe: {ta_error}")

    col_chart, col_panel = st.columns([3, 1], gap="medium")

    with col_chart:
        st.plotly_chart(
            build_candlestick_figure(ohlcv, zones, symbol, title=pair_label),
            use_container_width=True,
        )

    trade = build_trade_recommendation(
        zone_summary,
        zones,
        account_balance=float(account_balance),
        risk_pct=float(risk_pct),
    )

    with col_panel:
        st.subheader("Zone status")
        st.write(f"**Demand:** {zone_summary['demand_zone_status']}")
        st.write(f"**Supply:** {zone_summary['supply_zone_status']}")

        st.subheader("Trade recommendation & sizing")
        if trade.get("active"):
            broker_sym = _broker_symbol(symbol)
            execution = build_execution_bundle(
                trade,
                symbol=symbol,
                account_balance=float(account_balance),
                risk_pct=float(risk_pct),
                broker_symbol=broker_sym,
            )
            order_type = execution["order_type"]
            take_profit = execution["take_profit"]
            alert_payload = execution["payload"]

            st.markdown(
                f"""
                <div style="
                    background: rgba(8, 153, 129, 0.08);
                    border: 1px solid rgba(8, 153, 129, 0.35);
                    border-radius: 8px;
                    padding: 0.85rem 1rem;
                    margin-bottom: 0.75rem;
                    font-size: clamp(0.85rem, 2.5vw, 0.95rem);
                    line-height: 1.55;
                    word-wrap: break-word;
                    overflow-wrap: anywhere;
                ">
                    <div><strong>Setup:</strong> {trade["setup"]}</div>
                    <div><strong>Order type:</strong> {order_type}</div>
                    <div><strong>Entry price:</strong> {format_price(symbol, trade["entry_price"])}</div>
                    <div><strong>Stop loss:</strong> {format_price(symbol, trade["stop_loss_price"])}</div>
                    <div><strong>Take profit:</strong> {format_price(symbol, take_profit)}</div>
                    <div><strong>Stop distance:</strong> {trade["stop_loss_pips"]:.1f} pips</div>
                    <div><strong>Account risk ({risk_pct:.1f}%):</strong> ${trade["risk_amount_usd"]:,.2f}</div>
                    <div><strong>Lot size:</strong> {trade["lot_size"]:.2f} standard lots</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.caption(
                f"SL placed beyond zone ({format_price(symbol, trade['zone_bottom'])} - "
                f"{format_price(symbol, trade['zone_top'])}). "
                f"Broker symbol: {broker_sym}."
            )
            with st.expander("Preview webhook JSON"):
                st.code(trade_alert_to_json(alert_payload), language="json")

            st.markdown("**Execute signal**")
            webhook_col, mt5_col = st.columns(2)
            with webhook_col:
                send_webhook = st.button(
                    "Send Webhook Signal",
                    use_container_width=True,
                    type="primary",
                    help="For cloud, mobile, and tablet users.",
                )
            with mt5_col:
                send_mt5 = False
                if MT5_AVAILABLE:
                    send_mt5 = st.button(
                        "Execute on MT5",
                        use_container_width=True,
                        disabled=not st.session_state.mt5_connected,
                        help="Windows PC with MT5 terminal connected.",
                    )

            if send_webhook:
                with st.spinner("Posting trade alert…"):
                    ok, hook_msg = send_trade_webhook(
                        st.session_state.webhook_url,
                        alert_payload,
                        secret=st.session_state.webhook_secret or None,
                    )
                if ok:
                    st.session_state.last_webhook_payload = trade_alert_to_json(alert_payload)
                    st.success(hook_msg)
                else:
                    st.error(hook_msg)

            if MT5_AVAILABLE and send_mt5:
                if not st.session_state.mt5_connected:
                    st.warning("Connect MT5 in the sidebar before local execution.")
                else:
                    with st.spinner("Sending order to MT5…"):
                        ok, exec_msg = place_zone_order(
                            symbol=symbol,
                            order_type=order_type,
                            lot_size=float(trade["lot_size"]),
                            stop_loss=float(trade["stop_loss_price"]),
                            take_profit=take_profit,
                        )
                    if ok:
                        st.success(exec_msg)
                    else:
                        st.error(exec_msg)
            elif MT5_AVAILABLE and not st.session_state.mt5_connected:
                st.caption("Connect MT5 in the sidebar to enable local execution on this PC.")
            elif not MT5_AVAILABLE:
                st.caption("Use **Send Webhook Signal** to execute from any device.")
        else:
            st.info(trade.get("message", "No trade setup available."))

        st.subheader("Active demand")
        if zone_summary["active_demand_zones"]:
            for z in zone_summary["active_demand_zones"]:
                st.write(
                    f"{format_price(symbol, z['bottom'])} - {format_price(symbol, z['top'])} "
                    f"({z['impulse_at']})"
                )
        else:
            st.write("None")

        st.subheader("Active supply")
        if zone_summary["active_supply_zones"]:
            for z in zone_summary["active_supply_zones"]:
                st.write(
                    f"{format_price(symbol, z['bottom'])} - {format_price(symbol, z['top'])} "
                    f"({z['impulse_at']})"
                )
        else:
            st.write("None")

        if ta_summary:
            st.subheader("Indicators")
            st.write(f"RSI (14): **{ta_summary['rsi_14']:.2f}**")
            st.write(f"SMA 50: `{format_price(symbol, ta_summary['sma_50'])}`")
            st.write(f"SMA 200: `{format_price(symbol, ta_summary['sma_200'])}`")
            st.write(f"Golden cross: {ta_summary['golden_cross_status']}")
            st.write(f"Death cross: {ta_summary['death_cross_status']}")

    st.markdown("---")
    st.subheader("Multi-pair summary")
    st.caption(f"Same timeframe as sidebar: {timeframe_label}")

    with st.spinner("Scanning major pairs…"):
        summary_df = build_multi_pair_table(tf["interval"], tf["period"])

    st.dataframe(
        summary_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Pair": st.column_config.TextColumn("Pair", width="small"),
            "Current Price": st.column_config.TextColumn("Current Price", width="medium"),
            "Active Trend": st.column_config.TextColumn("Active Trend", width="small"),
            "Zone Status": st.column_config.TextColumn("Zone Status", width="large"),
        },
    )


if __name__ == "__main__":
    main()
