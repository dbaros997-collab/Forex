#!/usr/bin/env python3
"""
Forex technical analysis scanner — main orchestrator.

Flow: fetch_forex_data -> enrich_with_indicators -> evaluate_signals /
      calculate_supply_demand_zones -> evaluate_zone_signals -> print report.

Backtesting: loop over index i, pass df.iloc[: i + 1] to evaluate_signals() and
evaluate_zone_signals(), and record rows where crosses or zone tests occur.
Swap fetch_forex_data() in data.py for CSV or a custom REST API.
"""

from __future__ import annotations

import argparse
import sys

from data import fetch_forex_data
from indicators import calculate_supply_demand_zones, enrich_with_indicators
from signals import evaluate_signals, evaluate_zone_signals, format_zone_summary


def print_scan_report(summary: dict) -> None:
    trend = "YES" if summary["sma50_above_sma200"] else "NO"
    print(f"=== Forex TA Scan: {summary['symbol']} ===")
    print(f"As of: {summary['as_of']}")
    print(f"Close: {summary['close']:.5f}")
    print(f"SMA 50: {summary['sma_50']:.5f}")
    print(f"SMA 200: {summary['sma_200']:.5f}")
    print(f"RSI (14): {summary['rsi_14']:.2f}")
    print(f"SMA50 > SMA200 (trend): {trend}")
    print("--- Rule conditions (latest bar) ---")
    print(f"Golden cross: {summary['golden_cross_status']}")
    print(f"Death cross: {summary['death_cross_status']}")


def _print_zone_line(label: str, zone: dict | None) -> None:
    if zone is None:
        print(f"{label}: none")
        return
    print(
        f"{label}: {zone['bottom']:.5f} - {zone['top']:.5f} "
        f"(impulse {zone['impulse_at']})"
    )


def print_zone_report(zone_summary: dict) -> None:
    print()
    print(f"=== Supply / Demand Scan: {zone_summary['symbol']} ===")
    print(f"As of: {zone_summary['as_of']}")
    print(f"Close: {zone_summary['close']:.5f}  (H: {zone_summary['high']:.5f}, L: {zone_summary['low']:.5f})")
    print("--- Active zones ---")
    print(f"Demand zones ({len(zone_summary['active_demand_zones'])}):")
    if zone_summary["active_demand_zones"]:
        for z in zone_summary["active_demand_zones"]:
            print(f"  {z['bottom']:.5f} - {z['top']:.5f}  impulse {z['impulse_at']}")
    else:
        print("  (none)")
    print(f"Supply zones ({len(zone_summary['active_supply_zones'])}):")
    if zone_summary["active_supply_zones"]:
        for z in zone_summary["active_supply_zones"]:
            print(f"  {z['bottom']:.5f} - {z['top']:.5f}  impulse {z['impulse_at']}")
    else:
        print("  (none)")
    print("--- Nearest zones ---")
    _print_zone_line("Nearest demand", zone_summary["nearest_demand"])
    _print_zone_line("Nearest supply", zone_summary["nearest_supply"])
    print("--- Zone test (latest bar) ---")
    print(f"Demand pullback test: {zone_summary['demand_zone_status']}")
    print(f"Supply rally test:   {zone_summary['supply_zone_status']}")
    print(f"Summary: {format_zone_summary(zone_summary)}")


def run(symbol: str = "EURUSD=X", period: str = "2y", interval: str = "1d") -> dict:
    ohlcv = fetch_forex_data(symbol, period=period, interval=interval)
    df = enrich_with_indicators(ohlcv)
    zones = calculate_supply_demand_zones(ohlcv)

    summary = evaluate_signals(df, symbol=symbol)
    zone_summary = evaluate_zone_signals(ohlcv, zones, symbol=symbol)

    print_scan_report(summary)
    print_zone_report(zone_summary)

    return {"ta": summary, "zones": zone_summary}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Forex TA scanner")
    parser.add_argument("--symbol", default="EURUSD=X", help="Yahoo Finance ticker")
    parser.add_argument("--period", default="2y", help="Download history length")
    parser.add_argument("--interval", default="1d", help="Bar interval")
    args = parser.parse_args(argv)

    try:
        run(symbol=args.symbol, period=args.period, interval=args.interval)
        return 0
    except Exception as exc:
        print(f"Scan failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
