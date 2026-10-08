"""R0.4: isolate queue-model and exchange book-cross fill differences.

This is an exchange-timestamp, zero-order-latency, single-order causal
fixture audit, NOT a full replay parity claim against the original
yue_maker wall-clock simulator.
"""

import argparse
from pathlib import Path

import numpy as np

from hftbacktest.order import FILLED, NEW

from hbt_r0_3_single_order_audit import (
    choose_fixture,
    run_native,
    status_name,
    trace_fill_window,
)


VARIANTS = (
    ("native_queue_native_exchange", "native", "native"),
    ("yue_queue_native_exchange", "yue_strict", "native"),
    ("yue_queue_trade_only_exchange", "yue_strict", "trade_only"),
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("npz", type=Path)
    parser.add_argument("--fixture-horizon-s", type=float, default=10.0)
    parser.add_argument("--qty", type=float, default=0.01)
    parser.add_argument("--tick-size", type=float, default=0.01)
    parser.add_argument("--lot-size", type=float, default=0.01)
    args = parser.parse_args()

    with np.load(args.npz) as z:
        data = z["data"]

    fixture = choose_fixture(data, args.fixture_horizon_s)

    print("===== HBT-R0.4 TRADE-ONLY FILL PARITY =====")
    print("scope=one deterministic BUY fixture")
    print("old_reference=yue_maker_strict_rule_replayed_in_exchange_timestamp")
    print("order_latency=0ns")
    print("NOTE=original_yue_maker_uses_recv_wall_ns_not_exchange_timestamp")
    print(f"entry_local_ts={fixture['entry_local_ts']}")
    print(f"price={fixture['price']:.4f}")
    print(f"exchange_best_bid={fixture['exchange_best_bid']:.4f}")
    print(f"exchange_best_ask={fixture['exchange_best_ask']:.4f}")
    print(
        f"strict_initial_queue_ahead="
        f"{fixture['strict_initial_queue_ahead']:.8f}"
    )
    print(f"old_strict_fill_exch_ts={fixture['strict_fill_exch_ts']}")
    print(f"old_strict_fill_reason={fixture['strict_fill_reason']}")

    results = {}

    for label, queue_model, exchange_model in VARIANTS:
        result = run_native(
            args.npz,
            fixture,
            args.qty,
            args.tick_size,
            args.lot_size,
            queue_model,
            exchange_model,
        )
        results[label] = result

        filled = result["terminal_status"] == FILLED
        delta = (
            result["terminal_exch_ts"] - fixture["strict_fill_exch_ts"]
            if filled else None
        )

        print()
        print(f"===== {label.upper()} =====")
        print(f"queue_model={queue_model}")
        print(f"exchange_model={exchange_model}")
        print(f"initial_status={status_name(result['initial_status'])}")
        print(f"terminal_status={status_name(result['terminal_status'])}")
        print(f"terminal_exch_ts={result['terminal_exch_ts']}")
        print(f"terminal_local_ts={result['terminal_local_ts']}")
        print(f"filled={filled}")
        print(f"delta_from_old_strict_exch_ns={delta}")
        print(f"exec_price={result['exec_price']:.4f}")
        print(f"exec_qty={result['exec_qty']:.8f}")
        print(f"position={result['position']:.8f}")
        print(f"cancel_rc={result['cancel_rc']}")

    native = results["native_queue_native_exchange"]
    strict_native = results["yue_queue_native_exchange"]
    strict_trade = results["yue_queue_trade_only_exchange"]

    print()
    print("===== CAUSAL COMPARISON =====")
    print(
        "queue_switch_changed_fill_timestamp="
        f"{native['terminal_exch_ts'] != strict_native['terminal_exch_ts']}"
    )
    print(
        "exchange_switch_changed_fill_timestamp="
        f"{strict_native['terminal_exch_ts'] != strict_trade['terminal_exch_ts']}"
    )
    print(
        "native_vs_old_strict_ns="
        f"{native['terminal_exch_ts'] - fixture['strict_fill_exch_ts']}"
    )
    print(
        "trade_only_vs_old_strict_ns="
        f"{strict_trade['terminal_exch_ts'] - fixture['strict_fill_exch_ts']}"
    )

    if native["terminal_status"] == FILLED and strict_trade["terminal_status"] == FILLED:
        trace_fill_window(
            data,
            fixture,
            native["terminal_exch_ts"],
            strict_trade["terminal_exch_ts"],
        )

    lifecycle_ok = all(
        result["initial_status"] == NEW
        for result in results.values()
    )
    fixture_parity = (
        strict_trade["terminal_status"] == FILLED
        and strict_trade["terminal_exch_ts"] == fixture["strict_fill_exch_ts"]
        and abs(strict_trade["exec_price"] - fixture["price"]) < 1e-9
        and abs(strict_trade["exec_qty"] - args.qty) < 1e-9
    )

    print()
    print(f"SUBMIT_LIFECYCLE={'PASS' if lifecycle_ok else 'FAIL'}")
    print(f"SINGLE_FIXTURE_PARITY={'PASS' if fixture_parity else 'DIFF'}")
    print(
        "FULL_ENGINE_PARITY=NOT_TESTED "
        "(requires paired multi-order replay, wall-clock alignment, "
        "latency/cancel scenarios and event-order attribution)"
    )

    raise SystemExit(0 if (lifecycle_ok and fixture_parity) else 1)


if __name__ == "__main__":
    main()
