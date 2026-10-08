import argparse
from pathlib import Path

import numpy as np

from hftbacktest import (
    BacktestAsset,
    HashMapMarketDepthBacktest,
    BUY,
    BUY_EVENT,
    SELL_EVENT,
    DEPTH_EVENT,
    DEPTH_CLEAR_EVENT,
    DEPTH_SNAPSHOT_EVENT,
    TRADE_EVENT,
    EXCH_EVENT,
    LOCAL_EVENT,
    GTX,
    LIMIT,
)
from hftbacktest.order import NEW, FILLED, CANCELED, EXPIRED


BASE_EVENT_MASK = (1 << 28) - 1
NS_PER_S = 1_000_000_000


def apply_depth(row, bids, asks):
    ev = int(row["ev"])
    base = ev & BASE_EVENT_MASK

    is_buy = (ev & BUY_EVENT) != 0
    is_sell = (ev & SELL_EVENT) != 0

    if base == DEPTH_CLEAR_EVENT:
        # This R0 dataset has only the bootstrap snapshot clear.
        if is_buy:
            bids.clear()
        elif is_sell:
            asks.clear()
        return

    if base not in (DEPTH_EVENT, DEPTH_SNAPSHOT_EVENT):
        return

    px = float(row["px"])
    qty = float(row["qty"])

    if is_buy:
        side = bids
    elif is_sell:
        side = asks
    else:
        return

    if qty == 0.0:
        side.pop(px, None)
    else:
        side[px] = qty


def local_best_bid_groups(local_events):
    bids = {}
    asks = {}
    out = []

    i = 0
    n = len(local_events)

    while i < n:
        ts = int(local_events[i]["local_ts"])
        j = i

        while j < n and int(local_events[j]["local_ts"]) == ts:
            apply_depth(local_events[j], bids, asks)
            j += 1

        if bids and asks:
            best_bid = max(bids)
            best_ask = min(asks)
            out.append((ts, best_bid, bids[best_bid], best_ask))

        i = j

    return out


def strict_buy_fill(exch_events, start_idx, price, queue_ahead, deadline):
    """
    Frozen yue_maker VirtualOrder BUY semantics:
    - cancellations/depth reductions never improve queue position;
    - only seller-aggressive same-price trades consume queue ahead;
    - seller-aggressive trade below the resting bid is trade-through;
    - fill when same-price cumulative aggressive qty consumes queue_ahead.
    """
    q = float(queue_ahead)

    for k in range(start_idx, len(exch_events)):
        row = exch_events[k]
        ts = int(row["exch_ts"])

        if ts > deadline:
            break

        ev = int(row["ev"])
        base = ev & BASE_EVENT_MASK

        if base != TRADE_EVENT:
            continue

        if (ev & SELL_EVENT) == 0:
            continue

        px = float(row["px"])
        qty = float(row["qty"])

        if px < price and not np.isclose(px, price, rtol=0.0, atol=1e-12):
            return ts, "trade_through"

        if np.isclose(px, price, rtol=0.0, atol=1e-12):
            q -= qty
            if q <= 0.0:
                return ts, "queue_consumed"

    return None, None


def choose_fixture(data, horizon_s):
    local_events = data[(data["ev"] & LOCAL_EVENT) == LOCAL_EVENT]
    exch_events = data[(data["ev"] & EXCH_EVENT) == EXCH_EVENT]

    local_groups = local_best_bid_groups(local_events)
    if not local_groups:
        raise RuntimeError("no local book groups")

    bids = {}
    asks = {}
    exch_i = 0

    first_ts = local_groups[0][0]
    not_before = first_ts + 5 * NS_PER_S
    horizon_ns = int(horizon_s * NS_PER_S)

    for group_no, (local_ts, local_bid, _, _) in enumerate(local_groups):
        while exch_i < len(exch_events) and int(exch_events[exch_i]["exch_ts"]) <= local_ts:
            apply_depth(exch_events[exch_i], bids, asks)
            exch_i += 1

        if local_ts < not_before:
            continue

        # Search sparsely; this is only deterministic fixture selection,
        # not a trading strategy or alpha rule.
        if group_no % 10 != 0:
            continue

        if not bids or not asks:
            continue

        exch_best_bid = max(bids)
        exch_best_ask = min(asks)

        # Keep the fixture simple: the locally visible best bid must still
        # be the exchange-side best bid when the zero-latency order arrives.
        if not np.isclose(local_bid, exch_best_bid, rtol=0.0, atol=1e-12):
            continue

        if local_bid >= exch_best_ask:
            continue

        queue_ahead = float(bids.get(local_bid, 0.0))
        if queue_ahead <= 0.0:
            continue

        deadline = local_ts + horizon_ns
        fill_ts, reason = strict_buy_fill(
            exch_events,
            exch_i,
            local_bid,
            queue_ahead,
            deadline,
        )

        if fill_ts is None:
            continue

        return {
            "entry_local_ts": int(local_ts),
            "price": float(local_bid),
            "exchange_best_bid": float(exch_best_bid),
            "exchange_best_ask": float(exch_best_ask),
            "strict_initial_queue_ahead": queue_ahead,
            "strict_fill_exch_ts": int(fill_ts),
            "strict_fill_reason": reason,
            "deadline": int(deadline),
        }

    raise RuntimeError(
        f"no deterministic BUY fixture with strict fill inside {horizon_s}s"
    )


def advance_to_timestamp(hbt, target_ts):
    while True:
        rc = hbt.wait_next_feed(False, 60 * NS_PER_S)

        if rc == 1:
            raise RuntimeError("end of data before fixture entry")

        if rc not in (0, 2):
            raise RuntimeError(f"unexpected rc while advancing: {rc}")

        if hbt.current_timestamp == target_ts:
            return

        if hbt.current_timestamp > target_ts:
            raise RuntimeError(
                "HBT skipped fixture entry timestamp: "
                f"target={target_ts} actual={hbt.current_timestamp}"
            )


def run_native(npz, fixture, qty, tick_size, lot_size, queue_model):
    asset = (
        BacktestAsset()
        .data(str(npz))
        .linear_asset(1.0)
        .constant_order_latency(0, 0)
    )
    if queue_model == "native":
        asset = asset.risk_adverse_queue_model()
    elif queue_model == "yue_strict":
        asset = asset.yue_strict_queue_model()
    else:
        raise ValueError(queue_model)

    asset = (
        asset
        .no_partial_fill_exchange()
        .trading_value_fee_model(0.0, 0.0)
        .tick_size(tick_size)
        .lot_size(lot_size)
    )

    hbt = HashMapMarketDepthBacktest([asset])
    order_id = 1

    try:
        advance_to_timestamp(hbt, fixture["entry_local_ts"])

        depth = hbt.depth(0)
        local_bid_at_submit = float(depth.best_bid)
        local_ask_at_submit = float(depth.best_ask)

        rc = hbt.submit_buy_order(
            0,
            order_id,
            fixture["price"],
            qty,
            GTX,
            LIMIT,
            True,
        )

        if rc != 0:
            raise RuntimeError(f"submit_buy_order rc={rc}")

        order = hbt.orders(0).get(order_id)
        if order is None:
            raise RuntimeError("order missing after submit response")

        initial_status = int(order.status)
        initial_exch_ts = int(order.exch_timestamp)

        if initial_status != NEW:
            return {
                "initial_status": initial_status,
                "initial_exch_ts": initial_exch_ts,
                "local_bid_at_submit": local_bid_at_submit,
                "local_ask_at_submit": local_ask_at_submit,
                "terminal_status": initial_status,
                "terminal_exch_ts": initial_exch_ts,
                "terminal_local_ts": int(hbt.current_timestamp),
                "exec_price": float(order.exec_price),
                "exec_qty": float(order.exec_qty),
                "position": float(hbt.position(0)),
                "cancel_rc": None,
            }

        watch_until = max(
            fixture["deadline"],
            fixture["strict_fill_exch_ts"] + NS_PER_S,
        )

        while hbt.current_timestamp < watch_until:
            remaining = watch_until - hbt.current_timestamp
            rc = hbt.wait_next_feed(True, min(remaining, NS_PER_S))

            order = hbt.orders(0).get(order_id)
            if order is None:
                raise RuntimeError("order disappeared from local order map")

            if int(order.status) in (FILLED, CANCELED, EXPIRED):
                break

            if rc == 1:
                break

        order = hbt.orders(0).get(order_id)
        if order is None:
            raise RuntimeError("order missing before terminal handling")

        cancel_rc = None
        if int(order.status) == NEW:
            cancel_rc = hbt.cancel(0, order_id, True)
            order = hbt.orders(0).get(order_id)
            if order is None:
                raise RuntimeError("order missing after cancel")

        return {
            "initial_status": initial_status,
            "initial_exch_ts": initial_exch_ts,
            "local_bid_at_submit": local_bid_at_submit,
            "local_ask_at_submit": local_ask_at_submit,
            "terminal_status": int(order.status),
            "terminal_exch_ts": int(order.exch_timestamp),
            "terminal_local_ts": int(hbt.current_timestamp),
            "exec_price": float(order.exec_price),
            "exec_qty": float(order.exec_qty),
            "position": float(hbt.position(0)),
            "cancel_rc": cancel_rc,
        }
    finally:
        hbt.close()


def trace_fill_window(data, fixture, native_ts, yue_ts):
    """Audit the exact exchange events underlying the fill-time difference."""
    quote = float(fixture["price"])
    start_ts = int(fixture["entry_local_ts"])
    end_ts = max(
        int(fixture["strict_fill_exch_ts"]),
        int(native_ts),
        int(yue_ts),
    )

    exch = data[(data["ev"] & EXCH_EVENT) == EXCH_EVENT]

    same_price_aggressive_sell = 0.0
    native_best_bid_depth_min = float(fixture["strict_initial_queue_ahead"])
    depth_reductions_at_bid = 0
    cross_ask_candidate = 0

    event_window = []
    lower = min(native_ts, yue_ts, fixture["strict_fill_exch_ts"]) - 80_000_000
    upper = max(native_ts, yue_ts, fixture["strict_fill_exch_ts"]) + 30_000_000

    for row in exch:
        ts = int(row["exch_ts"])
        if ts < start_ts:
            continue
        if ts > max(end_ts, upper):
            break

        ev = int(row["ev"])
        base = ev & BASE_EVENT_MASK
        px = float(row["px"])
        qty = float(row["qty"])
        is_buy = bool(ev & BUY_EVENT)
        is_sell = bool(ev & SELL_EVENT)

        if ts <= native_ts:
            if base == TRADE_EVENT and is_sell and abs(px - quote) < 1e-9:
                same_price_aggressive_sell += qty
            if base == DEPTH_EVENT and is_buy and abs(px - quote) < 1e-9:
                if qty < native_best_bid_depth_min:
                    depth_reductions_at_bid += 1
                    native_best_bid_depth_min = qty
            if base == DEPTH_EVENT and is_sell and px <= quote and qty > 0:
                cross_ask_candidate += 1

        if lower <= ts <= upper:
            if (
                (base == TRADE_EVENT and is_sell and px <= quote + 0.01)
                or (base == DEPTH_EVENT and abs(px - quote) <= 0.02)
            ):
                side = "BUY" if is_buy else ("SELL" if is_sell else "NONE")
                kind = "TRADE" if base == TRADE_EVENT else "DEPTH"
                event_window.append((ts, kind, side, px, qty))

    print()
    print("===== EXCHANGE EVENT DIAGNOSTICS =====")
    print(f"same_price_aggressive_sell_qty_until_native={same_price_aggressive_sell:.8f}")
    print(f"min_displayed_bid_qty_until_native={native_best_bid_depth_min:.8f}")
    print(f"bid_depth_shrink_events_until_native={depth_reductions_at_bid}")
    print(f"cross_ask_update_candidates_until_native={cross_ask_candidate}")
    print(f"candidate_events_around_fill={len(event_window)}")
    print("first_35_relevant_exchange_events:")
    for ts, kind, side, px, qty in event_window[:35]:
        anchor = (
            "NATIVE_FILL" if ts == native_ts else
            "YUE_FILL" if ts == yue_ts else
            "OLD_STRICT_FILL" if ts == fixture["strict_fill_exch_ts"] else
            "-"
        )
        print(
            f"  ts={ts} {kind:5s} {side:4s} "
            f"price={px:.4f} qty={qty:.8f} {anchor}"
        )


def status_name(x):
    return {
        NEW: "NEW",
        FILLED: "FILLED",
        CANCELED: "CANCELED",
        EXPIRED: "EXPIRED",
    }.get(x, str(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", type=Path)
    ap.add_argument("--fixture-horizon-s", type=float, default=10.0)
    ap.add_argument("--qty", type=float, default=0.01)
    ap.add_argument("--tick-size", type=float, default=0.01)
    ap.add_argument("--lot-size", type=float, default=0.01)
    args = ap.parse_args()

    with np.load(args.npz) as z:
        data = z["data"]

    fixture = choose_fixture(data, args.fixture_horizon_s)

    print("===== HBT-R0.3 SINGLE ORDER AUDIT =====")
    print("side=BUY")
    print("selection=first deterministic best-bid fixture with old-strict fill")
    print("order_latency=0ns")
    print("queue_model=hbt_native_risk_adverse")
    print(f"entry_local_ts={fixture['entry_local_ts']}")
    print(f"price={fixture['price']:.4f}")
    print(f"exchange_best_bid={fixture['exchange_best_bid']:.4f}")
    print(f"exchange_best_ask={fixture['exchange_best_ask']:.4f}")
    print(
        "strict_initial_queue_ahead="
        f"{fixture['strict_initial_queue_ahead']:.8f}"
    )
    print(f"strict_fill_exch_ts={fixture['strict_fill_exch_ts']}")
    print(f"strict_fill_reason={fixture['strict_fill_reason']}")

    native = run_native(
        args.npz,
        fixture,
        args.qty,
        args.tick_size,
        args.lot_size,
        "native",
    )
    yue = run_native(
        args.npz,
        fixture,
        args.qty,
        args.tick_size,
        args.lot_size,
        "yue_strict",
    )

    for label, result in (
        ("HBT NATIVE RISK-ADVERSE", native),
        ("HBT YUE STRICT QUEUE", yue),
    ):
        print()
        print(f"===== {label} LIFECYCLE =====")
        print(f"local_bid_at_submit={result['local_bid_at_submit']:.4f}")
        print(f"local_ask_at_submit={result['local_ask_at_submit']:.4f}")
        print(f"initial_status={status_name(result['initial_status'])}")
        print(f"initial_exch_ts={result['initial_exch_ts']}")
        print(f"terminal_status={status_name(result['terminal_status'])}")
        print(f"terminal_exch_ts={result['terminal_exch_ts']}")
        print(f"terminal_local_ts={result['terminal_local_ts']}")
        print(f"exec_price={result['exec_price']:.4f}")
        print(f"exec_qty={result['exec_qty']:.8f}")
        print(f"position={result['position']:.8f}")
        print(f"cancel_rc={result['cancel_rc']}")

    print()
    print("===== FILL TIMESTAMP PARITY =====")
    old_fill_ts = fixture["strict_fill_exch_ts"]
    native_filled = native["terminal_status"] == FILLED
    yue_filled = yue["terminal_status"] == FILLED
    print(f"old_strict_fill_exch_ts={old_fill_ts}")
    print(f"hbt_native_filled={native_filled}")
    print(f"hbt_yue_strict_filled={yue_filled}")
    native_delta = (
        native["terminal_exch_ts"] - old_fill_ts
        if native_filled else None
    )
    yue_delta = (
        yue["terminal_exch_ts"] - old_fill_ts
        if yue_filled else None
    )
    print(f"native_minus_old_strict_ns={native_delta}")
    print(f"yue_minus_old_strict_ns={yue_delta}")
    fill_parity = yue_filled and yue_delta == 0
    print("YUE_STRICT_FILL_TIMESTAMP_PARITY="
          + ("PASS" if fill_parity else "DIFF"))

    if native_filled and yue_filled:
        trace_fill_window(
            data, fixture,
            native["terminal_exch_ts"],
            yue["terminal_exch_ts"],
        )

    harness_ok = all(
        result["initial_status"] == NEW
        and result["terminal_status"] in (FILLED, CANCELED)
        for result in (native, yue)
    )

    print()
    print("HARNESS_STATUS=" + ("PASS" if harness_ok else "FAIL"))
    print(
        "NOTE=YueStrictQueueModel matches old strict queue advancement, "
        "but HBT exchange book-cross fills can still cause timestamp DIFF."
    )

    raise SystemExit(0 if harness_ok else 1)


if __name__ == "__main__":
    main()
