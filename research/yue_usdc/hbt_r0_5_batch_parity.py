"""HBT-R0.5: deterministic paired BUY/SELL maker-order parity audit.

Purpose
-------
Test the trade-only exchange + YueStrictQueueModel against an independent
trades-only reference across many orders, including no-fill cancellations and
post-only rejections. This tests order semantics on HBT's exchange clock with
zero order latency. It does NOT validate the legacy yue_maker wall-clock
replay, measured network latency, adverse selection, or actual exchange fills.

Each fixture is created by a deterministic 30-second schedule, not by
looking ahead for fills. Orders do not overlap (TTL < schedule spacing).
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hftbacktest import (
    BacktestAsset,
    HashMapMarketDepthBacktest,
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
NS_PER_MS = 1_000_000


@dataclass(frozen=True)
class Fixture:
    number: int
    local_ts: int
    side: str
    price: float
    local_bid: float
    local_ask: float
    deadline: int
    exchange_bid: float
    exchange_ask: float
    queue_ahead: float
    expected_status: int
    expected_fill_ts: int | None
    expected_fill_reason: str | None


def apply_depth(ev_row, bids: dict[float, float], asks: dict[float, float]):
    ev = int(ev_row["ev"])
    kind = ev & BASE_EVENT_MASK

    if kind == DEPTH_CLEAR_EVENT:
        if ev & BUY_EVENT:
            bids.clear()
        elif ev & SELL_EVENT:
            asks.clear()
        else:
            bids.clear()
            asks.clear()
        return

    if kind not in (DEPTH_EVENT, DEPTH_SNAPSHOT_EVENT):
        return

    if ev & BUY_EVENT:
        book = bids
    elif ev & SELL_EVENT:
        book = asks
    else:
        raise ValueError("depth row with no bid/ask side")

    px = float(ev_row["px"])
    qty = float(ev_row["qty"])
    if qty <= 0.0:
        book.pop(px, None)
    else:
        book[px] = qty


def choose_scheduled_local_quotes(local_events, step_ns: int, ttl_ns: int):
    """Produce scheduled quotes without consulting any future trade activity."""
    bids, asks = {}, {}
    n = len(local_events)

    if n == 0:
        raise RuntimeError("No local rows")

    next_slot = int(local_events[0]["local_ts"]) + 10 * NS_PER_S
    last_ts = int(local_events[-1]["local_ts"])
    end_slot = last_ts - ttl_ns - 2 * NS_PER_S

    results = []
    group = 0
    i = 0

    while i < n:
        now = int(local_events[i]["local_ts"])
        j = i

        while j < n and int(local_events[j]["local_ts"]) == now:
            apply_depth(local_events[j], bids, asks)
            j += 1

        if now >= next_slot and now <= end_slot and bids and asks:
            bid = max(bids)
            ask = min(asks)
            if bid < ask:
                side = "BUY" if len(results) % 2 == 0 else "SELL"
                price = bid if side == "BUY" else ask
                results.append(
                    (len(results), now, side, price, bid, ask, now + ttl_ns)
                )

            # If there are no valid quotes, still move the schedule forward,
            # rather than oversampling an unusual future market condition.
            next_slot += step_ns
            while next_slot <= now:
                next_slot += step_ns

        i = j
        group += 1

    return results, group


def reference_trade_fill(exch, exch_ts, start_idx, deadline, side, price, queue):
    """Old yue_maker strict BUY/SELL trade-only queue semantics."""
    remaining = float(queue)

    for idx in range(start_idx, len(exch)):
        ts = int(exch_ts[idx])
        if ts > deadline:
            break

        row = exch[idx]
        ev = int(row["ev"])
        if (ev & BASE_EVENT_MASK) != TRADE_EVENT:
            continue

        correct_aggressor = (
            (side == "BUY" and bool(ev & SELL_EVENT))
            or (side == "SELL" and bool(ev & BUY_EVENT))
        )
        if not correct_aggressor:
            continue

        trade_px = float(row["px"])
        trade_qty = float(row["qty"])

        if (side == "BUY" and trade_px < price - 1e-9) or (
            side == "SELL" and trade_px > price + 1e-9
        ):
            return ts, "trade_through"

        if abs(trade_px - price) <= 1e-9:
            remaining -= trade_qty
            if remaining <= 0.0:
                return ts, "queue_consumed"

    return None, None


def prepare_fixtures(exch, quote_rows):
    """
    Observe exchange depth at each deterministic local quote timestamp.
    The reference starts AFTER the order's 0-latency exchange acceptance;
    all exchange events have ms granularity while local timestamps include ns.
    """
    exch_ts = exch["exch_ts"]
    if len(exch_ts) > 1 and np.any(exch_ts[1:] < exch_ts[:-1]):
        raise RuntimeError("Exchange events are not time ordered")

    bid_book, ask_book = {}, {}
    exch_cursor = 0
    fixtures: list[Fixture] = []
    skipped_missing_exchange_book = 0

    for number, local_ts, side, price, local_bid, local_ask, deadline in quote_rows:
        end = int(np.searchsorted(exch_ts, local_ts, side="right"))
        while exch_cursor < end:
            apply_depth(exch[exch_cursor], bid_book, ask_book)
            exch_cursor += 1

        if not bid_book or not ask_book:
            skipped_missing_exchange_book += 1
            continue

        exchange_bid = max(bid_book)
        exchange_ask = min(ask_book)
        queue = float(
            (bid_book if side == "BUY" else ask_book).get(price, 0.0)
        )

        post_only_expired = (
            (side == "BUY" and price >= exchange_ask)
            or (side == "SELL" and price <= exchange_bid)
        )

        if post_only_expired:
            expected = EXPIRED
            fill_ts, reason = None, None
        else:
            fill_ts, reason = reference_trade_fill(
                exch, exch_ts, end, deadline, side, price, queue
            )
            expected = FILLED if fill_ts is not None else CANCELED

        fixtures.append(Fixture(
            number, local_ts, side, price, local_bid, local_ask,
            deadline, exchange_bid, exchange_ask, queue,
            expected, fill_ts, reason,
        ))

    return fixtures, skipped_missing_exchange_book


def run_hbt(npz: Path, fixtures: list[Fixture], qty: float, tick: float, lot: float):
    asset = (
        BacktestAsset()
        .data(str(npz))
        .linear_asset(1.0)
        .constant_order_latency(0, 0)
        .yue_strict_queue_model()
        .yue_strict_trade_only_exchange()
        .trading_value_fee_model(0.0, 0.0)
        .tick_size(tick)
        .lot_size(lot)
    )

    hbt = HashMapMarketDepthBacktest([asset])
    results = []

    try:
        # The engine starts at INT64_MAX and initializes on the first step.
        init_rc = hbt.wait_next_feed(False, 60 * NS_PER_S)
        if init_rc != 2:
            raise RuntimeError(f"Engine failed to initialize: rc={init_rc}")

        for case in fixtures:
            advance = int(case.local_ts - hbt.current_timestamp)
            if advance < 0:
                raise RuntimeError(f"Fixture time went backwards: #{case.number}")

            rc = hbt.elapse(advance)
            if rc != 0:
                raise RuntimeError(
                    f"Failed advancing to order #{case.number}: rc={rc}"
                )

            depth = hbt.depth(0)
            actual_bid = float(depth.best_bid)
            actual_ask = float(depth.best_ask)
            local_book_ok = (
                abs(actual_bid - case.local_bid) < 1e-9
                and abs(actual_ask - case.local_ask) < 1e-9
            )

            order_id = case.number + 1
            if case.side == "BUY":
                submit_rc = hbt.submit_buy_order(
                    0, order_id, case.price, qty, GTX, LIMIT, True
                )
            else:
                submit_rc = hbt.submit_sell_order(
                    0, order_id, case.price, qty, GTX, LIMIT, True
                )

            if submit_rc != 0:
                raise RuntimeError(
                    f"Submit order #{case.number} returned rc={submit_rc}"
                )

            order = hbt.orders(0).get(order_id)
            if order is None:
                raise RuntimeError(f"Missing local order #{case.number}")

            ack_status = int(order.status)
            ack_exch_ts = int(order.exch_timestamp)
            cancel_rc = None

            if ack_status == NEW:
                delta = int(case.deadline - hbt.current_timestamp)
                if delta < 0:
                    raise RuntimeError("Submission advanced past expiry")

                rc = hbt.elapse(delta)
                if rc != 0:
                    raise RuntimeError(
                        f"Failed to reach order expiry #{case.number}: rc={rc}"
                    )

                order = hbt.orders(0).get(order_id)
                if order is None:
                    raise RuntimeError("Order missing after horizon")

                if int(order.status) == NEW:
                    cancel_rc = int(hbt.cancel(0, order_id, True))
                    if cancel_rc != 0:
                        raise RuntimeError(
                            f"Cancel #{case.number} returned rc={cancel_rc}"
                        )
                    order = hbt.orders(0).get(order_id)

            results.append({
                "case": case,
                "local_book_ok": local_book_ok,
                "ack_status": ack_status,
                "ack_exch_ts": ack_exch_ts,
                "status": int(order.status),
                "terminal_exch_ts": int(order.exch_timestamp),
                "exec_price": (
                    float(order.exec_price) if int(order.status) == FILLED
                    else None
                ),
                "exec_qty": (
                    float(order.exec_qty) if int(order.status) == FILLED
                    else None
                ),
                "cancel_rc": cancel_rc,
            })

        return results

    finally:
        hbt.close()


def name(status):
    return {
        NEW: "NEW",
        FILLED: "FILLED",
        CANCELED: "CANCELED",
        EXPIRED: "EXPIRED",
    }.get(status, str(status))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", type=Path)
    ap.add_argument("--step-s", type=float, default=30.0)
    ap.add_argument("--ttl-s", type=float, default=4.123457)
    ap.add_argument("--qty", type=float, default=0.01)
    ap.add_argument("--tick-size", type=float, default=0.01)
    ap.add_argument("--lot-size", type=float, default=0.01)
    args = ap.parse_args()

    step_ns = int(args.step_s * NS_PER_S)
    ttl_ns = int(args.ttl_s * NS_PER_S)
    if ttl_ns <= 0 or step_ns <= ttl_ns + NS_PER_S:
        ap.error("Require step-s > ttl-s + 1 second")

    with np.load(args.npz) as z:
        data = z["data"]

    local = data[(data["ev"] & LOCAL_EVENT) == LOCAL_EVENT]
    exch = data[(data["ev"] & EXCH_EVENT) == EXCH_EVENT]

    quotes, local_groups = choose_scheduled_local_quotes(
        local, step_ns, ttl_ns
    )
    fixtures, skipped_missing_book = prepare_fixtures(exch, quotes)

    print("===== HBT-R0.5 MULTI-ORDER TRADE-ONLY PARITY =====")
    print("reference=yue_maker_strict_queue_on_exchange_time")
    print("queue=YueStrictQueueModel")
    print("exchange=YueStrictTradeOnlyExchange")
    print("order_latency=0ns")
    print("selection=deterministic_30s_schedule_alternating_BUY_SELL")
    print("scope=single_dataset_nonoverlapping_post_only_orders")
    print("full_yue_maker_wall_clock_parity=NOT_TESTED")
    print(f"npz={args.npz}")
    print(f"local_groups={local_groups}")
    print(f"quotes_selected={len(quotes)}")
    print(f"missing_exchange_book_skipped={skipped_missing_book}")
    print(f"fixtures={len(fixtures)}")
    print(f"ttl_ns={ttl_ns}")
    print(f"step_ns={step_ns}")

    if len(fixtures) < 40:
        raise RuntimeError("Insufficient scheduled fixtures; need >=40")

    actual = run_hbt(
        args.npz, fixtures, args.qty, args.tick_size, args.lot_size
    )

    counts = Counter()
    mismatches = []

    for result in actual:
        case = result["case"]
        counts["total"] += 1
        counts["side_" + case.side] += 1
        counts["expected_" + name(case.expected_status)] += 1
        counts["actual_" + name(result["status"])] += 1

        status_ok = result["status"] == case.expected_status
        fill_time_ok = (
            case.expected_fill_ts == result["terminal_exch_ts"]
            if case.expected_status == FILLED and result["status"] == FILLED
            else case.expected_status != FILLED
        )
        fill_qty_ok = (
            result["exec_qty"] is not None
            and abs(result["exec_qty"] - args.qty) < 1e-9
            and result["exec_price"] is not None
            and abs(result["exec_price"] - case.price) < 1e-9
        ) if case.expected_status == FILLED else True

        ack_ok = (
            result["ack_status"]
            == (EXPIRED if case.expected_status == EXPIRED else NEW)
        )
        entry_time_ok = result["ack_exch_ts"] == case.local_ts
        book_ok = result["local_book_ok"]
        cancel_ok = (
            result["cancel_rc"] == 0
            if case.expected_status == CANCELED else
            result["cancel_rc"] is None
        )

        checks = {
            "status": status_ok,
            "fill_time": fill_time_ok,
            "fill_qty_price": fill_qty_ok,
            "submit_ack": ack_ok,
            "entry_timestamp": entry_time_ok,
            "local_book": book_ok,
            "cancel_path": cancel_ok,
        }

        for check, okay in checks.items():
            if not okay:
                counts["mismatch_" + check] += 1

        if not all(checks.values()):
            mismatches.append((case, result, checks))

    print()
    print("===== STATUS DISTRIBUTION =====")
    for key in sorted(counts):
        if key.startswith("expected_") or key.startswith("actual_") or key.startswith("side_"):
            print(f"{key}={counts[key]}")

    print()
    print("===== VALIDATION CHECKS =====")
    for key in (
        "status", "fill_time", "fill_qty_price", "submit_ack",
        "entry_timestamp", "local_book", "cancel_path"
    ):
        print(f"{key}_mismatches={counts['mismatch_' + key]}")
    print(f"total_bad_orders={len(mismatches)}")

    if mismatches:
        print()
        print("===== FIRST 20 MISMATCHES =====")
        for case, result, checks in mismatches[:20]:
            failed = ",".join(
                k for k, ok in checks.items() if not ok
            )
            print(
                f"case={case.number} side={case.side} entry={case.local_ts} "
                f"price={case.price:.4f} queue={case.queue_ahead:.8f} "
                f"expected={name(case.expected_status)} "
                f"expected_fill={case.expected_fill_ts} "
                f"fill_reason={case.expected_fill_reason} "
                f"actual={name(result['status'])} "
                f"actual_exch_ts={result['terminal_exch_ts']} "
                f"ack={name(result['ack_status'])} "
                f"failed={failed}"
            )

    n_buy = counts["side_BUY"]
    n_sell = counts["side_SELL"]
    n_fill = counts["expected_FILLED"]
    n_cancel = counts["expected_CANCELED"]
    n_post_only_expired = counts["expected_EXPIRED"]

    coverage_ok = (
        n_buy > 0 and n_sell > 0
        and n_fill > 0 and n_cancel > 0
    )
    parity_ok = not mismatches
    print()
    print("BOTH_SIDES_COVERED=" + str(n_buy > 0 and n_sell > 0))
    print("FILL_PATH_COVERED=" + str(n_fill > 0))
    print("CANCELLATION_PATH_COVERED=" + str(n_cancel > 0))
    print("POST_ONLY_EXPIRE_PATH_COVERED=" + str(n_post_only_expired > 0))
    print("BATCH_PARITY_STATUS=" + (
        "PASS" if parity_ok and coverage_ok else "FAIL"
    ))
    print(
        "LIMITATION=zero-order-latency/exchange-time only; "
        "does not validate recv_wall_ns or live fill likelihood"
    )

    raise SystemExit(0 if parity_ok and coverage_ok else 1)


if __name__ == "__main__":
    main()
