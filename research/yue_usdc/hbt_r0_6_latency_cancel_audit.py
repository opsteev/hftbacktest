"""R0.6: maker entry/cancel latency and race parity on one SOLUSDC capture.

Deterministic 30s sampling, BUY/SELL alternating, independently computed
strict trade-only exchange reference. In the reference, the order becomes
live at local_submit + entry_latency; if still active, a cancel requested at
local_submit + TTL reaches the exchange after entry_latency. Response
latency affects observations, not when an exchange-side fill occurs.

Post-only reject smoke uses deliberately marketable quotes at zero latency,
not representative maker-alpha sampling. Exact timestamp ties follow the
source engine EventSet ordering, with EXCH feed processed before EXCH orders.
All results use HBT's corrected exchange clock; this is not legacy
yue_maker recv_wall_ns replay parity.

Exit != 0 on any mismatch; preserve diagnostics in the terminal output.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hftbacktest import BacktestAsset, HashMapMarketDepthBacktest, GTX, LIMIT
from hftbacktest.order import NEW, FILLED, CANCELED, EXPIRED

from hbt_r0_5_batch_parity import (
    NS_PER_MS,
    NS_PER_S,
    apply_depth,
    choose_scheduled_local_quotes,
    name,
    reference_trade_fill,
)


@dataclass(frozen=True)
class Expected:
    number: int
    local_ts: int
    deadline: int
    entry_exch_ts: int
    cancel_exch_ts: int
    side: str
    price: float
    bid: float
    ask: float
    exch_bid: float
    exch_ask: float
    queue_ahead: float
    status: int
    fill_ts: int | None
    fill_reason: str | None


def build_reference(exch, quotes, entry_ns):
    """Independent exchange-side reference; only state at acceptance is used."""
    exch_ts = exch["exch_ts"]
    if np.any(exch_ts[1:] < exch_ts[:-1]):
        raise RuntimeError("Exchange events are not sorted")

    bids, asks = {}, {}
    cursor = 0
    rows = []
    missing_book = 0
    entry_ties = 0
    cancel_ties = 0

    for number, local_ts, side, px, bid, ask, deadline in quotes:
        entry_exch_ts = local_ts + entry_ns
        cancel_exch_ts = deadline + entry_ns

        # HBT EventSet breaks identical timestamps by event kind:
        # LocalData -> LocalOrder -> ExchData -> ExchOrder.
        # Consequently all exchange feed rows at entry_exch_ts are
        # processed BEFORE the new order reaches the exchange; likewise,
        # all exchange trades at cancel_exch_ts occur before cancel takes
        # effect. Do not shift, discard, or fabricate timestamps.
        #
        # The "right" search excludes same-ts feed from future trade fills
        # (already processed before order acceptance), while the reference
        # trade loop includes same-ts feed up to cancel_exch_ts inclusive.
        entry_left = int(np.searchsorted(exch_ts, entry_exch_ts, side="left"))
        end = int(np.searchsorted(exch_ts, entry_exch_ts, side="right"))
        cancel_left = int(np.searchsorted(exch_ts, cancel_exch_ts, side="left"))
        cancel_right = int(np.searchsorted(exch_ts, cancel_exch_ts, side="right"))
        entry_ties += int(end > entry_left)
        cancel_ties += int(cancel_right > cancel_left)
        while cursor < end:
            apply_depth(exch[cursor], bids, asks)
            cursor += 1

        if not bids or not asks:
            missing_book += 1
            continue

        ebid, eask = max(bids), min(asks)
        queue = float((bids if side == "BUY" else asks).get(px, 0.0))
        crosses = (
            (side == "BUY" and px >= eask)
            or (side == "SELL" and px <= ebid)
        )

        if crosses:
            status = EXPIRED
            fill_ts, reason = None, None
        else:
            fill_ts, reason = reference_trade_fill(
                exch, exch_ts, end, cancel_exch_ts,
                side, px, queue,
            )
            status = FILLED if fill_ts is not None else CANCELED

        rows.append(
            Expected(
                number, local_ts, deadline, entry_exch_ts, cancel_exch_ts,
                side, px, bid, ask, ebid, eask, queue, status,
                fill_ts, reason,
            )
        )

    return rows, missing_book, entry_ties, cancel_ties


def make_hbt(npz, entry_ns, response_ns, tick, lot):
    asset = (
        BacktestAsset()
        .data(str(npz))
        .linear_asset(1.0)
        .constant_order_latency(entry_ns, response_ns)
        .yue_strict_queue_model()
        .yue_strict_trade_only_exchange()
        .trading_value_fee_model(0.0, 0.0)
        .tick_size(tick)
        .lot_size(lot)
    )
    return HashMapMarketDepthBacktest([asset])


def step_to(hbt, ts, context):
    current = int(hbt.current_timestamp)
    if ts < current:
        raise RuntimeError(f"time already past {context}: {ts} < {current}")
    rc = int(hbt.elapse(ts - current))
    if rc != 0:
        raise RuntimeError(f"elapse {context} failed: rc={rc}")
    if int(hbt.current_timestamp) != ts:
        raise RuntimeError(f"timestamp discrepancy at {context}")


def submit(hbt, case, qty):
    order_id = case.number + 1

    if case.side == "BUY":
        rc = int(hbt.submit_buy_order(
            0, order_id, case.price, qty, GTX, LIMIT, True,
        ))
    else:
        rc = int(hbt.submit_sell_order(
            0, order_id, case.price, qty, GTX, LIMIT, True,
        ))

    if rc != 0:
        raise RuntimeError(f"submit #{case.number} returned {rc}")

    order = hbt.orders(0).get(order_id)
    if order is None:
        raise RuntimeError(f"no local order #{case.number}")

    return order_id, int(order.status), int(order.exch_timestamp)


def run_scenario(npz, cases, entry_ns, response_ns, qty, tick, lot):
    hbt = make_hbt(npz, entry_ns, response_ns, tick, lot)
    rows = []

    try:
        rc = int(hbt.wait_next_feed(False, 60 * NS_PER_S))
        if rc != 2:
            raise RuntimeError(f"engine init rc={rc}")

        for case in cases:
            step_to(hbt, case.local_ts, f"entry {case.number}")
            depth = hbt.depth(0)
            book_ok = (
                abs(float(depth.best_bid) - case.bid) < 1e-9
                and abs(float(depth.best_ask) - case.ask) < 1e-9
            )
            order_id, ack_status, ack_exch_ts = submit(hbt, case, qty)
            ack_local_ts = int(hbt.current_timestamp)

            # A fill might already have occurred before the entry ACK,
            # particularly for the longest latency scenario.
            status_at_deadline = ack_status
            cancel_rc = None
            if ack_status == NEW:
                step_to(hbt, case.deadline, f"cancel request {case.number}")
                order = hbt.orders(0).get(order_id)
                if order is None:
                    raise RuntimeError(f"missing order before cancel {case.number}")
                status_at_deadline = int(order.status)

                if status_at_deadline == NEW:
                    cancel_rc = int(hbt.cancel(0, order_id, True))
                    if cancel_rc != 0:
                        raise RuntimeError(f"cancel #{case.number} returned {cancel_rc}")

            order = hbt.orders(0).get(order_id)
            if order is None:
                raise RuntimeError(f"missing final order #{case.number}")

            terminal = int(order.status)
            terminal_ts = int(order.exch_timestamp)
            fill_price = float(order.exec_price) if terminal == FILLED else None
            fill_qty = float(order.exec_qty) if terminal == FILLED else None
            rows.append({
                "case": case,
                "book_ok": book_ok,
                "ack_status": ack_status,
                "ack_exch_ts": ack_exch_ts,
                "ack_local_ts": ack_local_ts,
                "status_at_deadline": status_at_deadline,
                "cancel_rc": cancel_rc,
                "status": terminal,
                "terminal_exch_ts": terminal_ts,
                "exec_price": fill_price,
                "exec_qty": fill_qty,
            })

        return rows
    finally:
        hbt.close()


def post_only_rejection_smoke(npz, quotes, tick, lot):
    """Intentionally cross the spread to exercise both GTX expire paths."""
    results = []

    for _, local_ts, side, _, bid, ask, _ in quotes[:2]:
        # The first two alternating quotes are BUY then SELL.
        crossing_price = (
            round((ask + 1.0) / tick) * tick if side == "BUY"
            else round((bid - 1.0) / tick) * tick
        )

        hbt = make_hbt(npz, 0, 0, tick, lot)
        try:
            rc = int(hbt.wait_next_feed(False, 60 * NS_PER_S))
            if rc != 2:
                raise RuntimeError(f"GTX probe init rc={rc}")

            step_to(hbt, local_ts, f"GTX probe {side}")
            order_id = 9001 if side == "BUY" else 9002

            if side == "BUY":
                rc = int(hbt.submit_buy_order(
                    0, order_id, crossing_price, lot, GTX, LIMIT, True
                ))
            else:
                rc = int(hbt.submit_sell_order(
                    0, order_id, crossing_price, lot, GTX, LIMIT, True
                ))

            if rc != 0:
                raise RuntimeError(f"GTX probe {side} submit rc={rc}")

            order = hbt.orders(0).get(order_id)
            if order is None:
                raise RuntimeError(f"GTX probe {side}: local order missing")

            status = int(order.status)
            results.append((side, crossing_price, status))
        finally:
            hbt.close()

    return results


def compare_scenario(rows, qty, entry_ns, response_ns):
    counts = Counter()
    bad = []

    for item in rows:
        case = item["case"]
        counts["total"] += 1
        counts["expected_" + name(case.status)] += 1
        counts["actual_" + name(item["status"])] += 1

        if case.fill_ts is not None and case.fill_ts > case.deadline:
            counts["reference_fills_after_cancel_requested"] += 1
        if (
            item["status"] == FILLED
            and item["status_at_deadline"] == NEW
            and item["cancel_rc"] == 0
        ):
            counts["actual_filled_while_cancel_pending"] += 1

        # At the entry ACK, an order may already be FILLED, but
        # exchange acceptance itself was at local_ts + entry_latency.
        checks = {
            "status": item["status"] == case.status,
            "fill_timestamp": (
                item["terminal_exch_ts"] == case.fill_ts
                if case.status == FILLED and item["status"] == FILLED
                else case.status != FILLED
            ),
            "fill_price_qty": (
                item["exec_price"] is not None
                and abs(item["exec_price"] - case.price) < 1e-9
                and item["exec_qty"] is not None
                and abs(item["exec_qty"] - qty) < 1e-9
                if case.status == FILLED and item["status"] == FILLED
                else case.status != FILLED
            ),
            "ack_exchange_timestamp": (
                item["ack_exch_ts"] == case.entry_exch_ts
                if case.status != FILLED or item["ack_status"] != FILLED
                else True
            ),
            "local_book": item["book_ok"],
            "cancel_status": (
                item["cancel_rc"] == 0
                if case.status == CANCELED
                else True
            ),
        }

        for k, ok in checks.items():
            if not ok:
                counts["mismatch_" + k] += 1
        if not all(checks.values()):
            bad.append((case, item, tuple(k for k, ok in checks.items() if not ok)))

    print()
    print(
        "===== LATENCY SCENARIO "
        f"entry_ms={entry_ns / NS_PER_MS:.0f} "
        f"response_ms={response_ns / NS_PER_MS:.0f} ====="
    )

    for key in sorted(counts):
        if key.startswith(("expected_", "actual_")):
            print(f"{key}={counts[key]}")

    for key in (
        "status", "fill_timestamp", "fill_price_qty",
        "ack_exchange_timestamp", "local_book", "cancel_status",
    ):
        print(f"{key}_mismatches={counts['mismatch_' + key]}")

    print(
        "reference_fills_after_cancel_requested="
        f"{counts['reference_fills_after_cancel_requested']}"
    )
    print(
        "actual_filled_while_cancel_pending="
        f"{counts['actual_filled_while_cancel_pending']}"
    )
    print(f"total_bad_orders={len(bad)}")

    for case, item, issues in bad[:12]:
        print(
            f"  mismatch case={case.number} side={case.side} "
            f"px={case.price:.4f} local_ts={case.local_ts} "
            f"entry_ts={case.entry_exch_ts} cancel_ts={case.cancel_exch_ts} "
            f"queue={case.queue_ahead:.8f} expected={name(case.status)} "
            f"expected_fill_ts={case.fill_ts} reason={case.fill_reason} "
            f"actual={name(item['status'])} "
            f"actual_fill_ts={item['terminal_exch_ts']} "
            f"ack={name(item['ack_status'])} "
            f"at_deadline={name(item['status_at_deadline'])} "
            f"cancel_rc={item['cancel_rc']} checks={','.join(issues)}"
        )

    coverage = (
        counts["total"] >= 40
        and counts["expected_FILLED"] > 0
        and counts["expected_CANCELED"] > 0
    )
    passed = len(bad) == 0 and coverage
    print(
        f"LATENCY_SCENARIO_STATUS={'PASS' if passed else 'FAIL'}"
        f" coverage={'PASS' if coverage else 'INSUFFICIENT'}"
    )
    return passed, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", type=Path)
    ap.add_argument("--step-s", type=float, default=30.0)
    ap.add_argument("--ttl-s", type=float, default=4.123457)
    ap.add_argument("--qty", type=float, default=0.01)
    ap.add_argument("--tick-size", type=float, default=0.01)
    ap.add_argument("--lot-size", type=float, default=0.01)
    ap.add_argument("--latencies-ms", default="0,50,100,250")
    args = ap.parse_args()

    entry_values = [
        int(x.strip()) * NS_PER_MS
        for x in args.latencies_ms.split(",")
        if x.strip()
    ]
    if not entry_values or any(x < 0 for x in entry_values):
        ap.error("Expected comma-separated nonnegative integer ms latencies")

    step_ns = int(args.step_s * NS_PER_S)
    ttl_ns = int(args.ttl_s * NS_PER_S)
    if ttl_ns < 3 * max(entry_values) or step_ns <= ttl_ns + 2 * max(entry_values) + NS_PER_S:
        ap.error("TTL must exceed 3x max entry latency; schedule must avoid overlap")

    with np.load(args.npz) as z:
        data = z["data"]

    from hftbacktest import EXCH_EVENT, LOCAL_EVENT
    local = data[(data["ev"] & LOCAL_EVENT) == LOCAL_EVENT]
    exch = data[(data["ev"] & EXCH_EVENT) == EXCH_EVENT]

    quotes, groups = choose_scheduled_local_quotes(local, step_ns, ttl_ns)
    if len(quotes) < 40:
        raise RuntimeError(f"Only {len(quotes)} scheduled quotes")

    print("===== HBT-R0.6 ENTRY/CANCEL LATENCY AND GTX AUDIT =====")
    print(f"npz={args.npz}")
    print("clock=corrected_HBT_exchange_ns")
    print("reference=strict_trade_only_exchange_side")
    print("queue=YueStrictQueueModel")
    print("exchange=YueStrictTradeOnlyExchange")
    print("order_entry_latency_ms=0,50,100,250 (default)")
    print("response_latency_ms=same_as_entry")
    print("cancel_latency_ms=same_as_order_entry")
    print(f"local_timestamp_groups={groups}")
    print(f"deterministic_scheduled_quotes={len(quotes)}")
    print("full_original_yue_maker_recv_wall_ns_parity=NOT_TESTED")
    print("order_ack_or_cancel_response_races_are_reported_not_suppressed")

    any_bad = False

    for entry_ns in entry_values:
        response_ns = entry_ns
        cases, skipped, entry_ties, cancel_ties = build_reference(
            exch, quotes, entry_ns
        )
        print(
            f"scenario_entry_ms={entry_ns / NS_PER_MS:.0f} "
            f"exact_exchange_feed_ties_on_entry={entry_ties} "
            f"exact_exchange_feed_ties_on_cancel_effective={cancel_ties}"
        )
        if skipped:
            print(f"missing_exchange_book_skipped={skipped}")
        observed = run_scenario(
            args.npz, cases, entry_ns, response_ns,
            args.qty, args.tick_size, args.lot_size,
        )
        passed, _ = compare_scenario(
            observed, args.qty, entry_ns, response_ns
        )
        if not passed:
            any_bad = True

    probes = post_only_rejection_smoke(
        args.npz, quotes, args.tick_size, args.lot_size,
    )
    print()
    print("===== POST-ONLY GTX CONTROLLED REJECTION =====")
    for side, px, status in probes:
        print(f"{side} marketable_quote={px:.4f} status={name(status)}")

    probe_passed = (
        len(probes) == 2
        and {s for s, _, _ in probes} == {"BUY", "SELL"}
        and all(status == EXPIRED for _, _, status in probes)
    )
    print("POST_ONLY_REJECTION_STATUS=" + (
        "PASS" if probe_passed else "FAIL"
    ))

    passed = not any_bad and probe_passed
    print()
    print("R0_6_STATUS=" + ("PASS" if passed else "DIAGNOSTIC_FAILURE"))
    print(
        "LIMITATION=single capture, synthetic fixed latencies; "
        "no measured order latency, no recv_wall_ns original yue_maker "
        "full replay comparison, no live fill verification."
    )
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
