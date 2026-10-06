import argparse
from pathlib import Path

import numpy as np
from numba import njit

from hftbacktest import (
    BacktestAsset,
    HashMapMarketDepthBacktest,
    BUY_EVENT,
    SELL_EVENT,
    DEPTH_EVENT,
    DEPTH_CLEAR_EVENT,
    DEPTH_SNAPSHOT_EVENT,
    LOCAL_EVENT,
)


BASE_EVENT_MASK = (1 << 28) - 1


def build_reference(local_events):
    """
    Independent minimal MBP reconstruction.

    This intentionally does NOT use any HBT market-depth implementation.
    It applies the local-visible HBT/Binance depth events directly and
    records one checkpoint per distinct local timestamp.
    """
    bids = {}
    asks = {}

    checkpoint_ts = []
    exp_bid = []
    exp_bid_qty = []
    exp_ask = []
    exp_ask_qty = []

    i = 0
    n = len(local_events)

    while i < n:
        ts = int(local_events[i]["local_ts"])
        j = i

        while j < n and int(local_events[j]["local_ts"]) == ts:
            row = local_events[j]
            ev = int(row["ev"])
            base = ev & BASE_EVENT_MASK

            is_buy = (ev & BUY_EVENT) != 0
            is_sell = (ev & SELL_EVENT) != 0

            if base == DEPTH_CLEAR_EVENT:
                # R0 starts from an empty book. The bootstrap clear events
                # therefore reduce to clearing the corresponding side.
                if is_buy:
                    bids.clear()
                elif is_sell:
                    asks.clear()

            elif base in (DEPTH_EVENT, DEPTH_SNAPSHOT_EVENT):
                px = float(row["px"])
                qty = float(row["qty"])

                if is_buy:
                    side = bids
                elif is_sell:
                    side = asks
                else:
                    raise RuntimeError(
                        f"depth event without side at local row {j}"
                    )

                if qty == 0.0:
                    side.pop(px, None)
                else:
                    side[px] = qty

            # Trades do not mutate the public MBP book here. Binance
            # depthUpdate remains authoritative for public depth state.

            j += 1

        if not bids or not asks:
            raise RuntimeError(
                f"reference book empty after local timestamp {ts}"
            )

        best_bid = max(bids)
        best_ask = min(asks)

        checkpoint_ts.append(ts)
        exp_bid.append(best_bid)
        exp_bid_qty.append(bids[best_bid])
        exp_ask.append(best_ask)
        exp_ask_qty.append(asks[best_ask])

        i = j

    return (
        np.asarray(checkpoint_ts, dtype=np.int64),
        np.asarray(exp_bid, dtype=np.float64),
        np.asarray(exp_bid_qty, dtype=np.float64),
        np.asarray(exp_ask, dtype=np.float64),
        np.asarray(exp_ask_qty, dtype=np.float64),
    )


@njit
def replay_hbt(hbt, expected_ts):
    """
    HBT advances wait_next_feed() by local feed timestamp groups, not by
    individual event rows. The first local group is already loaded when the
    backtester is constructed, so capture that initial state before waiting.
    """
    n = len(expected_ts)

    actual_ts = np.empty(n, dtype=np.int64)
    actual_bid = np.empty(n, dtype=np.float64)
    actual_bid_qty = np.empty(n, dtype=np.float64)
    actual_ask = np.empty(n, dtype=np.float64)
    actual_ask_qty = np.empty(n, dtype=np.float64)

    cp = 0
    feed_count = 0
    last_rc = 0
    timestamp_jump = 0
    jump_expected = 0
    jump_actual = 0

    # HBT initializes the first local-visible feed group before the first
    # wait_next_feed() call.
    if n > 0 and hbt.current_timestamp == expected_ts[0]:
        depth = hbt.depth(0)

        actual_ts[0] = hbt.current_timestamp
        actual_bid[0] = depth.best_bid
        actual_bid_qty[0] = depth.best_bid_qty
        actual_ask[0] = depth.best_ask
        actual_ask_qty[0] = depth.best_ask_qty

        cp = 1

    timeout = 60_000_000_000

    while cp < n:
        rc = hbt.wait_next_feed(False, timeout)
        last_rc = rc

        if rc == 1:
            break

        if rc == 0:
            continue

        if rc != 2:
            break

        feed_count += 1
        ts = hbt.current_timestamp

        # Defensive handling if HBT ever returns more than once for a local
        # timestamp. This should not occur for the current engine/data pair.
        if ts < expected_ts[cp]:
            continue

        if ts > expected_ts[cp]:
            timestamp_jump = 1
            jump_expected = expected_ts[cp]
            jump_actual = ts
            break

        depth = hbt.depth(0)

        actual_ts[cp] = ts
        actual_bid[cp] = depth.best_bid
        actual_bid_qty[cp] = depth.best_bid_qty
        actual_ask[cp] = depth.best_ask
        actual_ask_qty[cp] = depth.best_ask_qty

        cp += 1

    return (
        cp,
        feed_count,
        last_rc,
        timestamp_jump,
        jump_expected,
        jump_actual,
        actual_ts,
        actual_bid,
        actual_bid_qty,
        actual_ask,
        actual_ask_qty,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", type=Path)
    ap.add_argument("--tick-size", type=float, default=0.01)
    ap.add_argument("--lot-size", type=float, default=0.01)
    args = ap.parse_args()

    with np.load(args.npz) as z:
        data = z["data"]

    local_mask = (data["ev"] & LOCAL_EVENT) == LOCAL_EVENT
    local_events = data[local_mask]

    print("===== HBT-R0.2 BOOK REPLAY =====")
    print(f"npz={args.npz}")
    print(f"ordered_rows={len(data)}")
    print(f"local_feed_rows={len(local_events)}")
    print(f"tick_size={args.tick_size}")
    print(f"lot_size={args.lot_size}")

    (
        expected_ts,
        expected_bid,
        expected_bid_qty,
        expected_ask,
        expected_ask_qty,
    ) = build_reference(local_events)

    print(f"local_timestamp_groups={len(expected_ts)}")

    asset = (
        BacktestAsset()
        .data(str(args.npz))
        .linear_asset(1.0)
        .constant_order_latency(0, 0)
        .risk_adverse_queue_model()
        .no_partial_fill_exchange()
        .trading_value_fee_model(0.0, 0.0)
        .tick_size(args.tick_size)
        .lot_size(args.lot_size)
    )

    hbt = HashMapMarketDepthBacktest([asset])

    try:
        initial_ts = int(hbt.current_timestamp)

        (
            completed,
            feed_count,
            last_rc,
            timestamp_jump,
            jump_expected,
            jump_actual,
            actual_ts,
            actual_bid,
            actual_bid_qty,
            actual_ask,
            actual_ask_qty,
        ) = replay_hbt(hbt, expected_ts)
    finally:
        hbt.close()

    print()
    print("===== ENGINE REPLAY =====")
    print(f"initial_hbt_timestamp={initial_ts}")
    print(f"first_expected_timestamp={int(expected_ts[0])}")
    print(f"checkpoints_expected={len(expected_ts)}")
    print(f"checkpoints_completed={completed}")
    print(f"wait_next_feed_calls_with_market_feed={feed_count}")
    print(f"last_rc={last_rc}")

    if timestamp_jump:
        raise RuntimeError(
            "HBT skipped an expected local timestamp group: "
            f"expected={jump_expected} actual={jump_actual}"
        )

    if completed != len(expected_ts):
        raise RuntimeError(
            "HBT ended before all local timestamp groups were observed: "
            f"{completed}/{len(expected_ts)}"
        )

    ts_bad = actual_ts != expected_ts

    bid_bad = ~np.isclose(
        actual_bid,
        expected_bid,
        atol=1e-12,
        rtol=0.0,
    )
    ask_bad = ~np.isclose(
        actual_ask,
        expected_ask,
        atol=1e-12,
        rtol=0.0,
    )

    bid_qty_bad = ~np.isclose(
        actual_bid_qty,
        expected_bid_qty,
        atol=1e-9,
        rtol=0.0,
    )
    ask_qty_bad = ~np.isclose(
        actual_ask_qty,
        expected_ask_qty,
        atol=1e-9,
        rtol=0.0,
    )

    bad = (
        ts_bad
        | bid_bad
        | ask_bad
        | bid_qty_bad
        | ask_qty_bad
    )

    bad_idx = np.flatnonzero(bad)

    print()
    print("===== PARITY =====")
    print(f"timestamp_mismatches={int(ts_bad.sum())}")
    print(f"best_bid_mismatches={int(bid_bad.sum())}")
    print(f"best_ask_mismatches={int(ask_bad.sum())}")
    print(f"best_bid_qty_mismatches={int(bid_qty_bad.sum())}")
    print(f"best_ask_qty_mismatches={int(ask_qty_bad.sum())}")
    print(f"total_bad_checkpoints={len(bad_idx)}")

    if len(bad_idx):
        print()
        print("===== FIRST MISMATCHES =====")

        for idx in bad_idx[:20]:
            print(
                f"idx={idx} "
                f"expected_ts={expected_ts[idx]} "
                f"actual_ts={actual_ts[idx]} "
                f"expected="
                f"{expected_bid[idx]:.4f}@{expected_bid_qty[idx]:.8f}"
                f" / "
                f"{expected_ask[idx]:.4f}@{expected_ask_qty[idx]:.8f} "
                f"actual="
                f"{actual_bid[idx]:.4f}@{actual_bid_qty[idx]:.8f}"
                f" / "
                f"{actual_ask[idx]:.4f}@{actual_ask_qty[idx]:.8f}"
            )

        raise SystemExit(1)

    print()
    print("===== FINAL BOOK =====")
    print(
        f"bid={actual_bid[-1]:.4f} "
        f"qty={actual_bid_qty[-1]:.8f}"
    )
    print(
        f"ask={actual_ask[-1]:.4f} "
        f"qty={actual_ask_qty[-1]:.8f}"
    )

    print()
    print("STATUS=PASS")


if __name__ == "__main__":
    main()
