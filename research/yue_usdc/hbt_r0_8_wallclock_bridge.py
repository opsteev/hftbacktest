"""R0.8: pair old yue_maker wall-clock virtual fills with HBT orders.

Deterministic 30s SOLUSDC bookTicker quote candidates. Price, displayed
queue and decision timestamp come from the original raw bookTicker records
(the original yue_maker input). Raw aggTrade fills use VirtualOrder semantics
on recv_wall_ns; HBT runs SAME quote as a real GTX order through YueStrict
Queue + TradeOnlyExchange, on the converted exchange/local event clocks.

This is a DIAGNOSTIC BRIDGE, NOT engine parity:
- old virtual orders do not model entry latency, post-only acknowledgments,
  HBT's exchange-time queue-ahead, or exchange-side cancellation;
- old yue_maker full strategy has state/fair cancellations, not replayed here;
- HBT's per-order responses are simulated; market feed receives use a global
  ex-post shift calibrated on the entire captured hour.
Outputs differences without claiming the two executions should match.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess

import numpy as np

from hftbacktest import (
    BacktestAsset, HashMapMarketDepthBacktest, GTX, LIMIT, LOCAL_EVENT,
)
from hftbacktest.order import NEW, FILLED, CANCELED, EXPIRED

NS_PER_S = 1_000_000_000
NS_PER_MS = 1_000_000


@dataclass(frozen=True)
class Quote:
    idx: int
    wall_ns: int
    side: str
    price: float
    displayed_qty: float
    bid: float
    ask: float
    deadline_wall_ns: int


@dataclass(frozen=True)
class Trade:
    wall_ns: int
    exch_ns: int
    px: float
    qty: float
    seller_aggressor: bool


def iter_records(path: Path):
    proc = subprocess.Popen(
        ["zstd", "-dc", str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1024 * 1024,
    )
    assert proc.stdout is not None
    try:
        for line_no, line in enumerate(proc.stdout, 1):
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"bad JSON line {line_no}: {e}") from e
    finally:
        proc.stdout.close()
        err = proc.stderr.read() if proc.stderr else ""
        if proc.stderr:
            proc.stderr.close()
        if proc.wait() != 0:
            raise RuntimeError(f"zstd failed: {err[:300]}")


def raw_candidates(raw_path, symbol, interval_ns, ttl_ns, lower_wall, upper_wall):
    """Select every 30s from raw bookTicker; no future fill inspection."""
    quotes = []
    trades = []
    n_books = 0
    n_trades = 0
    next_slot = int(lower_wall + 10 * NS_PER_S)
    last_recv = None
    nonmonotonic_receive = 0

    for rec in iter_records(raw_path):
        if rec.get("kind") != "stream":
            continue

        d = rec.get("data") or {}
        stream = str(rec.get("stream") or "")
        ts = int(rec["recv_wall_ns"])
        event = d.get("e")
        data_symbol = str(d.get("s") or "").upper()
        if data_symbol != symbol:
            continue

        if event == "aggTrade" and stream.endswith("@aggTrade"):
            n_trades += 1
            if ts <= upper_wall + ttl_ns + NS_PER_S:
                trades.append(
                    Trade(
                        ts, int(d["T"]) * NS_PER_MS,
                        float(d["p"]), float(d["q"]),
                        bool(d["m"]),
                    )
                )
        elif event == "bookTicker" or stream.endswith("@bookTicker"):
            n_books += 1
            if last_recv is not None and ts < last_recv:
                nonmonotonic_receive += 1
            last_recv = ts

            if ts < next_slot or ts > upper_wall:
                continue

            try:
                bid = float(d["b"])
                ask = float(d["a"])
                bid_qty = float(d["B"])
                ask_qty = float(d["A"])
            except (TypeError, KeyError, ValueError):
                continue

            if bid > 0 and ask > bid and bid_qty > 0 and ask_qty > 0:
                side = "BUY" if len(quotes) % 2 == 0 else "SELL"
                price = bid if side == "BUY" else ask
                qty = bid_qty if side == "BUY" else ask_qty
                quotes.append(
                    Quote(len(quotes), ts, side, price, qty,
                          bid, ask, ts + ttl_ns)
                )

            next_slot += interval_ns
            while next_slot <= ts:
                next_slot += interval_ns

    if nonmonotonic_receive:
        raise RuntimeError(
            f"bookTicker receive wall clock backtracks={nonmonotonic_receive}"
        )
    if any(trades[i].wall_ns > trades[i+1].wall_ns
           for i in range(len(trades)-1)):
        raise RuntimeError("aggTrade recv_wall_ns order backtracks")
    return quotes, trades, n_books, n_trades


def virtual_fill(quote: Quote, trades: list[Trade], start: int):
    """Exact same-side trade/queue logic as yue_maker VirtualOrder."""
    remaining = quote.displayed_qty
    for i in range(start, len(trades)):
        t = trades[i]
        if t.wall_ns > quote.deadline_wall_ns:
            break

        if t.wall_ns <= quote.wall_ns:
            continue

        if (quote.side == "BUY") != t.seller_aggressor:
            continue

        tolerance = max(1.0, abs(t.px), abs(quote.price)) * 1e-12
        equal = abs(t.px - quote.price) <= tolerance

        if quote.side == "BUY":
            crossed = t.px < quote.price and not equal
        else:
            crossed = t.px > quote.price and not equal

        if crossed:
            return t, "trade_through"

        if equal:
            remaining -= t.qty
            if remaining <= 0:
                return t, "queue_consumed"

    return None, None


def legacy_outcomes(quotes, trades):
    from bisect import bisect_right
    times = [t.wall_ns for t in trades]
    return [
        virtual_fill(q, trades, bisect_right(times, q.wall_ns))
        for q in quotes
    ]


def build_hbt(npz: Path):
    asset = (
        BacktestAsset()
        .data(str(npz))
        .linear_asset(1.0)
        .constant_order_latency(0, 0)
        .yue_strict_queue_model()
        .yue_strict_trade_only_exchange()
        .trading_value_fee_model(0.0, 0.0)
        .tick_size(0.01)
        .lot_size(0.01)
    )
    return HashMapMarketDepthBacktest([asset])


def move_to(hbt, timestamp, label):
    now = int(hbt.current_timestamp)
    if timestamp < now:
        raise RuntimeError(f"nonmonotonic HBT target {label}: {timestamp} < {now}")
    rc = int(hbt.elapse(timestamp - now))
    if rc != 0:
        raise RuntimeError(f"HBT elapse to {label} returned {rc}")


def run_hbt(npz, quotes, shift):
    hbt = build_hbt(npz)
    results = []

    try:
        rc = int(hbt.wait_next_feed(False, 60 * NS_PER_S))
        if rc != 2:
            raise RuntimeError(f"HBT init rc={rc}")

        for q in quotes:
            entry = int(q.wall_ns + shift)
            deadline = int(q.deadline_wall_ns + shift)
            move_to(hbt, entry, f"entry #{q.idx}")
            depth = hbt.depth(0)
            bid = float(depth.best_bid)
            ask = float(depth.best_ask)

            if q.side == "BUY":
                rc = int(hbt.submit_buy_order(
                    0, q.idx + 1, q.price, 0.01, GTX, LIMIT, True
                ))
            else:
                rc = int(hbt.submit_sell_order(
                    0, q.idx + 1, q.price, 0.01, GTX, LIMIT, True
                ))
            if rc != 0:
                raise RuntimeError(f"HBT submit #{q.idx} rc={rc}")

            order = hbt.orders(0).get(q.idx + 1)
            if order is None:
                raise RuntimeError(f"HBT no order after submit #{q.idx}")

            ack_status = int(order.status)
            if ack_status == NEW:
                move_to(hbt, deadline, f"TTL #{q.idx}")
                order = hbt.orders(0).get(q.idx + 1)
                if order is None:
                    raise RuntimeError(f"HBT no order after TTL #{q.idx}")
                if int(order.status) == NEW:
                    rc = int(hbt.cancel(0, q.idx + 1, True))
                    if rc != 0:
                        raise RuntimeError(f"HBT cancel #{q.idx} rc={rc}")
                    order = hbt.orders(0).get(q.idx + 1)

            status = int(order.status)
            if status not in (FILLED, CANCELED, EXPIRED):
                raise RuntimeError(
                    f"unexpected final status #{q.idx}: {status}"
                )

            results.append({
                "idx": q.idx,
                "status": status,
                "ack_status": ack_status,
                "exchange_ts": int(order.exch_timestamp),
                "bid_at_entry": bid,
                "ask_at_entry": ask,
                "local_book_matches_bookTicker": (
                    abs(bid - q.bid) < 1e-9
                    and abs(ask - q.ask) < 1e-9
                ),
                "executed_price": (
                    float(order.exec_price) if status == FILLED else None
                ),
            })
    finally:
        hbt.close()

    return results


def pctl(x, scale=1e6):
    if not x:
        return "n=0"
    arr = np.asarray(x, dtype=np.float64) / scale
    values = np.percentile(arr, [0, 50, 95, 99, 100])
    return (
        f"n={len(arr)} min={values[0]:.3f} "
        f"p50={values[1]:.3f} p95={values[2]:.3f} "
        f"p99={values[3]:.3f} max={values[4]:.3f}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture_dir", type=Path)
    ap.add_argument("npz", type=Path)
    ap.add_argument("--symbol", default="SOLUSDC")
    ap.add_argument("--interval-s", type=float, default=30.0)
    ap.add_argument("--ttl-s", type=float, default=4.123457)
    args = ap.parse_args()

    symbol = args.symbol.upper()
    raw_path = args.capture_dir / "raw.jsonl.zst"
    meta = args.npz.with_name(
        f"{symbol.lower()}_hbt_r0_1_meta.json"
    )
    if not raw_path.exists() or not meta.exists():
        raise RuntimeError(f"missing raw/meta: {raw_path} {meta}")

    info = json.loads(meta.read_text())
    if info.get("symbol") != symbol:
        raise RuntimeError("symbol metadata mismatch")
    shift = int(info["local_timestamp_shift_ns"])

    interval_ns = int(args.interval_s * NS_PER_S)
    ttl_ns = int(args.ttl_s * NS_PER_S)
    if interval_ns <= ttl_ns + 2 * NS_PER_S or ttl_ns <= 0:
        raise RuntimeError("require interval > TTL + 2s")

    with np.load(args.npz) as z:
        data = z["data"]
    mask = (data["ev"] & LOCAL_EVENT) == LOCAL_EVENT
    local = data[mask]
    start_hbt = int(local[0]["local_ts"])
    last_hbt = int(local[-1]["local_ts"])

    # Avoid pre-snapshot quotes and end-of-data orders.
    lower_wall = start_hbt - shift
    upper_wall = last_hbt - shift - ttl_ns - 3 * NS_PER_S

    quotes, trades, n_books, n_trades = raw_candidates(
        raw_path, symbol, interval_ns, ttl_ns,
        lower_wall, upper_wall,
    )

    if len(quotes) < 40:
        raise RuntimeError(f"insufficient raw bookTicker quotes: {len(quotes)}")

    old = legacy_outcomes(quotes, trades)
    native = run_hbt(args.npz, quotes, shift)

    if len(native) != len(old) or len(native) != len(quotes):
        raise RuntimeError("outcome length discrepancy")

    by_exchange_trade = defaultdict(list)
    for t in trades:
        by_exchange_trade[int(t.exch_ns)].append(t)

    count = Counter()
    different = []
    hbt_to_trade_recv = []
    old_minus_hbt_trade_recv = []
    raw_quote_to_hbt_book_price_mismatch = []

    for q, (legacy_trade, reason), h in zip(quotes, old, native, strict=True):
        old_filled = legacy_trade is not None
        hbt_filled = h["status"] == FILLED

        count["total"] += 1
        count["side_"+q.side] += 1
        count["legacy_filled" if old_filled else "legacy_not_filled"] += 1
        count["hbt_filled" if hbt_filled else "hbt_not_filled"] += 1
        count["hbt_"+(
            "FILLED" if hbt_filled
            else "CANCELED" if h["status"] == CANCELED else "EXPIRED"
        )] += 1
        count["both_filled" if old_filled and hbt_filled
              else "only_legacy" if old_filled
              else "only_hbt" if hbt_filled else "neither_filled"] += 1

        if not h["local_book_matches_bookTicker"]:
            count["hbt_depth_vs_bookTicker_best_mismatch"] += 1
            raw_quote_to_hbt_book_price_mismatch.append((
                q.idx, q.bid, q.ask, h["bid_at_entry"], h["ask_at_entry"]
            ))

        aligned = None
        trigger_recv = None
        if hbt_filled:
            trigger_trades = [
                t for t in by_exchange_trade.get(h["exchange_ts"], [])
                if ((q.side == "BUY") == t.seller_aggressor)
                and (t.px <= q.price + 1e-9 if q.side == "BUY"
                     else t.px >= q.price - 1e-9)
            ]
            if trigger_trades:
                trigger_recv = min(t.wall_ns for t in trigger_trades)
                # Feed receipt must not precede its corrected exchange time.
                hbt_to_trade_recv.append(
                    trigger_recv + shift - h["exchange_ts"]
                )
                count["hbt_exchange_fill_trade_received_in_raw"] += 1
            else:
                count["hbt_exchange_fill_without_matching_raw_trade"] += 1

        if old_filled and hbt_filled:
            if trigger_recv is not None:
                old_minus_hbt_trade_recv.append(
                    legacy_trade.wall_ns - trigger_recv
                )
            aligned = (legacy_trade.exch_ns == h["exchange_ts"])
            count["same_exchange_fill_time" if aligned
                  else "different_exchange_fill_time"] += 1

        if old_filled != hbt_filled or (old_filled and hbt_filled and not aligned):
            different.append({
                "id": q.idx, "side": q.side, "price": q.price,
                "legacy_fill_wall": (
                    legacy_trade.wall_ns if old_filled else None
                ),
                "legacy_fill_reason": reason,
                "legacy_fill_exch": (
                    legacy_trade.exch_ns if old_filled else None
                ),
                "hbt_status": (
                    "FILLED" if hbt_filled else
                    "CANCELED" if h["status"] == CANCELED else "EXPIRED"
                ),
                "hbt_fill_exch": h["exchange_ts"] if hbt_filled else None,
                "hbt_quoted_local_bid": h["bid_at_entry"],
                "hbt_quoted_local_ask": h["ask_at_entry"],
                "old_queue_displayed": q.displayed_qty,
            })

    print("===== HBT-R0.8 RAW WALL-CLOCK / HBT EXECUTION BRIDGE =====")
    print(f"symbol={symbol}")
    print(f"raw={raw_path}")
    print(f"npz={args.npz}")
    print(f"raw_bookTicker_records={n_books}")
    print(f"raw_aggTrade_records={n_trades}")
    print(f"quote_candidates={len(quotes)}")
    print(f"scheduled_spacing_ns={interval_ns}")
    print(f"ttl_ns={ttl_ns}")
    print(f"hbt_local_global_shift_ns={shift}")
    print("order_selection=raw_bookTicker_alternating_BUY_SELL")
    print("legacy_reference=virtual_fill_on_recv_wall_ns_no_order_latency")
    print("hbt=YueStrictQueueModel_TradeOnlyExchange_GTX_zero_order_latency")
    print("full_legacy_strategy_parity=NOT_TESTED")

    print()
    print("===== OUTCOME CROSS-TAB =====")
    for k in (
        "total", "side_BUY", "side_SELL", "legacy_filled", "legacy_not_filled",
        "hbt_filled", "hbt_not_filled", "hbt_CANCELED", "hbt_EXPIRED",
        "both_filled", "only_legacy", "only_hbt", "neither_filled",
        "same_exchange_fill_time", "different_exchange_fill_time",
        "hbt_depth_vs_bookTicker_best_mismatch",
        "hbt_exchange_fill_trade_received_in_raw",
        "hbt_exchange_fill_without_matching_raw_trade",
    ):
        print(f"{k}={count[k]}")

    print()
    print("===== CROSS-CLOCK DIAGNOSTICS =====")
    print("hbt_fill_exchange_to_raw_trade_receive_corrected_ms="+
          pctl(hbt_to_trade_recv))
    print("legacy_fill_wall_minus_hbt_trigger_receive_ms="+
          pctl(old_minus_hbt_trade_recv))
    print(f"outcome_or_fill_time_differences={len(different)}")
    print(f"quote_best_mismatches={len(raw_quote_to_hbt_book_price_mismatch)}")
    for item in different[:16]:
        print("DIFF " + json.dumps(item, sort_keys=True))

    print()
    print("===== INTERPRETATION =====")
    print(
        "Differences are expected because old VirtualOrder uses bookTicker "
        "displayed queue and aggTrade receipt timestamps while HBT uses "
        "exchange depth/queue and exchange-trade timestamps; old VirtualOrder "
        "does not implement GTX reject. No synthetic parity is asserted."
    )
    print(
        "R0_8_BRIDGE_STATUS=PASS"
        if len(quotes) >= 40
        and count["total"] == len(quotes)
        and count["hbt_exchange_fill_without_matching_raw_trade"] == 0
        else "R0_8_BRIDGE_STATUS=FAIL"
    )
    print("FULL_YUE_MAKER_STRATEGY_PARITY=NOT_TESTED")
    raise SystemExit(
        0 if len(quotes) >= 40 and
        count["hbt_exchange_fill_without_matching_raw_trade"] == 0
        else 1
    )


if __name__ == "__main__":
    main()
