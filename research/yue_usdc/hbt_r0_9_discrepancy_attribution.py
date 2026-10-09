"""R0.9: event-level attribution of raw bookTicker/VirtualOrder vs HBT fills.

Reuses R0.8's frozen deterministic quote selection and BOTH replay paths.
Examines the exact order-entry exchange book, queue at quote vs queue at
exchange acceptance, GTX status, and corresponding trade timelines for every
mismatch. This is diagnostic, not a claim of full legacy strategy parity.

The exchange book is rebuilt independently from EXCH_DEPTH rows in the R0.1
NPZ; the local BBO shown is that obtained from the actual HBT replay.
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter
import json
from pathlib import Path

import numpy as np

from hftbacktest import BUY_EVENT, SELL_EVENT, DEPTH_EVENT, DEPTH_SNAPSHOT_EVENT
from hftbacktest import DEPTH_CLEAR_EVENT, EXCH_EVENT, LOCAL_EVENT
from hftbacktest.order import FILLED, CANCELED, EXPIRED

from hbt_r0_5_batch_parity import BASE_EVENT_MASK, apply_depth, reference_trade_fill
from hbt_r0_8_wallclock_bridge import (
    NS_PER_MS, NS_PER_S, raw_candidates, legacy_outcomes, run_hbt,
)


def price_matches(a, b):
    return abs(float(a) - float(b)) <= 1e-9


def load_fixtures(capture_dir, npz, symbol, step_s, ttl_s):
    metadata_path = npz.with_name(f"{symbol.lower()}_hbt_r0_1_meta.json")
    meta = json.loads(metadata_path.read_text())
    if meta.get("symbol") != symbol:
        raise RuntimeError("metadata symbol mismatch")
    shift = int(meta["local_timestamp_shift_ns"])

    with np.load(npz) as z:
        data = z["data"]

    local = data[(data["ev"] & LOCAL_EVENT) == LOCAL_EVENT]
    if len(local) == 0:
        raise RuntimeError("no local event data")

    ttl_ns = int(ttl_s * NS_PER_S)
    step_ns = int(step_s * NS_PER_S)
    if ttl_ns <= 0 or step_ns <= ttl_ns + 2 * NS_PER_S:
        raise ValueError("step must be > TTL + 2 seconds")

    lower_wall = int(local[0]["local_ts"]) - shift
    upper_wall = int(local[-1]["local_ts"]) - shift - ttl_ns - 3 * NS_PER_S

    quotes, trades, _, _ = raw_candidates(
        capture_dir / "raw.jsonl.zst",
        symbol, step_ns, ttl_ns, lower_wall, upper_wall,
    )
    old = legacy_outcomes(quotes, trades)
    hbt = run_hbt(npz, quotes, shift)
    exch = data[(data["ev"] & EXCH_EVENT) == EXCH_EVENT]

    if not (len(quotes) == len(old) == len(hbt)):
        raise RuntimeError("paired replay length mismatch")
    return quotes, trades, old, hbt, exch, shift


def status_name(value):
    return {
        FILLED: "FILLED", CANCELED: "CANCELED", EXPIRED: "EXPIRED",
    }.get(int(value), str(value))


def first_deviation_indices(quotes, old, hbt):
    indices = []
    for i, (_, (legacy_trade, _), native) in enumerate(zip(
        quotes, old, hbt, strict=True
    )):
        old_filled = legacy_trade is not None
        hbt_filled = native["status"] == FILLED
        if old_filled != hbt_filled or (
            old_filled and hbt_filled
            and legacy_trade.exch_ns != native["exchange_ts"]
        ):
            indices.append(i)
    return indices


def exchange_entry_state(exch, quote, shift):
    """Rebuild exchange book through the order-entry timestamp (inclusive)."""
    target_ns = quote.wall_ns + shift
    exch_ts = exch["exch_ts"]
    upper = int(np.searchsorted(exch_ts, target_ns, side="right"))
    bids, asks = {}, {}

    for row in exch[:upper]:
        apply_depth(row, bids, asks)

    if not bids or not asks:
        raise RuntimeError(f"empty exchange book at quote #{quote.idx}")

    exchange_bid = max(bids)
    exchange_ask = min(asks)
    sidebook = bids if quote.side == "BUY" else asks
    ahead = float(sidebook.get(quote.price, 0.0))
    post_only_reject = (
        quote.price >= exchange_ask
        if quote.side == "BUY"
        else quote.price <= exchange_bid
    )

    deadline = quote.deadline_wall_ns + shift
    ref_fill, ref_reason = None, None
    if not post_only_reject:
        ref_fill, ref_reason = reference_trade_fill(
            exch, exch_ts, upper, deadline, quote.side,
            quote.price, ahead,
        )

    return {
        "exch_entry_ts": target_ns,
        "exch_best_bid": exchange_bid,
        "exch_best_ask": exchange_ask,
        "exch_qty_at_quote": ahead,
        "exch_depth_events_through_entry": upper,
        "gtx_should_expire": post_only_reject,
        "exch_trade_reference_fill": ref_fill,
        "exch_trade_reference_reason": ref_reason,
    }


def raw_trade_trace(quote, trades, shift, native_fill_ns, max_items):
    """Events at/through the resting quote, annotated with both clock domains."""
    from bisect import bisect_left
    times = [t.wall_ns for t in trades]
    lookback_ns = 250 * NS_PER_MS
    start = bisect_left(times, quote.wall_ns - lookback_ns)
    upper = quote.deadline_wall_ns + 350 * NS_PER_MS

    same_price_cum_raw = 0.0
    eligible = []
    behind = 0
    before_recv = 0

    for t in trades[start:]:
        if t.wall_ns > upper:
            break
        correct_aggressor = (
            (quote.side == "BUY") == t.seller_aggressor
        )
        if not correct_aggressor:
            continue

        same_px = price_matches(t.px, quote.price)
        crosses = (
            t.px < quote.price - 1e-9 if quote.side == "BUY"
            else t.px > quote.price + 1e-9
        )
        if not (same_px or crosses):
            continue

        before_quote = t.wall_ns <= quote.wall_ns
        before_recv += int(before_quote)
        if not before_quote and t.wall_ns <= quote.deadline_wall_ns and same_px:
            same_price_cum_raw += t.qty

        if native_fill_ns is not None and t.exch_ns == native_fill_ns:
            label = "HBT_FILL_EVENT"
        else:
            label = "TRADE"

        eligible.append({
            "trade_exch_ns": t.exch_ns,
            "raw_recv_ns": t.wall_ns,
            "recv_after_quote_ms": round(
                (t.wall_ns - quote.wall_ns) / NS_PER_MS, 3
            ),
            "exch_after_hbt_entry_ms": round(
                (t.exch_ns - quote.wall_ns - shift) / NS_PER_MS, 3
            ),
            "trade_recv_minus_corrected_exch_ms": round(
                (t.wall_ns + shift - t.exch_ns) / NS_PER_MS, 3
            ),
            "price": t.px,
            "qty": t.qty,
            "kind": "trade_through" if crosses else "same_price",
            "before_quote_receipt": before_quote,
            "after_quote_ttl": t.wall_ns > quote.deadline_wall_ns,
            "label": label,
        })

    shown = eligible[:max_items]
    if native_fill_ns is not None:
        fills = [e for e in eligible if e["trade_exch_ns"] == native_fill_ns]
        if fills and not any(
            e["trade_exch_ns"] == native_fill_ns for e in shown
        ):
            shown.extend(fills[:2])

    return {
        "eligible_aggressive_trades_in_window": len(eligible),
        "eligible_received_before_quote": before_recv,
        "same_price_qty_after_quote_before_ttl": round(
            same_price_cum_raw, 8
        ),
        "trace_truncated": len(eligible) > len(shown),
        "events": shown,
    }


def classify(quote, old_trade, native, entry):
    if native["status"] == EXPIRED:
        return (
            "GTX_REJECT_AT_EXCHANGE"
            if entry["gtx_should_expire"] else
            "GTX_STATUS_NEEDS_AUDIT"
        )

    if (not price_matches(quote.bid, native["bid_at_entry"])
            or not price_matches(quote.ask, native["ask_at_entry"])):
        return "BOOKTICKER_L2_LOCAL_BBO_DIVERGENCE"

    if not price_matches(quote.displayed_qty, entry["exch_qty_at_quote"]):
        return "BOOKTICKER_VS_EXCHANGE_QUEUE_AHEAD_DIVERGENCE"

    if old_trade is not None and native["status"] == FILLED:
        return "TRADE_ORDER_OR_CLOCK_DOMAIN_DIVERGENCE"

    return "FILL_STATUS_OR_EXCHANGE_MODEL_REQUIRES_EVENT_REVIEW"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("capture_dir", type=Path)
    p.add_argument("npz", type=Path)
    p.add_argument("--symbol", default="SOLUSDC")
    p.add_argument("--interval-s", type=float, default=30.0)
    p.add_argument("--ttl-s", type=float, default=4.123457)
    p.add_argument("--max-events", type=int, default=16)
    args = p.parse_args()

    quotes, trades, old, hbt, exch, shift = load_fixtures(
        args.capture_dir, args.npz, args.symbol.upper(),
        args.interval_s, args.ttl_s,
    )

    indices = first_deviation_indices(quotes, old, hbt)
    print("===== HBT-R0.9 ORDER-LEVEL DISCREPANCY ATTRIBUTION =====")
    print(f"quotes={len(quotes)}")
    print(f"mismatch_order_count={len(indices)}")
    print(f"mismatch_ids={[quotes[i].idx for i in indices]}")
    print(f"hbt_clock_shift_ns={shift}")
    print(
        "reference=raw_bookTicker_display_qty+raw_aggTrade_recv_wall "
        "versus HBT_strict_exchange_clock"
    )

    if not indices:
        print("R0_9_DIAGNOSTIC_STATUS=NO_DISCREPANCIES")
        return

    classifications = Counter()
    severe_engine_disagreements = 0

    for i in indices:
        q = quotes[i]
        old_trade, reason = old[i]
        native = hbt[i]
        entry = exchange_entry_state(exch, q, shift)
        old_filled = old_trade is not None
        hbt_filled = native["status"] == FILLED

        actual_reference_match = (
            (native["status"] == EXPIRED and entry["gtx_should_expire"])
            or (
                not entry["gtx_should_expire"]
                and (
                    (hbt_filled and
                     native["exchange_ts"] == entry["exch_trade_reference_fill"])
                    or (native["status"] == CANCELED
                        and entry["exch_trade_reference_fill"] is None)
                )
            )
        )

        if not actual_reference_match:
            severe_engine_disagreements += 1

        classification = classify(q, old_trade, native, entry)
        classifications[classification] += 1
        print()
        print(f"===== ORDER #{q.idx} {q.side} =====")
        print(f"classification_hint={classification}")
        print(f"raw_quote_recv_ns={q.wall_ns}")
        print(f"hbt_entry_exch_ns={entry['exch_entry_ts']}")
        print(f"quote_price={q.price:.4f}")
        print(f"raw_bookTicker_bid_ask={q.bid:.4f}/{q.ask:.4f}")
        print(
            f"hbt_local_bid_ask="
            f"{native['bid_at_entry']:.4f}/{native['ask_at_entry']:.4f}"
        )
        print(
            f"hbt_exchange_bid_ask="
            f"{entry['exch_best_bid']:.4f}/{entry['exch_best_ask']:.4f}"
        )
        print(f"raw_bookTicker_initial_queue={q.displayed_qty:.8f}")
        print(f"exch_book_qty_at_quote={entry['exch_qty_at_quote']:.8f}")
        print(f"gtx_should_expire={entry['gtx_should_expire']}")
        print(f"legacy_filled={old_filled}")
        print(f"legacy_reason={reason}")
        print(
            "legacy_fill_raw_recv_ns="
            f"{old_trade.wall_ns if old_filled else None}"
        )
        print(
            "legacy_fill_exch_ns="
            f"{old_trade.exch_ns if old_filled else None}"
        )
        print(f"hbt_status={status_name(native['status'])}")
        print(
            "hbt_fill_exch_ns="
            f"{native['exchange_ts'] if hbt_filled else None}"
        )
        print(
            "exchange_reference_fill_ns="
            f"{entry['exch_trade_reference_fill']}"
        )
        print(
            "exchange_reference_reason="
            f"{entry['exch_trade_reference_reason']}"
        )
        print(f"strict_exchange_reference_matches_hbt={actual_reference_match}")

        timeline = raw_trade_trace(
            q, trades, shift,
            native["exchange_ts"] if hbt_filled else None,
            max(0, args.max_events),
        )
        print(
            "raw_same_price_agg_qty_until_ttl="
            f"{timeline['same_price_qty_after_quote_before_ttl']:.8f}"
        )
        print(
            "eligible_trade_events="
            f"{timeline['eligible_aggressive_trades_in_window']}"
            f" (showing <= {args.max_events} plus HBT trigger)"
        )
        for row in timeline["events"]:
            print("  TRADE " + json.dumps(row, sort_keys=True))

    print()
    print("===== SUMMARY =====")
    for k, v in sorted(classifications.items()):
        print(f"{k}={v}")
    print(f"strict_exchange_reference_disagreements={severe_engine_disagreements}")
    print(f"expected_diagnostic_cases={len(indices)}")
    print(
        "CLASSIFICATIONS_ARE_HINTS=TRUE "
        "(book/queue/clock causes can coexist for one order)"
    )
    print(
        "R0_9_DIAGNOSTIC_STATUS="
        + ("PASS" if severe_engine_disagreements == 0 else "ENGINE_DIFF")
    )
    print("FULL_LEGACY_STRATEGY_PARITY=NOT_TESTED")
    raise SystemExit(0 if severe_engine_disagreements == 0 else 1)


if __name__ == "__main__":
    main()
