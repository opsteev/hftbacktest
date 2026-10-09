"""R1.1: paired R8.1 intent-tape fill-quality and raw receipt markouts.

Read R1.0 paired order CSVs. Independently reindex immutable SOLUSDC
raw bookTicker and aggTrade messages by receive-wall time, then score
old VirtualOrder fills against HBT Strict executions with the SAME raw
future-mid rule.

For HBT fills, use the earliest raw aggTrade receipt on the matching
exchange timestamp, aggressor side, and trade-through/same-price quote.
This is an observable-fill-receipt PROXY; if multiple eligible aggTrades
share that timestamp, exact triggering message remains ambiguous and is
reported. HBT exchange time is NOT treated as directly observable.

A quote is sampled at the first raw bookTicker reception at or after
(anchor_receipt + horizon), subject to --max-quote-lag-ms. The resulting
signed markout is gross execution quality, not strategy or portfolio P&L.
All decisions remain frozen from the old strategy: off-policy overlap
and selection bias are quantified, not silently eliminated.

NO live fill verification, no order-entry latency (from R1.0), no
inventory dynamics, no unwind/taker fees or funding. Development hour.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import statistics

from hbt_r0_8_wallclock_bridge import iter_records

NS_PER_MS = 1_000_000
NS_PER_S = 1_000_000_000
HORIZONS_MS = (0, 100, 1000, 5000)
STRATEGIES = (
    "hysteresis_50ms", "hysteresis_100ms", "hysteresis_250ms",
)


def parse_bool(x):
    return str(x).strip().lower() == "true"


def maybe_int(x):
    return int(x) if x not in ("", None) else None


def load_tape(path: Path):
    with path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise RuntimeError(f"empty paired CSV: {path}")
    ids = [int(r["order_id"]) for r in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError(f"duplicate order IDs in {path}")
    return rows


def collect_raw_market(raw_path: Path, symbol: str):
    quote_timestamps = []
    mids = []
    spreads = []
    trades = defaultdict(list)
    counts = Counter()
    last_quote_ts = None

    for record in iter_records(raw_path):
        if record.get("kind") != "stream":
            continue
        data = record.get("data") or {}
        if str(data.get("s", "")).upper() != symbol:
            continue
        stream = str(record.get("stream") or "")
        recv = int(record["recv_wall_ns"])

        if stream.endswith("@bookTicker") or data.get("e") == "bookTicker":
            try:
                bid = float(data["b"])
                ask = float(data["a"])
            except (TypeError, KeyError, ValueError):
                counts["bad_book_ticker"] += 1
                continue
            if not (math.isfinite(bid) and math.isfinite(ask)
                    and bid > 0 and ask > bid):
                counts["bad_book_ticker"] += 1
                continue
            if last_quote_ts is not None and recv < last_quote_ts:
                raise RuntimeError(
                    f"raw bookTicker recv_wall_ns backtracked at {recv}"
                )
            quote_timestamps.append(recv)
            mids.append((bid + ask) / 2)
            spreads.append((ask - bid) / ((ask + bid) / 2) * 10_000)
            last_quote_ts = recv
            counts["book_ticker"] += 1

        elif stream.endswith("@aggTrade") or data.get("e") == "aggTrade":
            try:
                exchange_ts = int(data["T"]) * NS_PER_MS
                px = float(data["p"])
                qty = float(data["q"])
                seller_aggressor = bool(data["m"])
            except (TypeError, KeyError, ValueError):
                counts["bad_agg_trade"] += 1
                continue
            trades[exchange_ts].append(
                (recv, px, qty, seller_aggressor)
            )
            counts["agg_trade"] += 1

    if not quote_timestamps or not trades:
        raise RuntimeError("raw market missing SOLUSDC bookTicker/aggTrade")
    return quote_timestamps, mids, spreads, trades, counts


def eligible_aggtrades(order_side, price, at_ns, trades):
    cand = []
    tol = max(1.0, abs(price)) * 1e-12
    for recv, px, qty, seller_aggressor in trades.get(at_ns, []):
        if order_side == "BUY":
            eligible = seller_aggressor and px <= price + tol
        else:
            eligible = (not seller_aggressor) and px >= price - tol
        if eligible:
            cand.append((recv, px, qty))
    return sorted(cand)


def quality_at_fill(
    side, px, fill_receipt_ns, quote_timestamps, mids,
    horizon_ms, max_sample_delay_ns,
):
    target = fill_receipt_ns + horizon_ms * NS_PER_MS
    i = bisect_left(quote_timestamps, target)
    if i >= len(quote_timestamps):
        return None, None
    observed = quote_timestamps[i]
    sample_lag = observed - target
    if sample_lag > max_sample_delay_ns:
        return None, sample_lag
    mid = mids[i]
    if side == "BUY":
        signed = (mid - px) / px * 10_000
    else:
        signed = (px - mid) / px * 10_000
    return signed, sample_lag


def summarize_values(xs):
    if not xs:
        return {
            "n": 0,
            "mean_bps": None,
            "median_bps": None,
            "p10_bps": None,
            "p90_bps": None,
            "positive_fraction": None,
        }
    seq = sorted(xs)
    return {
        "n": len(seq),
        "mean_bps": statistics.fmean(seq),
        "median_bps": statistics.median(seq),
        "p10_bps": seq[min(len(seq)-1, int(.1*(len(seq)-1)))],
        "p90_bps": seq[min(len(seq)-1, int(.9*(len(seq)-1)))],
        "positive_fraction": sum(v > 0 for v in seq) / len(seq),
    }


def fmt_summary(result):
    n = result["n"]
    if n == 0:
        return "n=0"
    return (
        f"n={n} mean_bps={result['mean_bps']:.4f} "
        f"median_bps={result['median_bps']:.4f} "
        f"p10_bps={result['p10_bps']:.4f} "
        f"p90_bps={result['p90_bps']:.4f} "
        f"positive_fraction={result['positive_fraction']:.3f}"
    )


def compare_tape(
    tape, quotes_ts, mids, raw_trades, max_lag_ns, shift
):
    rows = []
    buckets = defaultdict(list)
    counts = Counter()
    sample_lags = defaultdict(list)

    for original in tape:
        side = original["side"].upper()
        if side not in ("BUY", "SELL"):
            raise ValueError(f"bad side {side}")
        px = float(original["price"])
        old_fill = original["old_status"] == "filled"
        hbt_fill = original["hbt_status"] == "FILLED"
        both = old_fill and hbt_fill
        bucket = (
            "both" if both else
            "only_legacy" if old_fill else
            "only_hbt" if hbt_fill else "neither"
        )
        clean = (
            original["hbt_ack_status"] != "EXPIRED"
            and not parse_bool(original["hbt_local_price_diff"])
            and int(original["other_hbt_orders_live_at_entry"]) == 0
        )
        old_pending = parse_bool(original["old_pending_fill"])
        counts["orders"] += 1
        counts["bucket_" + bucket] += 1
        counts["clean_orders"] += int(clean)
        counts["gtx_rejected"] += int(
            original["hbt_ack_status"] == "EXPIRED"
        )
        counts["local_price_diff"] += int(
            parse_bool(original["hbt_local_price_diff"])
        )
        counts["hbt_live_overlap"] += int(
            int(original["other_hbt_orders_live_at_entry"]) > 0
        )
        counts["legacy_filled_while_cancel_pending"] += int(
            old_fill and old_pending
        )

        candidate_count = 0
        hbt_receipt = None
        hbt_recv_span_ms = None
        if hbt_fill:
            exchts = maybe_int(original["hbt_fill_exchange_ns"])
            if exchts is None:
                raise RuntimeError(
                    "FILLED order missing HBT exchange timestamp"
                )
            eligible = eligible_aggtrades(side, px, exchts, raw_trades)
            candidate_count = len(eligible)
            counts["hbt_fills"] += 1
            if eligible:
                hbt_receipt = eligible[0][0]
                if len(eligible) > 1:
                    counts["hbt_ambiguous_trigger_timestamp"] += 1
                hbt_recv_span_ms = (
                    (eligible[-1][0] - eligible[0][0]) / NS_PER_MS
                )
            else:
                counts["hbt_no_raw_trade_at_fill_timestamp"] += 1

        old_receipt = maybe_int(original["old_fill_wall_ns"])
        if old_fill and old_receipt is None:
            raise RuntimeError("legacy FILLED missing fill_wall_ns")

        measures = {}
        for fill_source, active, anchor in (
            ("legacy", old_fill, old_receipt),
            ("hbt", hbt_fill, hbt_receipt),
        ):
            for horizon in HORIZONS_MS:
                key = f"{fill_source}_markout_{horizon}ms_bps"
                lag_key = f"{fill_source}_sample_lag_{horizon}ms_ns"
                value, lag = (
                    quality_at_fill(
                        side, px, anchor, quotes_ts, mids, horizon,
                        max_lag_ns,
                    )
                    if active and anchor is not None
                    else (None, None)
                )
                measures[key] = value
                measures[lag_key] = lag

                if value is not None:
                    sample_lags[(fill_source, horizon)].append(lag)
                    buckets[(fill_source, "all", horizon)].append(value)
                    buckets[(fill_source, bucket, horizon)].append(value)
                    if clean:
                        buckets[(fill_source, "clean", horizon)].append(value)
                    if old_pending and fill_source == "legacy":
                        buckets[(fill_source, "cancel_pending", horizon)].append(value)
                    if old_pending and fill_source == "hbt" and old_fill:
                        buckets[(fill_source, "legacy_cancel_pending_pair", horizon)].append(value)
                elif active:
                    counts[f"{fill_source}_missing_markout_{horizon}ms"] += 1

            if active:
                counts[fill_source+"_filled"] += 1

        if both and hbt_receipt is not None:
            counts["paired_receipt_delta_count"] += 1
            delta_ms = (old_receipt - hbt_receipt) / NS_PER_MS
            measures["old_minus_hbt_receipt_ms"] = delta_ms
            for horizon in HORIZONS_MS:
                old_v = measures[f"legacy_markout_{horizon}ms_bps"]
                hbt_v = measures[f"hbt_markout_{horizon}ms_bps"]
                if old_v is not None and hbt_v is not None:
                    buckets[("difference", "both", horizon)].append(hbt_v-old_v)

        rows.append({
            **original,
            "outcome_bucket": bucket,
            "clean_comparable_intent": clean,
            "hbt_trigger_raw_trade_candidates": candidate_count,
            "hbt_trigger_raw_recv_ns": hbt_receipt,
            "hbt_trigger_recv_span_ms": hbt_recv_span_ms,
            **measures,
        })

    out_summary = {}
    for (src, name, h), values in sorted(buckets.items()):
        out_summary[f"{src}/{name}/{h}ms"] = summarize_values(values)

    print("orders_total="+str(counts["orders"]))
    for name in (
        "bucket_both", "bucket_only_legacy", "bucket_only_hbt",
        "bucket_neither", "legacy_filled", "hbt_filled",
        "gtx_rejected", "local_price_diff", "hbt_live_overlap",
        "clean_orders", "legacy_filled_while_cancel_pending",
        "hbt_no_raw_trade_at_fill_timestamp",
        "hbt_ambiguous_trigger_timestamp",
    ):
        print(f"{name}={counts[name]}")

    print()
    print("===== MARKOUT QUALITY: SAME RAW BOOKTICKER RECEIPT CLOCK =====")
    for horizon in (100, 1000, 5000):
        print(f"-- horizon_ms={horizon}")
        for source, group in (
            ("legacy", "all"),
            ("hbt", "all"),
            ("legacy", "clean"),
            ("hbt", "clean"),
            ("legacy", "only_legacy"),
            ("hbt", "only_hbt"),
            ("legacy", "cancel_pending"),
            ("difference", "both"),
        ):
            vals = buckets[(source, group, horizon)]
            print(f"  {source}/{group}: {fmt_summary(summarize_values(vals))}")

    print()
    print("===== OBSERVATION COMPLETENESS =====")
    for src in ("legacy", "hbt"):
        for h in HORIZONS_MS:
            print(
                f"{src}_missing_markout_{h}ms="
                f"{counts[f'{src}_missing_markout_{h}ms']}"
            )
    if counts["hbt_no_raw_trade_at_fill_timestamp"]:
        print(
            "CAUTION=HBT fills with no eligible raw aggTrade receipt "
            "cannot receive a markout; investigate execution provenance."
        )
    if counts["hbt_ambiguous_trigger_timestamp"]:
        print(
            "CAUTION=multiple matching raw trades share exchange fill "
            "timestamp; earliest receipt is an approximation."
        )

    return counts, rows, out_summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw", type=Path)
    ap.add_argument("paired_csv_dir", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--symbol", default="SOLUSDC")
    ap.add_argument("--max-quote-lag-ms", type=int, default=500)
    ap.add_argument("--strategies", default=",".join(STRATEGIES))
    args = ap.parse_args()

    if args.max_quote_lag_ms <= 0:
        ap.error("--max-quote-lag-ms must be positive")
    if not args.raw.is_file():
        raise FileNotFoundError(args.raw)

    quotes_ts, mids, spreads, raw_trades, feed_count = collect_raw_market(
        args.raw, args.symbol.upper(),
    )

    out_root = args.output
    out_root.mkdir(parents=True, exist_ok=True)
    summary = {
        "stage": "R1.1",
        "quote_source": "raw_bookTicker_recv_wall_ns",
        "fill_sources": (
            "legacy_raw_recv_wall_ns",
            "HBT_matching_aggTrade_raw_recv_wall_ns_proxy",
        ),
        "method": (
            "gross_signed_fill_to_future_mid_bps; first raw quote at or "
            "after receipt_anchor+horizon, max lag enforced; "
            "no entry latency, no inventory/unwind PnL"
        ),
        "bookticker_quotes": len(quotes_ts),
        "aggtrade_records": feed_count["agg_trade"],
        "strategies": {},
    }
    print("===== HBT-R1.1 ORIGINAL R8.1 VS HBT FILL QUALITY =====")
    print(f"raw={args.raw}")
    print(f"paired_csv_dir={args.paired_csv_dir}")
    print(f"bookTicker_records={len(quotes_ts)}")
    print(f"aggTrade_records={feed_count['agg_trade']}")
    print(f"max_quote_observation_lag_ms={args.max_quote_lag_ms}")
    print(
        "HBT_trigger_receipt=earliest_raw_aggTrade_matching_exchange_ts_and_side"
    )
    print(
        "metric=gross_signed_markout_bps_from_fill_price_to_future_raw_mid"
    )
    print("FROZEN_INTENT_NOT_CLOSED_LOOP=TRUE")
    print("PORTFOLIO_PNL_NOT_TESTED=TRUE")

    total_missing = 0
    selection = [
        x.strip() for x in args.strategies.split(",") if x.strip()
    ]
    for strategy in selection:
        path = args.paired_csv_dir / (
            f"r1_0_paired_orders_{strategy}.csv"
        )
        tape = load_tape(path)
        print()
        print(f"===== {strategy.upper()} =====")
        counts, rows, stats = compare_tape(
            tape, quotes_ts, mids, raw_trades,
            args.max_quote_lag_ms * NS_PER_MS, 0
        )
        total_missing += counts["hbt_no_raw_trade_at_fill_timestamp"]
        output_csv = out_root / f"r1_1_fill_quality_{strategy}.csv"
        with output_csv.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        summary["strategies"][strategy] = {
            "counts": dict(counts),
            "markout_quality": stats,
            "per_order_csv": str(output_csv),
        }
        print(f"per_order_csv={output_csv}")

    out_json = out_root / "r1_1_summary.json"
    out_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print()
    print(f"summary={out_json}")
    print("R1_1_DIAGNOSTIC_STATUS="+(
        "COMPLETE" if total_missing == 0 else
        "INCOMPLETE_MISSING_HBT_FILL_TRIGGERS"
    ))
    print(
        "LIMITATION=one-hour development capture; raw receipt markouts "
        "are not exchange-mid, queue truth, realized PnL, "
        "nor closed-loop strategy replay."
    )
    raise SystemExit(0 if total_missing == 0 else 1)


if __name__ == "__main__":
    main()
