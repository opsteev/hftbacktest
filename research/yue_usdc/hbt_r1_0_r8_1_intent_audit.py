"""R1.0 — Frozen R8.1 actual-strategy order-intent execution audit.

Consume unchanged yue_maker.r8_hysteresis CSV exports directly.
Use the original entry_wall_ns, quote price, side, quote TTL, cancel
request/effective timestamps and original VirtualOrder outcomes.

HBT receives each intent as an actual GTX limit order at
  entry_wall_ns + R0.1 globally corrected local clock shift.
An existing strategy cancellation takes effect on exchange at the
recorded legacy cancel_effective_ns; TTL is enforced via an HBT
cancellation at the legacy expiry clock.

This is a FROZEN-INTENT counterfactual audit. Legacy signal/entry/cancel
decisions are NOT recomputed from HBT fills; when the new execution
diverges from the original VirtualOrder outcome, later legacy entries
may be off-policy (including overlapping orders). The script reports
overlap explicitly and never calls the result full-strategy parity.

Zero extra order-entry latency (separate from legacy cancellation
latency already embedded in the CSV). Uses same one-hour dev capture.
No assertion of live queue position or realized P&L.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from hftbacktest import BacktestAsset, HashMapMarketDepthBacktest, GTX, LIMIT
from hftbacktest.order import NEW, FILLED, CANCELED, EXPIRED


NS_PER_S = 1_000_000_000
STRATEGIES = ("hysteresis_50ms", "hysteresis_100ms", "hysteresis_250ms")


def maybe_int(x):
    return int(x) if x not in (None, "") else None


@dataclass(frozen=True)
class Intent:
    order_id: int
    strategy: str
    side: str
    price: float
    entry_wall: int
    expiry_wall: int
    cancel_req_wall: int | None
    cancel_eff_wall: int | None
    old_status: str
    old_fill_wall: int | None
    old_reason: str | None
    old_queue_qty: float
    old_pending_fill: bool

    @property
    def stop_wall(self):
        # In the counterfactual replay, an unfilled quote lives until TTL
        # or an effective cancellation, whichever becomes effective first.
        if self.cancel_eff_wall is not None:
            return min(self.expiry_wall, self.cancel_eff_wall)
        return self.expiry_wall


def load_intents(path: Path, strategy: str):
    rows = []
    with path.open(newline="") as fh:
        for rec in csv.DictReader(fh):
            side = rec["side"].upper()
            if side not in ("BUY", "SELL"):
                raise ValueError(f"unsupported side: {side}")

            rows.append(Intent(
                order_id=int(rec["order_id"]),
                strategy=strategy,
                side=side,
                price=float(rec["entry_price"]),
                entry_wall=int(rec["entry_wall_ns"]),
                expiry_wall=int(rec["expires_wall_ns"]),
                cancel_req_wall=maybe_int(rec["cancel_requested_ns"]),
                cancel_eff_wall=maybe_int(rec["cancel_effective_ns"]),
                old_status=rec["terminal_status"],
                old_fill_wall=maybe_int(rec["fill_wall_ns"]),
                old_reason=rec["fill_reason"] or None,
                old_queue_qty=float(rec["initial_queue_ahead"]),
                old_pending_fill=rec["filled_while_cancel_pending"] == "True",
            ))
    rows.sort(key=lambda r: (r.entry_wall, r.order_id))
    if any(rows[i].order_id == rows[i + 1].order_id for i in range(len(rows)-1)):
        raise ValueError("duplicate order IDs in R8.1 CSV")
    if any(row.stop_wall < row.entry_wall for row in rows):
        raise ValueError("negative-order-lifetime in legacy CSV")
    return rows


def model(npz: Path):
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


def step_until(hbt, target, label):
    now = int(hbt.current_timestamp)
    if target < now:
        raise RuntimeError(
            f"HBT time reversed at {label}: target={target} current={now}"
        )
    rc = int(hbt.elapse(target - now))
    if rc == 1:
        return False
    if rc != 0:
        raise RuntimeError(f"elapse {label} returned rc={rc}")
    if int(hbt.current_timestamp) != target:
        raise RuntimeError(f"HBT did not reach target {label}")
    return True


def hbt_state(order):
    return (
        "FILLED" if int(order.status) == FILLED
        else "CANCELED" if int(order.status) == CANCELED
        else "EXPIRED" if int(order.status) == EXPIRED
        else "NEW" if int(order.status) == NEW
        else f"UNKNOWN_{int(order.status)}"
    )


def run_intent_tape(npz, intents, shift, qty, local_end):
    """One HBT instance per legacy strategy; all intents replayed in order."""
    actions = []
    skipped_outside = []
    for row in intents:
        # Require both entry and intended stop to lie within the data
        # horizon; otherwise this fixture is censored by dataset end.
        entry = row.entry_wall + shift
        stop = row.stop_wall + shift
        if entry >= local_end or stop >= local_end:
            skipped_outside.append(row)
            continue

        actions.append((entry, 0, row.order_id, "submit", row))
        actions.append((stop, 1, row.order_id, "stop", row))

    # If submit and stop coincide, submit first. No future-state leakage.
    actions.sort()
    hbt = model(npz)
    observations: dict[int, dict] = {}
    active_order_ids: set[int] = set()
    overlap_at_submit = 0

    try:
        rc = int(hbt.wait_next_feed(False, 60 * NS_PER_S))
        if rc != 2:
            raise RuntimeError(f"HBT failed initial feed: rc={rc}")

        for ts, _priority, order_id, kind, row in actions:
            if not step_until(hbt, ts, f"{kind} order {order_id}"):
                raise RuntimeError(
                    f"R1.0 unexpected EOD at {kind} order {order_id}"
                )

            if kind == "submit":
                if active_order_ids:
                    still_live = {
                        old_id for old_id in active_order_ids
                        if int(hbt.orders(0).get(old_id).status) == NEW
                    }
                    active_order_ids = still_live
                n_prior_active = len(active_order_ids)
                if n_prior_active:
                    overlap_at_submit += 1

                depth = hbt.depth(0)
                observed_bid = float(depth.best_bid)
                observed_ask = float(depth.best_ask)
                if row.side == "BUY":
                    rc = int(hbt.submit_buy_order(
                        0, order_id, row.price, qty, GTX, LIMIT, True
                    ))
                else:
                    rc = int(hbt.submit_sell_order(
                        0, order_id, row.price, qty, GTX, LIMIT, True
                    ))
                if rc != 0:
                    raise RuntimeError(f"submit failed id={order_id} rc={rc}")

                order = hbt.orders(0).get(order_id)
                if order is None:
                    raise RuntimeError(f"missing newly submitted {order_id}")

                observations[order_id] = {
                    "intent": row,
                    "at_entry_local_bid": observed_bid,
                    "at_entry_local_ask": observed_ask,
                    "local_price_diff": (
                        abs(row.price - (
                            observed_bid if row.side == "BUY" else observed_ask
                        )) > 1e-9
                    ),
                    "ack_status": hbt_state(order),
                    "ack_exchange_ns": int(order.exch_timestamp),
                    "prior_active_hbt_orders": n_prior_active,
                    "stop_submitted": False,
                }
                if int(order.status) == NEW:
                    active_order_ids.add(order_id)
            elif kind == "stop":
                observed = observations.get(order_id)
                if observed is None:
                    raise RuntimeError(f"stop without submit {order_id}")
                order = hbt.orders(0).get(order_id)
                if order is None:
                    raise RuntimeError(f"missing stop target {order_id}")
                observed["status_before_stop"] = hbt_state(order)
                if int(order.status) == NEW:
                    rc = int(hbt.cancel(0, order_id, True))
                    if rc != 0:
                        raise RuntimeError(
                            f"cancel request failed id={order_id} rc={rc}"
                        )
                    observed["stop_submitted"] = True
                    order = hbt.orders(0).get(order_id)
                    if order is None:
                        raise RuntimeError(f"missing order after cancel {order_id}")
                active_order_ids.discard(order_id)

        for order_id, obs in observations.items():
            order = hbt.orders(0).get(order_id)
            if order is None:
                raise RuntimeError(f"missing terminal order {order_id}")
            obs["hbt_status"] = hbt_state(order)
            obs["hbt_exchange_ns"] = int(order.exch_timestamp)
            obs["hbt_exec_price"] = (
                float(order.exec_price) if int(order.status) == FILLED
                else None
            )
            obs["hbt_exec_qty"] = (
                float(order.exec_qty) if int(order.status) == FILLED
                else None
            )
    finally:
        hbt.close()

    return observations, skipped_outside, overlap_at_submit


def summarize(strategy, intents, observed, skipped, overlapping, shift):
    count = Counter()
    diffs = []
    report_rows = []
    paired_fill_latency = []
    for row in intents:
        if row.order_id not in observed:
            count["censored_due_to_end_of_data"] += 1
            continue

        obs = observed[row.order_id]
        legacy_fill = row.old_status == "filled"
        hbt_fill = obs["hbt_status"] == "FILLED"
        count["orders_compared"] += 1
        count["old_" + row.old_status] += 1
        count["hbt_" + obs["hbt_status"]] += 1
        count["side_" + row.side] += 1
        count["legacy_filled"] += int(legacy_fill)
        count["hbt_filled"] += int(hbt_fill)
        count["both_filled"] += int(legacy_fill and hbt_fill)
        count["only_legacy"] += int(legacy_fill and not hbt_fill)
        count["only_hbt"] += int(hbt_fill and not legacy_fill)
        count["neither"] += int(not legacy_fill and not hbt_fill)
        count["gtx_rejected"] += int(obs["ack_status"] == "EXPIRED")
        count["legacy_filled_while_cancel_pending"] += int(
            row.old_pending_fill
        )
        count["hbt_prior_active_order"] += int(
            obs["prior_active_hbt_orders"] > 0
        )
        count["hbt_local_price_mismatch"] += int(obs["local_price_diff"])

        old_fill_exch_proxy = None
        hbt_fill_wall_proxy = None
        if hbt_fill:
            hbt_fill_wall_proxy = obs["hbt_exchange_ns"] - shift
        if legacy_fill and hbt_fill:
            paired_fill_latency.append(
                row.old_fill_wall - hbt_fill_wall_proxy
            )

        issues = []
        if legacy_fill != hbt_fill:
            issues.append("FILL_STATUS")
        if obs["ack_status"] == "EXPIRED":
            issues.append("GTX_REJECT")
        if obs["local_price_diff"]:
            issues.append("LOCAL_BBO_DIFF")
        if obs["prior_active_hbt_orders"]:
            issues.append("OFF_POLICY_OVERLAP")
        if legacy_fill and hbt_fill and (
            row.old_fill_wall < hbt_fill_wall_proxy
        ):
            issues.append("LEGACY_FILL_PRECEDES_HBT_EXCH_PROXY")
        if issues:
            diffs.append((row, obs, issues))

        report_rows.append({
            "strategy": row.strategy,
            "order_id": row.order_id,
            "side": row.side,
            "entry_wall_ns": row.entry_wall,
            "expiry_wall_ns": row.expiry_wall,
            "cancel_request_wall_ns": row.cancel_req_wall,
            "cancel_effective_wall_ns": row.cancel_eff_wall,
            "old_queue_qty": row.old_queue_qty,
            "price": row.price,
            "old_status": row.old_status,
            "old_fill_wall_ns": row.old_fill_wall,
            "old_fill_reason": row.old_reason,
            "old_pending_fill": row.old_pending_fill,
            "hbt_ack_status": obs["ack_status"],
            "hbt_status": obs["hbt_status"],
            "hbt_fill_exchange_ns": (
                obs["hbt_exchange_ns"] if hbt_fill else None
            ),
            "hbt_fill_wall_proxy_ns": hbt_fill_wall_proxy,
            "hbt_price": obs["hbt_exec_price"],
            "hbt_qty": obs["hbt_exec_qty"],
            "hbt_book_bid": obs["at_entry_local_bid"],
            "hbt_book_ask": obs["at_entry_local_ask"],
            "hbt_local_price_diff": obs["local_price_diff"],
            "other_hbt_orders_live_at_entry": obs["prior_active_hbt_orders"],
            "stop_submitted": obs["stop_submitted"],
            "issues": "|".join(issues),
        })

    print()
    print(f"===== {strategy.upper()} =====")
    for name in [
        "orders_compared", "censored_due_to_end_of_data",
        "side_BUY", "side_SELL", "legacy_filled", "hbt_filled",
        "both_filled", "only_legacy", "only_hbt", "neither",
        "gtx_rejected", "hbt_local_price_mismatch",
        "hbt_prior_active_order",
        "legacy_filled_while_cancel_pending",
    ]:
        print(f"{name}={count[name]}")
    print(f"counterfactual_overlap_entry_events={overlapping}")
    print(f"paired_old_fill_wall_minus_hbt_fill_exchange_proxy_n={len(paired_fill_latency)}")

    print(f"diagnostic_orders_with_flags={len(diffs)}")
    for row, obs, issues in diffs[:12]:
        print(
            f"  order={row.order_id} side={row.side} "
            f"price={row.price:.4f} entry_wall={row.entry_wall} "
            f"old={row.old_status} hbt={obs['hbt_status']} "
            f"ack={obs['ack_status']} "
            f"old_fill_wall={row.old_fill_wall} "
            f"hbt_exch={obs['hbt_exchange_ns'] if obs['hbt_status']=='FILLED' else None} "
            f"flags={','.join(issues)}"
        )

    return count, report_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", type=Path)
    ap.add_argument("legacy_csv_dir", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--qty", type=float, default=0.01)
    ap.add_argument("--strategies", default=",".join(STRATEGIES))
    args = ap.parse_args()

    meta_path = args.npz.with_name("solusdc_hbt_r0_1_meta.json")
    meta = json.loads(meta_path.read_text())
    shift = int(meta["local_timestamp_shift_ns"])
    with np.load(args.npz) as z:
        feed = z["data"]
    from hftbacktest import LOCAL_EVENT
    local = feed[(feed["ev"] & LOCAL_EVENT) == LOCAL_EVENT]
    local_end = int(local[-1]["local_ts"])

    print("===== HBT-R1.0 FROZEN R8.1 STRATEGY INTENT AUDIT =====")
    print("strategy_logic=original_yue_maker_r8_hysteresis_UNMODIFIED")
    print("intent_tape=original_r8_1_orders_csv")
    print("order_entry_latency_ns=0")
    print("legacy_cancel_effective_wall_ns=used_as_HBT_cancel_time")
    print("ttl=original_legacy_expiry_as_HBT_cancel")
    print("clock=original_recv_wall_ns_plus_global_84125us_shift")
    print("execution=HBT_YueStrictQueue_TradeOnlyExchange_GTX")
    print("policy_feedback=FROZEN_INTENTS_NO_RECOMPUTED_SIGNAL")
    print("portfolio_inventory_not_enforced=True")
    print("FULL_STRATEGY_PARITY=NOT_TESTED")
    print(f"npz={args.npz}")
    print(f"legacy_csv_dir={args.legacy_csv_dir}")
    print(f"clock_shift_ns={shift}")

    out_root = args.output
    out_root.mkdir(parents=True, exist_ok=True)
    summary = {
        "stage": "HBT-R1.0",
        "clock_shift_ns": shift,
        "full_strategy_parity": False,
        "strategies": {},
    }

    selection = [s.strip() for s in args.strategies.split(",") if s.strip()]
    for strategy in selection:
        path = args.legacy_csv_dir / f"r8_1_orders_{strategy}.csv"
        if not path.exists():
            raise FileNotFoundError(f"legacy R8.1 order CSV missing: {path}")

        intents = load_intents(path, strategy)
        observed, skipped, overlaps = run_intent_tape(
            args.npz, intents, shift, args.qty, local_end
        )
        counts, output_rows = summarize(
            strategy, intents, observed, skipped, overlaps, shift
        )
        out_csv = out_root / f"r1_0_paired_orders_{strategy}.csv"
        if output_rows:
            with out_csv.open("w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(output_rows[0]))
                writer.writeheader()
                writer.writerows(output_rows)
        else:
            out_csv.write_text("no_orders\n")

        summary["strategies"][strategy] = {
            "original_entries": len(intents),
            "orders_compared": counts["orders_compared"],
            "censored_due_to_end_of_data": counts["censored_due_to_end_of_data"],
            "legacy_filled": counts["legacy_filled"],
            "hbt_filled": counts["hbt_filled"],
            "both_filled": counts["both_filled"],
            "only_legacy": counts["only_legacy"],
            "only_hbt": counts["only_hbt"],
            "gtx_rejected": counts["gtx_rejected"],
            "hbt_local_price_mismatch": counts["hbt_local_price_mismatch"],
            "hbt_prior_active_order": counts["hbt_prior_active_order"],
            "paired_orders_csv": str(out_csv),
        }

    outfile = out_root / "r1_0_summary.json"
    outfile.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print()
    print(f"summary={outfile}")
    print("R1_0_INTENT_EXECUTION_STATUS=DIAGNOSTIC_COMPLETE")
    print("FULL_LEGACY_STRATEGY_PARITY=NOT_TESTED")


if __name__ == "__main__":
    main()
