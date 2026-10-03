#!/usr/bin/env python3
"""Replay the exact legacy yue_maker quote schedule through HftBacktest.

This is an execution-engine audit, not a strategy search. Entry side/price/time and legacy cancel
windows come from a frozen r4_orders_*.csv file. HftBacktest is allowed to disagree only on what
fills and when.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from hftbacktest import (
    FILLED,
    GTX,
    LIMIT,
    BacktestAsset,
    HashMapMarketDepthBacktest,
)


MARKOUT_HORIZONS_MS = (100, 1000, 5000)


def exact_optional_int(raw: str | None) -> int | None:
    if raw is None:
        return None
    s = raw.strip()
    if not s or s.lower() in {"none", "null"}:
        return None
    # Never route nanosecond timestamps through float.
    if any(ch in s for ch in ".eE"):
        raise ValueError(f"timestamp is not an exact decimal integer: {raw!r}")
    return int(s)


def optional_float(raw: str | None) -> float | None:
    if raw is None:
        return None
    s = raw.strip()
    if not s or s.lower() in {"none", "null"}:
        return None
    return float(s)


def parse_bool(raw: str | None) -> bool:
    if raw is None:
        return False
    return raw.strip().lower() in {"1", "true", "yes"}


@dataclass
class FrozenQuote:
    order_id: int
    side: str
    entry_ns: int
    price: float
    old_fill_ns: int | None
    old_fill_reason: str | None
    old_cancel_requested_ns: int | None
    old_cancel_effective_ns: int | None
    old_terminal_status: str
    old_terminal_ns: int | None
    old_filled_while_cancel_pending: bool
    old_markouts: dict[int, float | None]
    initial_queue_ahead: float | None
    queue_percentile: float | None

    def exposure_end_ns(self, quote_ttl_ms: int) -> int:
        candidates = [self.entry_ns + quote_ttl_ms * 1_000_000]
        if self.old_cancel_effective_ns is not None:
            candidates.append(self.old_cancel_effective_ns)
        if (
            self.old_terminal_status in {"canceled", "expired", "censored"}
            and self.old_terminal_ns is not None
        ):
            candidates.append(self.old_terminal_ns)
        return min(candidates)


def load_frozen_quotes(path: Path) -> list[FrozenQuote]:
    out: list[FrozenQuote] = []
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            side = str(row["side"]).strip().lower()
            if side not in {"buy", "sell"}:
                raise ValueError(f"unsupported side: {side}")
            old_fill_reason = (row.get("fill_reason") or "").strip() or None
            old_status = (row.get("terminal_status") or "").strip().lower()
            out.append(
                FrozenQuote(
                    order_id=int(row["order_id"]),
                    side=side,
                    entry_ns=exact_optional_int(row["entry_wall_ns"]) or 0,
                    price=float(row["entry_price"]),
                    old_fill_ns=exact_optional_int(row.get("fill_wall_ns")),
                    old_fill_reason=old_fill_reason,
                    old_cancel_requested_ns=exact_optional_int(
                        row.get("cancel_requested_ns")
                    ),
                    old_cancel_effective_ns=exact_optional_int(
                        row.get("cancel_effective_ns")
                    ),
                    old_terminal_status=old_status,
                    old_terminal_ns=exact_optional_int(row.get("terminal_ns")),
                    old_filled_while_cancel_pending=parse_bool(
                        row.get("filled_while_cancel_pending")
                    ),
                    old_markouts={
                        h: optional_float(row.get(f"markout_{h}ms_bps"))
                        for h in MARKOUT_HORIZONS_MS
                    },
                    initial_queue_ahead=optional_float(
                        row.get("initial_queue_ahead")
                    ),
                    queue_percentile=optional_float(row.get("queue_percentile")),
                )
            )
    if not out:
        raise RuntimeError(f"no frozen quotes in {path}")
    ids = [q.order_id for q in out]
    if len(ids) != len(set(ids)):
        raise RuntimeError("old order IDs are not unique in the selected strategy CSV")
    out.sort(key=lambda q: (q.entry_ns, q.order_id))
    return out


class BboSeries:
    def __init__(self, path: Path) -> None:
        z = np.load(path)
        self.ts = np.asarray(z["ts"], dtype=np.int64)
        self.bid = np.asarray(z["bid"], dtype=np.float64)
        self.ask = np.asarray(z["ask"], dtype=np.float64)
        if not (len(self.ts) == len(self.bid) == len(self.ask)):
            raise ValueError("invalid bbo.npz lengths")
        if len(self.ts) and np.any(np.diff(self.ts) < 0):
            raise ValueError("bbo timestamps are not sorted")

    def markout(
        self,
        side: str,
        fill_price: float,
        fill_ns: int,
        horizon_ms: int,
    ) -> float | None:
        target = fill_ns + horizon_ms * 1_000_000
        idx = int(np.searchsorted(self.ts, target, side="left"))
        if idx >= len(self.ts):
            return None
        mid = (float(self.bid[idx]) + float(self.ask[idx])) / 2.0
        if side == "buy":
            return (mid - fill_price) / fill_price * 10_000.0
        return (fill_price - mid) / fill_price * 10_000.0


class TradeSeries:
    def __init__(self, path: Path) -> None:
        z = np.load(path)
        self.ts = np.asarray(z["ts"], dtype=np.int64)
        self.px = np.asarray(z["px"], dtype=np.float64)
        self.buyer_is_maker = np.asarray(z["buyer_is_maker"], dtype=np.bool_)
        if not (len(self.ts) == len(self.px) == len(self.buyer_is_maker)):
            raise ValueError("invalid trades.npz lengths")
        if len(self.ts) and np.any(np.diff(self.ts) < 0):
            raise ValueError("trade timestamps are not sorted")

    def classify_fill(self, side: str, price: float, fill_ns: int) -> str:
        lo = int(np.searchsorted(self.ts, fill_ns, side="left"))
        hi = int(np.searchsorted(self.ts, fill_ns, side="right"))
        same_price = False
        for i in range(lo, hi):
            px = float(self.px[i])
            buyer_is_maker = bool(self.buyer_is_maker[i])
            if side == "buy":
                if not buyer_is_maker:
                    continue
                if px < price and not math.isclose(px, price, rel_tol=1e-12):
                    return "trade_through"
                if math.isclose(px, price, rel_tol=1e-12):
                    same_price = True
            else:
                if buyer_is_maker:
                    continue
                if px > price and not math.isclose(px, price, rel_tol=1e-12):
                    return "trade_through"
                if math.isclose(px, price, rel_tol=1e-12):
                    same_price = True
        if same_price:
            return "same_price_trade"
        return "depth_cross_or_book_update"


def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def build_asset(
    *,
    feed: Path,
    snapshot: Path,
    queue_model: str,
    power_n: float,
    tick_size: float,
    lot_size: float,
) -> BacktestAsset:
    asset = (
        BacktestAsset()
        .data(str(feed))
        .initial_snapshot(str(snapshot))
        .linear_asset(1.0)
        .constant_order_latency(0, 0)
        .tick_size(tick_size)
        .lot_size(lot_size)
        .no_partial_fill_exchange()
        .trading_value_fee_model(0.0, 0.0)
    )
    if queue_model == "trade_only":
        asset = asset.trade_only_queue_model()
    elif queue_model == "risk_adverse":
        asset = asset.risk_adverse_queue_model()
    elif queue_model == "power_prob":
        asset = asset.power_prob_queue_model(power_n)
    else:
        raise ValueError(queue_model)
    return asset


def hbt_status(order: Any | None) -> str:
    if order is None:
        return "missing"
    status = int(order.status)
    # Keep numeric value as a suffix so this report remains useful if status names evolve.
    if status == FILLED:
        return f"filled:{status}"
    return f"status:{status}"


def replay(
    *,
    feed: Path,
    snapshot: Path,
    bbo_path: Path,
    trades_path: Path,
    old_orders: Path,
    manifest_path: Path,
    output_dir: Path,
    queue_model: str,
    power_n: float,
    tick_size: float,
    lot_size: float,
    quote_ttl_ms: int,
) -> dict[str, Any]:
    quotes = load_frozen_quotes(old_orders)
    manifest = json.loads(manifest_path.read_text())
    first_sync_ns = int(manifest["first_sync_recv_ns"])

    first_entry = min(q.entry_ns for q in quotes)
    if first_entry < first_sync_ns:
        raise RuntimeError(
            f"first frozen entry {first_entry} precedes synchronized HBT feed {first_sync_ns}"
        )

    bbo = BboSeries(bbo_path)
    trades = TradeSeries(trades_path)

    asset = build_asset(
        feed=feed,
        snapshot=snapshot,
        queue_model=queue_model,
        power_n=power_n,
        tick_size=tick_size,
        lot_size=lot_size,
    )
    hbt = HashMapMarketDepthBacktest([asset])
    rc = int(hbt.elapse(0))
    if rc == 1:
        raise RuntimeError("HftBacktest feed ended during initialization")

    actions: list[tuple[int, int, str, FrozenQuote]] = []
    for q in quotes:
        actions.append((q.entry_ns, 1, "entry", q))
        actions.append((q.exposure_end_ns(quote_ttl_ms), 0, "cancel", q))
    actions.sort(key=lambda x: (x[0], x[1], x[3].order_id))

    submitted: set[int] = set()
    submit_status: dict[int, str] = {}
    cancel_attempted: set[int] = set()

    def advance_to(target_ns: int) -> None:
        current = int(hbt.current_timestamp)
        if target_ns < current:
            raise RuntimeError(
                f"cannot move HBT backward: current={current}, target={target_ns}"
            )
        if target_ns == current:
            return
        code = int(hbt.elapse(target_ns - current))
        if code == 1 and int(hbt.current_timestamp) < target_ns:
            raise RuntimeError(
                f"HBT feed ended before scheduled action at {target_ns}"
            )
        if code not in {0, 1}:
            raise RuntimeError(f"hbt.elapse returned {code}")

    for action_ns, _priority, kind, q in actions:
        advance_to(action_ns)
        orders = hbt.orders(0)

        if kind == "entry":
            if q.order_id in submitted:
                raise RuntimeError(f"duplicate submit for {q.order_id}")
            if q.side == "buy":
                code = int(
                    hbt.submit_buy_order(
                        0,
                        q.order_id,
                        q.price,
                        lot_size,
                        GTX,
                        LIMIT,
                        True,
                    )
                )
            else:
                code = int(
                    hbt.submit_sell_order(
                        0,
                        q.order_id,
                        q.price,
                        lot_size,
                        GTX,
                        LIMIT,
                        True,
                    )
                )
            if code not in {0, 1}:
                raise RuntimeError(
                    f"submit failed: order={q.order_id} code={code}"
                )
            submitted.add(q.order_id)
            submit_status[q.order_id] = hbt_status(hbt.orders(0).get(q.order_id))
            continue

        if q.order_id not in submitted:
            continue
        order = orders.get(q.order_id)
        if order is not None and bool(order.cancellable):
            code = int(hbt.cancel(0, q.order_id, True))
            if code not in {0, 1}:
                raise RuntimeError(
                    f"cancel failed: order={q.order_id} code={code}"
                )
            cancel_attempted.add(q.order_id)

    rows: list[dict[str, Any]] = []
    hbt_fills: list[tuple[int, str]] = []

    orders = hbt.orders(0)
    for q in quotes:
        order = orders.get(q.order_id)
        filled = order is not None and int(order.status) == FILLED
        fill_ns = int(order.exch_timestamp) if filled else None
        trigger = (
            trades.classify_fill(q.side, q.price, fill_ns)
            if fill_ns is not None
            else None
        )
        if fill_ns is not None:
            hbt_fills.append((fill_ns, q.side))

        hbt_markouts = {
            h: (
                bbo.markout(q.side, q.price, fill_ns, h)
                if fill_ns is not None
                else None
            )
            for h in MARKOUT_HORIZONS_MS
        }
        hbt_cancel_pending = (
            fill_ns is not None
            and q.old_cancel_requested_ns is not None
            and fill_ns >= q.old_cancel_requested_ns
            and (
                q.old_cancel_effective_ns is None
                or fill_ns < q.old_cancel_effective_ns
            )
        )
        row = {
            "order_id": q.order_id,
            "side": q.side,
            "entry_ns": q.entry_ns,
            "price": q.price,
            "initial_queue_ahead": q.initial_queue_ahead,
            "queue_percentile": q.queue_percentile,
            "old_filled": q.old_fill_ns is not None,
            "old_fill_ns": q.old_fill_ns,
            "old_fill_reason": q.old_fill_reason,
            "old_cancel_requested_ns": q.old_cancel_requested_ns,
            "old_cancel_effective_ns": q.old_cancel_effective_ns,
            "old_filled_while_cancel_pending": q.old_filled_while_cancel_pending,
            "old_terminal_status": q.old_terminal_status,
            "hbt_submit_status": submit_status.get(q.order_id),
            "hbt_filled": filled,
            "hbt_fill_ns": fill_ns,
            "hbt_fill_trigger": trigger,
            "hbt_filled_while_legacy_cancel_pending": hbt_cancel_pending,
            "hbt_cancel_attempted": q.order_id in cancel_attempted,
            "hbt_terminal_status": hbt_status(order),
            "fill_time_delta_ms": (
                (fill_ns - q.old_fill_ns) / 1e6
                if fill_ns is not None and q.old_fill_ns is not None
                else None
            ),
        }
        for h in MARKOUT_HORIZONS_MS:
            row[f"old_markout_{h}ms_bps"] = q.old_markouts[h]
            row[f"hbt_markout_{h}ms_bps"] = hbt_markouts[h]
        rows.append(row)

    both_fill = [r for r in rows if r["old_filled"] and r["hbt_filled"]]
    old_only = [r for r in rows if r["old_filled"] and not r["hbt_filled"]]
    hbt_only = [r for r in rows if not r["old_filled"] and r["hbt_filled"]]
    neither = [r for r in rows if not r["old_filled"] and not r["hbt_filled"]]

    fill_deltas = [
        float(r["fill_time_delta_ms"])
        for r in both_fill
        if r["fill_time_delta_ms"] is not None
    ]

    def side_summary(side: str) -> dict[str, Any]:
        group = [r for r in rows if r["side"] == side]
        return {
            "entries": len(group),
            "old_fills": sum(bool(r["old_filled"]) for r in group),
            "hbt_fills": sum(bool(r["hbt_filled"]) for r in group),
            "both_fill": sum(
                bool(r["old_filled"]) and bool(r["hbt_filled"]) for r in group
            ),
            "old_only": sum(
                bool(r["old_filled"]) and not bool(r["hbt_filled"]) for r in group
            ),
            "hbt_only": sum(
                not bool(r["old_filled"]) and bool(r["hbt_filled"]) for r in group
            ),
        }

    inventory = 0
    max_abs_inventory = 0
    for _ts, side in sorted(hbt_fills):
        inventory += 1 if side == "buy" else -1
        max_abs_inventory = max(max_abs_inventory, abs(inventory))

    summary: dict[str, Any] = {
        "research": "HBT-R0_ENGINE_PARITY",
        "queue_model": queue_model,
        "power_n": power_n if queue_model == "power_prob" else None,
        "clock_mode": manifest["clock_mode"],
        "tick_size": tick_size,
        "lot_size": lot_size,
        "quote_ttl_ms": quote_ttl_ms,
        "old_orders": str(old_orders),
        "entries": len(rows),
        "old_fills": sum(bool(r["old_filled"]) for r in rows),
        "hbt_fills": sum(bool(r["hbt_filled"]) for r in rows),
        "old_fill_rate": (
            sum(bool(r["old_filled"]) for r in rows) / len(rows)
            if rows else None
        ),
        "hbt_fill_rate": (
            sum(bool(r["hbt_filled"]) for r in rows) / len(rows)
            if rows else None
        ),
        "fill_path_confusion": {
            "both_fill": len(both_fill),
            "old_only": len(old_only),
            "hbt_only": len(hbt_only),
            "neither": len(neither),
            "agreement_fraction": (
                (len(both_fill) + len(neither)) / len(rows) if rows else None
            ),
            "fill_jaccard": (
                len(both_fill) / (len(both_fill) + len(old_only) + len(hbt_only))
                if (len(both_fill) + len(old_only) + len(hbt_only))
                else None
            ),
        },
        "matched_fill_time_delta_ms": {
            "n": len(fill_deltas),
            "mean": mean(fill_deltas),
            "median": median(fill_deltas),
            "mean_abs": mean([abs(x) for x in fill_deltas]),
            "max_abs": max([abs(x) for x in fill_deltas], default=None),
        },
        "hbt_fill_trigger_counts": {
            name: sum(r["hbt_fill_trigger"] == name for r in rows)
            for name in (
                "same_price_trade",
                "trade_through",
                "depth_cross_or_book_update",
            )
        },
        "cancel_pending": {
            "old_fills_while_pending": sum(
                bool(r["old_filled_while_cancel_pending"]) for r in rows
            ),
            "hbt_fills_while_legacy_pending": sum(
                bool(r["hbt_filled_while_legacy_cancel_pending"]) for r in rows
            ),
        },
        "inventory_path_fill_units": {
            "terminal": inventory,
            "max_abs": max_abs_inventory,
        },
        "by_side": {
            "buy": side_summary("buy"),
            "sell": side_summary("sell"),
        },
        "markout_bps": {},
        "notes": [
            "This replays a frozen legacy quote schedule; it does not search for a new alpha.",
            "HBT order latency is zero in R0. Legacy cancel exposure is reproduced by canceling at the legacy effective-cancel time.",
            "Each HBT quote quantity is one lot. TradeOnlyQueueModel declares a same-price fill when legacy queue-ahead reaches zero.",
            "HBT NoPartialFillExchange may additionally fill from opposite-best depth crossings; those fills are reported as depth_cross_or_book_update when no qualifying same-timestamp trade exists.",
            "Maker and taker fees are set to zero because R0 audits fill paths and markout, not PnL.",
        ],
    }

    for h in MARKOUT_HORIZONS_MS:
        old_vals = [
            float(r[f"old_markout_{h}ms_bps"])
            for r in rows
            if r[f"old_markout_{h}ms_bps"] is not None
        ]
        hbt_vals = [
            float(r[f"hbt_markout_{h}ms_bps"])
            for r in rows
            if r[f"hbt_markout_{h}ms_bps"] is not None
        ]
        summary["markout_bps"][f"{h}ms"] = {
            "old_n": len(old_vals),
            "old_mean": mean(old_vals),
            "hbt_n": len(hbt_vals),
            "hbt_mean": mean(hbt_vals),
            "delta_hbt_minus_old": (
                mean(hbt_vals) - mean(old_vals)
                if hbt_vals and old_vals
                else None
            ),
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "hbt_r0_orders.csv"
    fields = list(rows[0].keys())
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    (output_dir / "hbt_r0_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    hbt.close()
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Replay frozen yue_maker R4 quotes through HftBacktest."
    )
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--old-orders", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--queue-model",
        choices=("trade_only", "risk_adverse", "power_prob"),
        default="trade_only",
    )
    p.add_argument("--power-n", type=float, default=1.0)
    p.add_argument("--tick-size", type=float, required=True)
    p.add_argument("--lot-size", type=float, required=True)
    p.add_argument("--quote-ttl-ms", type=int, default=5000)
    return p


def main() -> None:
    args = build_parser().parse_args()
    d = args.input_dir
    summary = replay(
        feed=d / "feed.npz",
        snapshot=d / "snapshot.npz",
        bbo_path=d / "bbo.npz",
        trades_path=d / "trades.npz",
        manifest_path=d / "manifest.json",
        old_orders=args.old_orders,
        output_dir=args.output,
        queue_model=args.queue_model,
        power_n=args.power_n,
        tick_size=args.tick_size,
        lot_size=args.lot_size,
        quote_ttl_ms=args.quote_ttl_ms,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
