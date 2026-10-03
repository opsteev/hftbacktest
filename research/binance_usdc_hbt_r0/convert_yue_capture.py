#!/usr/bin/env python3
"""Convert a yue_maker Binance capture into HftBacktest parity-audit inputs.

R0 deliberately uses the yue_maker local receive clock for both HftBacktest exchange and
local timestamps. That removes exchange/local clock-offset correction from the first audit
and asks one narrow question: given the same observable event order, which frozen maker
orders does HftBacktest fill?

The source capture is the zstd JSONL written by yue_maker.collector.
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import orjson
import zstandard as zstd

from hftbacktest.data.validation import correct_event_order, validate_event_order
from hftbacktest.types import (
    BUY_EVENT,
    DEPTH_CLEAR_EVENT,
    DEPTH_EVENT,
    DEPTH_SNAPSHOT_EVENT,
    EXCH_EVENT,
    LOCAL_EVENT,
    SELL_EVENT,
    TRADE_EVENT,
    event_dtype,
)


class EventBuffer:
    def __init__(self, capacity: int = 1_000_000) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        self.data = np.empty(capacity, dtype=event_dtype)
        self.n = 0

    def append(
        self,
        ev: int,
        ts_ns: int,
        px: float,
        qty: float,
    ) -> None:
        if self.n >= len(self.data):
            grown = np.empty(len(self.data) * 2, dtype=event_dtype)
            grown[: self.n] = self.data[: self.n]
            self.data = grown
        self.data[self.n] = (
            ev,
            int(ts_ns),
            int(ts_ns),
            float(px),
            float(qty),
            0,
            0,
            0.0,
        )
        self.n += 1

    def finish(self) -> np.ndarray:
        return self.data[: self.n].copy()


def iter_records(path: Path) -> Iterable[dict[str, Any]]:
    dec = zstd.ZstdDecompressor()
    with path.open("rb") as raw:
        with dec.stream_reader(raw) as reader:
            text = io.TextIOWrapper(reader, encoding="utf-8")
            for line in text:
                if line.strip():
                    yield orjson.loads(line)


def find_snapshot(path: Path, symbol: str) -> tuple[int, dict[str, Any]]:
    for record in iter_records(path):
        if (
            record.get("kind") == "depth_snapshot"
            and str(record.get("symbol", "")).upper() == symbol
        ):
            data = record.get("data") or {}
            if "lastUpdateId" not in data:
                continue
            return int(record["recv_wall_ns"]), data
    raise RuntimeError(f"no depth snapshot found for {symbol}")


def snapshot_array(snapshot_recv_ns: int, snapshot: dict[str, Any]) -> np.ndarray:
    bids = snapshot.get("bids") or []
    asks = snapshot.get("asks") or []
    out = np.empty(len(bids) + len(asks), dtype=event_dtype)
    i = 0
    for px, qty in bids:
        out[i] = (
            DEPTH_SNAPSHOT_EVENT | BUY_EVENT | EXCH_EVENT | LOCAL_EVENT,
            snapshot_recv_ns,
            snapshot_recv_ns,
            float(px),
            float(qty),
            0,
            0,
            0.0,
        )
        i += 1
    for px, qty in asks:
        out[i] = (
            DEPTH_SNAPSHOT_EVENT | SELL_EVENT | EXCH_EVENT | LOCAL_EVENT,
            snapshot_recv_ns,
            snapshot_recv_ns,
            float(px),
            float(qty),
            0,
            0,
            0.0,
        )
        i += 1
    return out


def convert(
    raw_path: Path,
    output_dir: Path,
    symbol: str,
    initial_capacity: int,
) -> dict[str, Any]:
    symbol = symbol.upper()
    snapshot_recv_ns, snapshot = find_snapshot(raw_path, symbol)
    last_update_id = int(snapshot["lastUpdateId"])

    buf = EventBuffer(initial_capacity)
    bbo_ts: list[int] = []
    bbo_bid: list[float] = []
    bbo_ask: list[float] = []
    trade_ts: list[int] = []
    trade_px: list[float] = []
    trade_qty: list[float] = []
    trade_buyer_is_maker: list[bool] = []

    synced = False
    first_sync_recv_ns: int | None = None
    prev_u: int | None = None
    depth_events_seen = 0
    depth_events_used = 0
    agg_trades_seen = 0
    agg_trades_used = 0
    book_tickers_seen = 0
    previous_book_bid: float | None = None
    previous_book_ask: float | None = None

    for record in iter_records(raw_path):
        if record.get("kind") != "stream":
            continue
        data = record.get("data") or {}
        if str(data.get("s", "")).upper() != symbol:
            continue

        recv_ns = int(record["recv_wall_ns"])
        event = data.get("e")
        stream = str(record.get("stream") or "")

        is_book = event == "bookTicker" or stream.endswith("@bookTicker")
        if is_book:
            book_tickers_seen += 1
            try:
                bid = float(data["b"])
                bid_qty = float(data["B"])
                ask = float(data["a"])
                ask_qty = float(data["A"])
            except (KeyError, TypeError, ValueError):
                continue
            if bid <= 0 or ask <= bid:
                continue

            if recv_ns >= snapshot_recv_ns:
                bbo_ts.append(recv_ns)
                bbo_bid.append(bid)
                bbo_ask.append(ask)

            if synced:
                # R4 entries were created from bookTicker BBO, not from depth@100ms. Feed the
                # same BBO into HftBacktest so an order scheduled at this recv_wall_ns sees the
                # same best price and displayed best-level quantity as the legacy engine.
                #
                # When the best moves away, clear through the new best first so stale better
                # levels from the slower depth stream cannot remain as HBT's synthetic BBO.
                if previous_book_bid is not None and bid < previous_book_bid:
                    buf.append(DEPTH_CLEAR_EVENT | BUY_EVENT, recv_ns, bid, 0.0)
                if previous_book_ask is not None and ask > previous_book_ask:
                    buf.append(DEPTH_CLEAR_EVENT | SELL_EVENT, recv_ns, ask, 0.0)
                buf.append(DEPTH_EVENT | BUY_EVENT, recv_ns, bid, bid_qty)
                buf.append(DEPTH_EVENT | SELL_EVENT, recv_ns, ask, ask_qty)
                previous_book_bid = bid
                previous_book_ask = ask
            continue

        if event == "depthUpdate":
            depth_events_seen += 1
            try:
                first_u = int(data["U"])
                last_u = int(data["u"])
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError("depthUpdate missing U/u") from exc

            if not synced:
                if last_u < last_update_id:
                    continue
                if not (first_u <= last_update_id <= last_u):
                    if first_u > last_update_id:
                        raise RuntimeError(
                            "cannot bridge REST snapshot to first depth update: "
                            f"snapshot={last_update_id} U={first_u} u={last_u}"
                        )
                    continue
                synced = True
                first_sync_recv_ns = recv_ns
            else:
                try:
                    pu = int(data["pu"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise RuntimeError("synced depthUpdate missing pu") from exc
                if prev_u is not None and pu != prev_u:
                    raise RuntimeError(
                        f"depth gap after sync: previous_u={prev_u}, pu={pu}, "
                        f"U={first_u}, u={last_u}, recv_ns={recv_ns}"
                    )

            prev_u = last_u
            depth_events_used += 1

            for px, qty in data.get("b") or []:
                buf.append(DEPTH_EVENT | BUY_EVENT, recv_ns, float(px), float(qty))
            for px, qty in data.get("a") or []:
                buf.append(DEPTH_EVENT | SELL_EVENT, recv_ns, float(px), float(qty))
            continue

        if event == "aggTrade":
            agg_trades_seen += 1
            if not synced:
                continue
            try:
                px = float(data["p"])
                qty = float(data["q"])
                buyer_is_maker = bool(data["m"])
            except (KeyError, TypeError, ValueError):
                continue

            # Binance aggTrade m=True means the buyer is maker, hence the aggressor is SELL.
            side_flag = SELL_EVENT if buyer_is_maker else BUY_EVENT
            buf.append(TRADE_EVENT | side_flag, recv_ns, px, qty)
            agg_trades_used += 1
            trade_ts.append(recv_ns)
            trade_px.append(px)
            trade_qty.append(qty)
            trade_buyer_is_maker.append(buyer_is_maker)

    if not synced or first_sync_recv_ns is None:
        raise RuntimeError(f"failed to synchronize depth for {symbol}")

    raw_events = buf.finish()
    if len(raw_events) == 0:
        raise RuntimeError("no HftBacktest feed events produced")

    # local parity clock: exchange and local timestamps intentionally equal recv_wall_ns.
    feed = correct_event_order(
        raw_events,
        np.argsort(raw_events["exch_ts"], kind="mergesort"),
        np.argsort(raw_events["local_ts"], kind="mergesort"),
    )
    validate_event_order(feed)

    snap = snapshot_array(snapshot_recv_ns, snapshot)

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "feed.npz", data=feed)
    np.savez_compressed(output_dir / "snapshot.npz", data=snap)
    np.savez_compressed(
        output_dir / "bbo.npz",
        ts=np.asarray(bbo_ts, dtype=np.int64),
        bid=np.asarray(bbo_bid, dtype=np.float64),
        ask=np.asarray(bbo_ask, dtype=np.float64),
    )
    np.savez_compressed(
        output_dir / "trades.npz",
        ts=np.asarray(trade_ts, dtype=np.int64),
        px=np.asarray(trade_px, dtype=np.float64),
        qty=np.asarray(trade_qty, dtype=np.float64),
        buyer_is_maker=np.asarray(trade_buyer_is_maker, dtype=np.bool_),
    )

    manifest = {
        "research": "HBT-R0_ENGINE_PARITY",
        "symbol": symbol,
        "source_raw": str(raw_path),
        "clock_mode": "local_receive_parity",
        "timestamp_note": (
            "exch_ts and local_ts are both the original integer recv_wall_ns; "
            "no ns timestamp is converted through float"
        ),
        "snapshot_recv_ns": snapshot_recv_ns,
        "snapshot_last_update_id": last_update_id,
        "first_sync_recv_ns": first_sync_recv_ns,
        "last_depth_update_id": prev_u,
        "depth_events_seen": depth_events_seen,
        "depth_events_used": depth_events_used,
        "agg_trades_seen": agg_trades_seen,
        "agg_trades_used": agg_trades_used,
        "book_tickers_seen": book_tickers_seen,
        "hbt_raw_rows_before_event_order": int(len(raw_events)),
        "hbt_rows_after_event_order": int(len(feed)),
        "bbo_rows": len(bbo_ts),
        "trade_rows": len(trade_ts),
        "sync_rule": (
            "drop u < snapshot lastUpdateId; first kept event must satisfy "
            "U <= lastUpdateId <= u; thereafter require pu == previous u"
        ),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Convert yue_maker zstd capture to HftBacktest R0 parity inputs."
    )
    p.add_argument("raw", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--symbol", default="SOLUSDC")
    p.add_argument("--initial-capacity", type=int, default=1_000_000)
    return p


def main() -> None:
    args = build_parser().parse_args()
    manifest = convert(
        raw_path=args.raw,
        output_dir=args.output,
        symbol=args.symbol,
        initial_capacity=args.initial_capacity,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
