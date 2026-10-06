import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from hftbacktest import (
    BUY_EVENT,
    SELL_EVENT,
    DEPTH_EVENT,
    DEPTH_CLEAR_EVENT,
    DEPTH_SNAPSHOT_EVENT,
    TRADE_EVENT,
)
from hftbacktest.data import (
    correct_event_order,
    correct_local_timestamp,
    validate_event_order,
)
from hftbacktest.types import event_dtype


NS_PER_MS = 1_000_000


class EventBuffer:
    def __init__(self, initial_capacity=1_000_000):
        self.data = np.empty(initial_capacity, dtype=event_dtype)
        self.n = 0

    def append(
        self,
        ev,
        exch_ts,
        local_ts,
        px,
        qty,
        order_id=0,
        ival=0,
        fval=0.0,
    ):
        if self.n == len(self.data):
            new = np.empty(len(self.data) * 2, dtype=event_dtype)
            new[:self.n] = self.data[:self.n]
            self.data = new

        self.data[self.n] = (
            ev,
            exch_ts,
            local_ts,
            px,
            qty,
            order_id,
            ival,
            fval,
        )
        self.n += 1

    def finish(self):
        return self.data[:self.n].copy()


def iter_jsonl_zst(path: Path):
    proc = subprocess.Popen(
        ["zstd", "-dc", str(path)],
        stdout=subprocess.PIPE,
        text=True,
        bufsize=1024 * 1024,
    )
    assert proc.stdout is not None

    try:
        for line_no, line in enumerate(proc.stdout, 1):
            try:
                yield line_no, json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"invalid JSON at line {line_no}: {exc}"
                ) from exc
    finally:
        proc.stdout.close()

    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"zstd exited with status {rc}")


def latency_stats(name, values):
    values = np.asarray(values, dtype=np.int64)

    if len(values) == 0:
        print(f"{name}: n=0")
        return {}

    q = np.percentile(
        values / 1_000_000.0,
        [0, 1, 5, 50, 95, 99, 100],
    )

    out = {
        "n": int(len(values)),
        "min_ms": float(q[0]),
        "p01_ms": float(q[1]),
        "p05_ms": float(q[2]),
        "p50_ms": float(q[3]),
        "p95_ms": float(q[4]),
        "p99_ms": float(q[5]),
        "max_ms": float(q[6]),
    }

    print(
        f"{name}: n={out['n']} "
        f"min={out['min_ms']:.3f}ms "
        f"p01={out['p01_ms']:.3f}ms "
        f"p50={out['p50_ms']:.3f}ms "
        f"p99={out['p99_ms']:.3f}ms "
        f"max={out['max_ms']:.3f}ms"
    )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture_dir", type=Path)
    ap.add_argument("--symbol", default="SOLUSDC")
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path("research/results/hbt_r0_1"),
    )
    ap.add_argument(
        "--base-latency-ns",
        type=int,
        default=0,
    )
    args = ap.parse_args()

    symbol = args.symbol.upper()
    depth_stream = symbol.lower() + "@depth@100ms"
    trade_stream = symbol.lower() + "@aggTrade"

    raw = args.capture_dir / "raw.jsonl.zst"
    if not raw.exists():
        raise FileNotFoundError(raw)

    out_dir = args.output_dir / args.capture_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    buf = EventBuffer()

    snapshot_id = None
    snapshot_T = None
    snapshot_recv_ns = None
    depth_started = False
    prev_u = None

    counters = {
        "snapshot_rows": 0,
        "depth_messages": 0,
        "depth_rows": 0,
        "aggtrade_messages": 0,
        "aggtrade_rows": 0,
        "stale_depth_messages": 0,
        "pre_snapshot_trades": 0,
    }

    latency_all = []
    latency_depth = []
    latency_trade = []
    latency_snapshot = []

    for line_no, rec in iter_jsonl_zst(raw):
        kind = rec.get("kind")

        if kind == "depth_snapshot" and rec.get("symbol") == symbol:
            if snapshot_id is not None:
                raise RuntimeError(
                    f"{symbol}: multiple snapshots found"
                )

            d = rec["data"]

            snapshot_id = int(d["lastUpdateId"])
            snapshot_T = int(d["T"])
            snapshot_recv_ns = int(rec["recv_wall_ns"])

            exch_ts = snapshot_T * NS_PER_MS
            local_ts = snapshot_recv_ns

            latency = local_ts - exch_ts
            latency_all.append(latency)
            latency_snapshot.append(latency)

            bids = d["bids"]
            asks = d["asks"]

            if bids:
                buf.append(
                    DEPTH_CLEAR_EVENT | BUY_EVENT,
                    exch_ts,
                    local_ts,
                    float(bids[-1][0]),
                    0.0,
                )
                counters["snapshot_rows"] += 1

                for px, qty in bids:
                    buf.append(
                        DEPTH_SNAPSHOT_EVENT | BUY_EVENT,
                        exch_ts,
                        local_ts,
                        float(px),
                        float(qty),
                    )
                    counters["snapshot_rows"] += 1

            if asks:
                buf.append(
                    DEPTH_CLEAR_EVENT | SELL_EVENT,
                    exch_ts,
                    local_ts,
                    float(asks[-1][0]),
                    0.0,
                )
                counters["snapshot_rows"] += 1

                for px, qty in asks:
                    buf.append(
                        DEPTH_SNAPSHOT_EVENT | SELL_EVENT,
                        exch_ts,
                        local_ts,
                        float(px),
                        float(qty),
                    )
                    counters["snapshot_rows"] += 1

            continue

        if kind != "stream":
            continue

        stream = rec.get("stream")

        if stream == depth_stream:
            if snapshot_id is None:
                continue

            d = rec["data"]

            U = int(d["U"])
            u = int(d["u"])
            pu = int(d["pu"])

            if not depth_started:
                if u < snapshot_id:
                    counters["stale_depth_messages"] += 1
                    continue

                if not (U <= snapshot_id <= u):
                    raise RuntimeError(
                        f"{symbol}: first applicable depth update "
                        f"does not bridge snapshot at line {line_no}: "
                        f"snapshot={snapshot_id} U={U} u={u} pu={pu}"
                    )

                depth_started = True
            else:
                if pu != prev_u:
                    raise RuntimeError(
                        f"{symbol}: sequence gap at line {line_no}: "
                        f"prev_u={prev_u} pu={pu} U={U} u={u}"
                    )

            prev_u = u
            counters["depth_messages"] += 1

            exch_ts = int(d["T"]) * NS_PER_MS
            local_ts = int(rec["recv_wall_ns"])

            latency = local_ts - exch_ts
            latency_all.append(latency)
            latency_depth.append(latency)

            for px, qty in d["b"]:
                buf.append(
                    DEPTH_EVENT | BUY_EVENT,
                    exch_ts,
                    local_ts,
                    float(px),
                    float(qty),
                )
                counters["depth_rows"] += 1

            for px, qty in d["a"]:
                buf.append(
                    DEPTH_EVENT | SELL_EVENT,
                    exch_ts,
                    local_ts,
                    float(px),
                    float(qty),
                )
                counters["depth_rows"] += 1

            continue

        if stream == trade_stream:
            if snapshot_id is None:
                counters["pre_snapshot_trades"] += 1
                continue

            d = rec["data"]

            trade_T = int(d["T"])

            if trade_T < snapshot_T:
                counters["pre_snapshot_trades"] += 1
                continue

            exch_ts = trade_T * NS_PER_MS
            local_ts = int(rec["recv_wall_ns"])

            latency = local_ts - exch_ts
            latency_all.append(latency)
            latency_trade.append(latency)

            ev = (
                TRADE_EVENT | SELL_EVENT
                if bool(d["m"])
                else TRADE_EVENT | BUY_EVENT
            )

            buf.append(
                ev,
                exch_ts,
                local_ts,
                float(d["p"]),
                float(d["q"]),
            )

            counters["aggtrade_messages"] += 1
            counters["aggtrade_rows"] += 1

    if snapshot_id is None:
        raise RuntimeError(f"{symbol}: snapshot not found")

    if not depth_started:
        raise RuntimeError(f"{symbol}: depth stream never bridged snapshot")

    raw_data = buf.finish()

    print("===== HBT-R0.1 CAPTURE ADAPTER =====")
    print(f"symbol={symbol}")
    print(f"snapshot_id={snapshot_id}")
    print(f"snapshot_T={snapshot_T}")
    print(f"snapshot_recv_ns={snapshot_recv_ns}")
    print(f"raw_event_rows={len(raw_data)}")

    print()
    print("===== COUNTS =====")
    for k, v in counters.items():
        print(f"{k}={v}")

    print()
    print("===== RAW FEED LATENCY =====")
    stats = {
        "all": latency_stats("all", latency_all),
        "snapshot": latency_stats("snapshot", latency_snapshot),
        "depth": latency_stats("depth", latency_depth),
        "aggtrade": latency_stats("aggtrade", latency_trade),
    }

    raw_min_latency = int(
        np.min(raw_data["local_ts"] - raw_data["exch_ts"])
    )
    correction_ns = max(
        0,
        -raw_min_latency + args.base_latency_ns,
    )

    print()
    print("===== CLOCK CORRECTION =====")
    print(f"raw_min_latency_ns={raw_min_latency}")
    print(f"base_latency_ns={args.base_latency_ns}")
    print(f"expected_local_shift_ns={correction_ns}")

    corrected = raw_data.copy()
    corrected = correct_local_timestamp(
        corrected,
        args.base_latency_ns,
    )

    print()
    print("===== EVENT ORDER =====")
    ordered = correct_event_order(
        corrected,
        np.argsort(corrected["exch_ts"], kind="mergesort"),
        np.argsort(corrected["local_ts"], kind="mergesort"),
    )

    validate_event_order(ordered)

    output_npz = out_dir / f"{symbol.lower()}_hbt_r0_1.npz"
    np.savez_compressed(output_npz, data=ordered)

    meta = {
        "stage": "HBT-R0.1B",
        "symbol": symbol,
        "capture_dir": str(args.capture_dir),
        "snapshot_id": snapshot_id,
        "snapshot_T_ms": snapshot_T,
        "snapshot_recv_ns": snapshot_recv_ns,
        "raw_event_rows": int(len(raw_data)),
        "ordered_event_rows": int(len(ordered)),
        "raw_min_latency_ns": raw_min_latency,
        "base_latency_ns": args.base_latency_ns,
        "local_timestamp_shift_ns": correction_ns,
        "counters": counters,
        "latency_ms": stats,
        "output_npz": str(output_npz),
    }

    meta_file = out_dir / f"{symbol.lower()}_hbt_r0_1_meta.json"
    meta_file.write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n"
    )

    print()
    print("===== RESULT =====")
    print("event_order=PASS")
    print(f"npz={output_npz}")
    print(f"meta={meta_file}")
    print(f"ordered_event_rows={len(ordered)}")


if __name__ == "__main__":
    main()
