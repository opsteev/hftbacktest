"""HBT-R0.7: raw-capture <-> HBT NPZ timestamp and event provenance.

Independent of the R0.1 converter implementation. Compare every physical
depth/snapshot/trade event from immutable raw.jsonl.zst with BOTH its HBT
local-visible and exchange-visible rows, undoing only the shift saved in
the R0.1 conversion metadata.

This is a data integrity and clock-alignment audit. It does not assert
realistic exchange execution, queue fills, nor original wall-clock strategy
parity. All counters must match exactly; no sampling-based PASS.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

import numpy as np

from hftbacktest import (
    BUY_EVENT,
    SELL_EVENT,
    DEPTH_EVENT,
    DEPTH_CLEAR_EVENT,
    DEPTH_SNAPSHOT_EVENT,
    TRADE_EVENT,
    EXCH_EVENT,
    LOCAL_EVENT,
)


NS_PER_MS = 1_000_000
BASE_MASK = (1 << 28) - 1


def iter_raw(path: Path):
    proc = subprocess.Popen(
        ["zstd", "-dc", str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1024 * 1024,
    )
    assert proc.stdout is not None

    try:
        for lineno, line in enumerate(proc.stdout, 1):
            try:
                yield lineno, json.loads(line)
            except json.JSONDecodeError as e:
                raise RuntimeError(
                    f"corrupt raw JSON at line {lineno}: {e}"
                ) from e
    finally:
        proc.stdout.close()
        err = proc.stderr.read() if proc.stderr else ""
        if proc.stderr:
            proc.stderr.close()
        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(f"zstd failed rc={rc} {err[:800]}")


def make_key(ev, exch_ts, recv_ts, px, qty):
    return (int(ev) & BASE_MASK, int(exch_ts), int(recv_ts),
            float(px), float(qty))


def reconstruct_raw_rows(path: Path, symbol: str):
    depth_stream = symbol.lower() + "@depth@100ms"
    trade_stream = symbol.lower() + "@aggTrade"

    counter = Counter()
    counts = Counter()
    kinds = Counter()

    snapshot_id = None
    snapshot_T = None
    snapshot_recv = None
    depth_started = False
    prev_u = None

    min_latency = None
    max_latency = None
    first_wall = None
    last_wall = None
    last_recv = None
    recv_backtracks = 0
    recv_backtrack_max_ns = 0

    monotonic_checked = 0
    mono_backtracks = 0
    last_mono = None

    raw_input_line_no = 0
    recv_minus_exch = []
    trade_recv_minus_exch = []
    depth_recv_minus_exch = []
    trade_mono_missing = 0

    def emit(ev, exch_ns, recv_ns, px, qty):
        nonlocal min_latency, max_latency
        key = make_key(ev, exch_ns, recv_ns, px, qty)
        counter[key] += 1
        lag = int(recv_ns) - int(exch_ns)
        min_latency = lag if min_latency is None else min(min_latency, lag)
        max_latency = lag if max_latency is None else max(max_latency, lag)

    for lineno, rec in iter_raw(path):
        raw_input_line_no = lineno
        kind = rec.get("kind")
        stream = rec.get("stream", "")

        if kind == "depth_snapshot" and rec.get("symbol") == symbol:
            if snapshot_id is not None:
                raise RuntimeError("more than one target snapshot")
            d = rec["data"]
            snapshot_id = int(d["lastUpdateId"])
            snapshot_T = int(d["T"])
            snapshot_recv = int(rec["recv_wall_ns"])
            exch_ns = snapshot_T * NS_PER_MS

            bids = d.get("bids", [])
            asks = d.get("asks", [])
            if bids:
                emit(DEPTH_CLEAR_EVENT | BUY_EVENT, exch_ns,
                     snapshot_recv, bids[-1][0], 0.0)
                counts["snapshot_rows"] += 1
                for price, qty in bids:
                    emit(DEPTH_SNAPSHOT_EVENT | BUY_EVENT, exch_ns,
                         snapshot_recv, price, qty)
                    counts["snapshot_rows"] += 1
            if asks:
                emit(DEPTH_CLEAR_EVENT | SELL_EVENT, exch_ns,
                     snapshot_recv, asks[-1][0], 0.0)
                counts["snapshot_rows"] += 1
                for price, qty in asks:
                    emit(DEPTH_SNAPSHOT_EVENT | SELL_EVENT, exch_ns,
                         snapshot_recv, price, qty)
                    counts["snapshot_rows"] += 1
            continue

        if kind != "stream" or stream not in (depth_stream, trade_stream):
            continue

        if snapshot_id is None:
            if stream == trade_stream:
                counts["pre_snapshot_trades"] += 1
            else:
                counts["pre_snapshot_depth_skipped"] += 1
            continue

        recv_ns = int(rec["recv_wall_ns"])
        mono_ns = rec.get("recv_mono_ns")

        if stream == depth_stream:
            d = rec["data"]
            U, u, pu = int(d["U"]), int(d["u"]), int(d["pu"])
            if not depth_started:
                if u < snapshot_id:
                    counts["stale_depth_messages"] += 1
                    continue
                if not (U <= snapshot_id <= u):
                    raise RuntimeError(
                        f"snapshot bridge failure line={lineno} "
                        f"snapshot_id={snapshot_id} U={U} u={u}"
                    )
                depth_started = True
            elif pu != prev_u:
                raise RuntimeError(
                    f"sequence break line={lineno} pu={pu} prev={prev_u}"
                )

            prev_u = u
            exch_ns = int(d["T"]) * NS_PER_MS
            counts["depth_messages"] += 1
            depth_recv_minus_exch.append(recv_ns - exch_ns)

            for px, qty in d["b"]:
                emit(DEPTH_EVENT | BUY_EVENT, exch_ns, recv_ns, px, qty)
                counts["depth_rows"] += 1
            for px, qty in d["a"]:
                emit(DEPTH_EVENT | SELL_EVENT, exch_ns, recv_ns, px, qty)
                counts["depth_rows"] += 1
        else:
            d = rec["data"]
            T = int(d["T"])
            if T < snapshot_T:
                counts["pre_snapshot_trades"] += 1
                continue
            exch_ns = T * NS_PER_MS
            ev = (TRADE_EVENT | SELL_EVENT) if bool(d["m"]) else (
                TRADE_EVENT | BUY_EVENT
            )
            emit(ev, exch_ns, recv_ns, d["p"], d["q"])
            counts["aggtrade_messages"] += 1
            counts["aggtrade_rows"] += 1
            trade_recv_minus_exch.append(recv_ns - exch_ns)

        kinds[stream] += 1
        recv_minus_exch.append(recv_ns - exch_ns)

        if first_wall is None:
            first_wall = recv_ns
        last_wall = recv_ns
        if last_recv is not None and recv_ns < last_recv:
            recv_backtracks += 1
            recv_backtrack_max_ns = max(
                recv_backtrack_max_ns, last_recv - recv_ns
            )
        last_recv = recv_ns

        if mono_ns is None:
            trade_mono_missing += 1
        else:
            this_mono = int(mono_ns)
            if last_mono is not None and this_mono < last_mono:
                mono_backtracks += 1
            last_mono = this_mono
            monotonic_checked += 1

    if snapshot_id is None or not depth_started:
        raise RuntimeError("capture lacks snapshot and/or depth sequence bridge")

    return {
        "events": counter,
        "counts": counts,
        "snapshot_id": snapshot_id,
        "snapshot_T_ms": snapshot_T,
        "snapshot_recv_ns": snapshot_recv,
        "min_raw_lag_ns": min_latency,
        "max_raw_lag_ns": max_latency,
        "all_raw_lags": recv_minus_exch,
        "trade_raw_lags": trade_recv_minus_exch,
        "depth_raw_lags": depth_recv_minus_exch,
        "recv_backtracks": recv_backtracks,
        "recv_backtrack_max_ns": recv_backtrack_max_ns,
        "mono_checked": monotonic_checked,
        "mono_backtracks": mono_backtracks,
        "mono_missing": trade_mono_missing,
        "input_lines": raw_input_line_no,
        "streams": kinds,
        "first_wall": first_wall,
        "last_wall": last_wall,
    }


def npz_counters(npz_path: Path, shift_ns: int):
    with np.load(npz_path) as z:
        rows = z["data"]

    mask_l = (rows["ev"] & LOCAL_EVENT) == LOCAL_EVENT
    mask_e = (rows["ev"] & EXCH_EVENT) == EXCH_EVENT

    local = Counter()
    exch = Counter()
    invalid_local_lag = 0
    local_lags = []
    for row in rows[mask_l]:
        ev = int(row["ev"])
        exchange_ts = int(row["exch_ts"])
        local_ts = int(row["local_ts"])
        local[make_key(ev, exchange_ts, local_ts - shift_ns,
                       row["px"], row["qty"])] += 1
        lag = local_ts - exchange_ts
        if lag < 0:
            invalid_local_lag += 1
        local_lags.append(lag)

    for row in rows[mask_e]:
        exch[make_key(
            row["ev"], row["exch_ts"],
            int(row["local_ts"]) - shift_ns,
            row["px"], row["qty"]
        )] += 1

    return {
        "total_rows": len(rows),
        "local_rows": int(mask_l.sum()),
        "exch_rows": int(mask_e.sum()),
        "local": local,
        "exch": exch,
        "negative_corrected_local_lag_rows": invalid_local_lag,
        "corrected_lags": local_lags,
    }


def multiset_diff(raw: Counter, observed: Counter):
    missing = raw - observed
    extra = observed - raw
    return {
        "missing_n": sum(missing.values()),
        "extra_n": sum(extra.values()),
        "missing_examples": list(missing.items())[:3],
        "extra_examples": list(extra.items())[:3],
    }


def percentile_summary(xs):
    if not xs:
        return "n=0"
    xs = np.asarray(xs, dtype=np.float64) / NS_PER_MS
    q = np.percentile(xs, [0, 1, 50, 95, 99, 100])
    return (
        f"n={len(xs)} min={q[0]:.3f}ms "
        f"p01={q[1]:.3f}ms p50={q[2]:.3f}ms "
        f"p95={q[3]:.3f}ms p99={q[4]:.3f}ms "
        f"max={q[5]:.3f}ms"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture_dir", type=Path)
    ap.add_argument("npz", type=Path)
    ap.add_argument("--symbol", default="SOLUSDC")
    ap.add_argument("--meta", type=Path, default=None)
    args = ap.parse_args()

    meta_path = args.meta or args.npz.with_name(
        args.symbol.lower() + "_hbt_r0_1_meta.json"
    )
    if not meta_path.exists():
        raise FileNotFoundError(meta_path)

    metadata = json.loads(meta_path.read_text())
    symbol = args.symbol.upper()
    if metadata["symbol"] != symbol:
        raise RuntimeError("NPZ metadata symbol mismatch")

    shift = int(metadata["local_timestamp_shift_ns"])
    base_latency = int(metadata["base_latency_ns"])

    print("===== HBT-R0.7 TIMESTAMP AND EVENT PROVENANCE =====")
    print(f"symbol={symbol}")
    print(f"raw_capture={args.capture_dir / 'raw.jsonl.zst'}")
    print(f"npz={args.npz}")
    print(f"metadata={meta_path}")
    print(f"local_ts_shift_ns={shift}")
    print(f"base_latency_ns={base_latency}")

    raw = reconstruct_raw_rows(args.capture_dir / "raw.jsonl.zst", symbol)
    npz = npz_counters(args.npz, shift)

    raw_n = sum(raw["events"].values())
    local_diff = multiset_diff(raw["events"], npz["local"])
    exch_diff = multiset_diff(raw["events"], npz["exch"])

    expected_counts = metadata.get("counters", {})
    counts_diff = {
        k: (int(raw["counts"][k]), int(v))
        for k, v in expected_counts.items()
        if int(raw["counts"][k]) != int(v)
    }

    min_lag = raw["min_raw_lag_ns"]
    calc_shift = max(0, -int(min_lag) + base_latency)
    if int(min_lag) >= 0:
        # The actual correct_local_timestamp only shifts when min < 0.
        calc_shift = 0

    print()
    print("===== EVENT IDENTITY =====")
    print(f"raw_physical_events={raw_n}")
    print(f"npz_total_event_rows={npz['total_rows']}")
    print(f"npz_local_event_rows={npz['local_rows']}")
    print(f"npz_exchange_event_rows={npz['exch_rows']}")
    print(f"local_missing={local_diff['missing_n']}")
    print(f"local_extra={local_diff['extra_n']}")
    print(f"exchange_missing={exch_diff['missing_n']}")
    print(f"exchange_extra={exch_diff['extra_n']}")
    print(f"converter_count_mismatches={len(counts_diff)}")
    for side, diff in (("LOCAL", local_diff), ("EXCHANGE", exch_diff)):
        if diff["missing_n"] or diff["extra_n"]:
            print(f"{side}_missing_examples={diff['missing_examples']}")
            print(f"{side}_extra_examples={diff['extra_examples']}")
    if counts_diff:
        print(f"counts_diff={counts_diff}")

    print()
    print("===== CLOCK CORRECTION =====")
    print(f"raw_min_lag_ns={min_lag}")
    print(f"raw_max_lag_ns={raw['max_raw_lag_ns']}")
    print(f"recomputed_shift_ns={calc_shift}")
    print(f"meta_raw_min_lag_ns={metadata['raw_min_latency_ns']}")
    print(f"negative_corrected_feed_latency_rows={npz['negative_corrected_local_lag_rows']}")
    print(f"raw_depth_latency={percentile_summary(raw['depth_raw_lags'])}")
    print(f"raw_trade_latency={percentile_summary(raw['trade_raw_lags'])}")
    print(f"corrected_local_event_latency={percentile_summary(npz['corrected_lags'])}")

    print()
    print("===== RAW RECEIVE ORDER =====")
    print(f"raw_input_lines={raw['input_lines']}")
    print(f"accepted_target_streams={sum(raw['streams'].values())}")
    print(f"recv_wall_ns_backward_steps={raw['recv_backtracks']}")
    print(f"largest_wall_backward_ns={raw['recv_backtrack_max_ns']}")
    print(f"recv_mono_ns_samples={raw['mono_checked']}")
    print(f"recv_mono_ns_missing={raw['mono_missing']}")
    print(f"recv_mono_ns_backward_steps={raw['mono_backtracks']}")
    print(
        "CAUTION=raw receive timestamps and HBT globally shifted local timestamps "
        "are distinct representations; constant shift estimated from the full "
        "capture is ex-post clock calibration, not demonstrated live-calibrated latency."
    )

    integrity_pass = (
        raw_n == npz["local_rows"] == npz["exch_rows"]
        and all(diff["missing_n"] == 0 and diff["extra_n"] == 0
                for diff in (local_diff, exch_diff))
        and not counts_diff
        and raw["snapshot_id"] == int(metadata["snapshot_id"])
        and raw["snapshot_T_ms"] == int(metadata["snapshot_T_ms"])
        and raw["snapshot_recv_ns"] == int(metadata["snapshot_recv_ns"])
        and int(min_lag) == int(metadata["raw_min_latency_ns"])
        and shift == calc_shift
        and npz["negative_corrected_local_lag_rows"] == 0
    )
    print()
    print("R0_7_INTEGRITY_STATUS=" + ("PASS" if integrity_pass else "FAIL"))
    print("FULL_LEGACY_WALL_CLOCK_EXECUTION_PARITY=NOT_TESTED")
    raise SystemExit(0 if integrity_pass else 1)


if __name__ == "__main__":
    main()
