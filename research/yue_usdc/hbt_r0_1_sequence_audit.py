import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SymbolState:
    snapshot_id: int | None = None
    snapshot_recv_ns: int | None = None
    started: bool = False
    prev_u: int | None = None

    depth_events: int = 0
    skipped_before_snapshot: int = 0
    skipped_stale: int = 0
    bridges: int = 0
    gaps: int = 0

    first_U: int | None = None
    first_u: int | None = None
    first_pu: int | None = None


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture_dir", type=Path)
    ap.add_argument(
        "--symbols",
        nargs="+",
        default=["SOLUSDC", "SOLUSDT"],
    )
    args = ap.parse_args()

    symbols = set(args.symbols)
    states = {symbol: SymbolState() for symbol in symbols}

    raw = args.capture_dir / "raw.jsonl.zst"
    if not raw.exists():
        raise FileNotFoundError(raw)

    gap_examples = []

    for line_no, rec in iter_jsonl_zst(raw):
        kind = rec.get("kind")

        if kind == "depth_snapshot":
            symbol = rec.get("symbol")
            if symbol not in symbols:
                continue

            data = rec["data"]
            st = states[symbol]

            if st.snapshot_id is not None:
                raise RuntimeError(
                    f"{symbol}: multiple snapshots encountered; "
                    f"R0 expects one bootstrap snapshot"
                )

            st.snapshot_id = int(data["lastUpdateId"])
            st.snapshot_recv_ns = int(rec["recv_wall_ns"])
            continue

        if kind != "stream":
            continue

        stream = rec.get("stream", "")
        if not stream.endswith("@depth@100ms"):
            continue

        data = rec.get("data", {})
        symbol = data.get("s")

        if symbol not in symbols:
            continue

        st = states[symbol]
        st.depth_events += 1

        U = int(data["U"])
        u = int(data["u"])
        pu = int(data["pu"])

        if st.snapshot_id is None:
            st.skipped_before_snapshot += 1
            continue

        if rec["recv_wall_ns"] < st.snapshot_recv_ns:
            st.skipped_before_snapshot += 1

        if not st.started:
            if u < st.snapshot_id:
                st.skipped_stale += 1
                continue

            if not (U <= st.snapshot_id <= u):
                st.gaps += 1
                gap_examples.append(
                    (
                        symbol,
                        line_no,
                        "snapshot_bridge",
                        st.snapshot_id,
                        U,
                        u,
                        pu,
                    )
                )
                continue

            st.started = True
            st.bridges += 1
            st.prev_u = u
            st.first_U = U
            st.first_u = u
            st.first_pu = pu
            continue

        if pu != st.prev_u:
            st.gaps += 1
            if len(gap_examples) < 20:
                gap_examples.append(
                    (
                        symbol,
                        line_no,
                        "pu_mismatch",
                        st.prev_u,
                        U,
                        u,
                        pu,
                    )
                )

        st.prev_u = u

    print("===== HBT-R0.1 SEQUENCE AUDIT =====")

    failed = False

    for symbol in sorted(symbols):
        st = states[symbol]

        print()
        print(f"[{symbol}]")
        print(f"snapshot_id={st.snapshot_id}")
        print(f"depth_events={st.depth_events}")
        print(f"skipped_before_snapshot={st.skipped_before_snapshot}")
        print(f"skipped_stale={st.skipped_stale}")
        print(f"bridge_count={st.bridges}")
        print(
            "first_bridge="
            f"U={st.first_U} u={st.first_u} pu={st.first_pu}"
        )
        print(f"sequence_gaps={st.gaps}")

        ok = (
            st.snapshot_id is not None
            and st.bridges == 1
            and st.started
            and st.gaps == 0
        )

        print("STATUS=" + ("PASS" if ok else "FAIL"))
        failed |= not ok

    if gap_examples:
        print()
        print("===== FIRST GAP EXAMPLES =====")
        for x in gap_examples[:20]:
            print(x)

    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
