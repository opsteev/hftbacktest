#!/usr/bin/env python3
"""Run the frozen R8.1 50/100/250 ms parity matrix and render one report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from replay_frozen_orders import replay


LATENCIES_MS = (50, 100, 250)


def pct(x: float | None) -> str:
    return "NA" if x is None else f"{100.0 * x:.2f}%"


def num(x: float | int | None, digits: int = 3) -> str:
    if x is None:
        return "NA"
    if isinstance(x, int):
        return str(x)
    return f"{x:.{digits}f}"


def render_report(
    summaries: dict[int, dict[str, Any]],
    *,
    queue_model: str,
    power_n: float,
    source_data: Path,
    old_output_dir: Path,
) -> str:
    title = "# HBT-R0 Engine Parity Report"
    lines = [
        title,
        "",
        "This report audits the frozen Binance SOLUSDC selective-maker execution path. "
        "It does not optimize PnL or select a queue model by profitability.",
        "",
        f"- queue model: `{queue_model}`"
        + (f" (n={power_n:g})" if queue_model == "power_prob" else ""),
        f"- HBT data: `{source_data}`",
        f"- frozen legacy schedules: `{old_output_dir}`",
        "- clock: original integer recv_wall_ns used as both HBT exchange/local time",
        "",
        "## Entry-book parity",
        "",
        "| cancel | price=BBO | queue exact | mean |queue delta| | max |queue delta| |",
        "|---:|---:|---:|---:|---:|",
    ]

    for latency in LATENCIES_MS:
        e = summaries[latency]["entry_book_parity"]
        lines.append(
            "| "
            + " | ".join(
                [
                    f"{latency}ms",
                    pct(e["price_match_fraction"]),
                    pct(e["queue_exact_fraction"]),
                    num(e["queue_delta_mean_abs"]),
                    num(e["queue_delta_max_abs"]),
                ]
            )
            + " |"
        )

    lines += [
        "",
        "## Fill-path parity",
        "",
        "| cancel | entries | old fill | HBT fill | agreement | fill Jaccard | old-only | HBT-only | matched | |Δfill time| mean ms | HBT depth-cross/other |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for latency in LATENCIES_MS:
        s = summaries[latency]
        c = s["fill_path_confusion"]
        dt = s["matched_fill_time_delta_ms"]
        trig = s["hbt_fill_trigger_counts"]
        lines.append(
            "| "
            + " | ".join(
                [
                    f"{latency}ms",
                    str(s["entries"]),
                    pct(s["old_fill_rate"]),
                    pct(s["hbt_fill_rate"]),
                    pct(c["agreement_fraction"]),
                    num(c["fill_jaccard"]),
                    str(c["old_only"]),
                    str(c["hbt_only"]),
                    str(c["both_fill"]),
                    num(dt["mean_abs"]),
                    str(trig["depth_cross_or_book_update"]),
                ]
            )
            + " |"
        )

    lines += [
        "",
        "## Cancel-pending fills",
        "",
        "| cancel | old pending fills | HBT fills inside legacy pending window |",
        "|---:|---:|---:|",
    ]
    for latency in LATENCIES_MS:
        s = summaries[latency]["cancel_pending"]
        lines.append(
            f"| {latency}ms | {s['old_fills_while_pending']} | "
            f"{s['hbt_fills_while_legacy_pending']} |"
        )

    lines += [
        "",
        "## Markout population means (bps)",
        "",
        "| cancel | horizon | old population | HBT population | matched old | matched HBT |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for latency in LATENCIES_MS:
        marks = summaries[latency]["markout_bps"]
        for h in (100, 1000, 5000):
            m = marks[f"{h}ms"]
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"{latency}ms",
                        f"{h}ms",
                        num(m["old_population_mean"]),
                        num(m["hbt_population_mean"]),
                        num(m["matched_both_fill_old_mean"]),
                        num(m["matched_both_fill_hbt_mean"]),
                    ]
                )
                + " |"
            )

    lines += [
        "",
        "## Timing parity",
        "",
        "| cancel | old TTF mean ms | HBT TTF mean ms | old lifetime mean ms | HBT lifetime mean ms | old trade-through/fill | HBT trade-through/fill |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for latency in LATENCIES_MS:
        s = summaries[latency]
        ttf = s["time_to_fill_ms"]
        life = s["order_lifetime_ms"]
        tt = s["trade_through"]
        lines.append(
            "| "
            + " | ".join(
                [
                    f"{latency}ms",
                    num(ttf["old_mean"]),
                    num(ttf["hbt_mean"]),
                    num(life["old_mean"]),
                    num(life["hbt_mean"]),
                    pct(tt["old_rate_among_fills"]),
                    pct(tt["hbt_trade_rate_among_fills"]),
                ]
            )
            + " |"
        )

    lines += [
        "",
        "## Inventory path in one-lot fill units",
        "",
        "| cancel | old terminal | HBT terminal | old max abs | HBT max abs |",
        "|---:|---:|---:|---:|---:|",
    ]
    for latency in LATENCIES_MS:
        inv = summaries[latency]["inventory_path_fill_units"]
        lines.append(
            f"| {latency}ms | {inv['old']['terminal']} | {inv['hbt']['terminal']} | "
            f"{inv['old']['max_abs']} | {inv['hbt']['max_abs']} |"
        )

    mismatch_total = sum(
        summaries[l]["fill_path_confusion"]["old_only"]
        + summaries[l]["fill_path_confusion"]["hbt_only"]
        for l in LATENCIES_MS
    )
    depth_cross_total = sum(
        summaries[l]["hbt_fill_trigger_counts"]["depth_cross_or_book_update"]
        for l in LATENCIES_MS
    )

    lines += [
        "",
        "## Audit interpretation",
        "",
    ]
    if mismatch_total == 0:
        lines.append(
            "The strict trade-only replay has exact fill/non-fill agreement across all three "
            "frozen schedules. Continue to stock HBT queue-model sensitivity."
        )
    else:
        lines.append(
            f"There are {mismatch_total} fill/non-fill disagreements across the three schedules. "
            "Do not interpret economics until the mismatch rows are inspected."
        )
        if depth_cross_total:
            lines.append(
                f"HBT classified {depth_cross_total} fills as depth-cross/book-update rather than "
                "a same-timestamp qualifying aggressive trade. These are the first mismatch class "
                "to inspect because the legacy virtual engine did not use depth crossing as proof "
                "of trade-through."
            )
        lines.append(
            "Use each run's `hbt_r0_orders.csv` to inspect old-only and HBT-only rows, starting "
            "with the earliest timestamp. The purpose is to account for engine semantics, not to "
            "pick the run with the best markout."
        )

    lines += [
        "",
        "## Gate to HBT-R1",
        "",
        "Proceed only after material R0 mismatch classes are explained. R1 then replays the same "
        "frozen schedules under `risk_adverse` and `power_prob` n=1/2/3. A mechanism is "
        "interesting only if its direction survives reasonable queue assumptions; no queue exponent "
        "is selected because it produces the best result.",
        "",
    ]
    return "\n".join(lines)


def run_matrix(
    *,
    input_dir: Path,
    old_output_dir: Path,
    output_root: Path,
    tick_size: float,
    lot_size: float,
    quote_ttl_ms: int,
    queue_model: str,
    power_n: float,
) -> dict[int, dict[str, Any]]:
    summaries: dict[int, dict[str, Any]] = {}
    for latency in LATENCIES_MS:
        old_orders = (
            old_output_dir / f"r8_1_orders_hysteresis_{latency}ms.csv"
        )
        if not old_orders.exists():
            raise FileNotFoundError(old_orders)
        out = output_root / f"hysteresis_{latency}ms_{queue_model}"
        summaries[latency] = replay(
            feed=input_dir / "feed.npz",
            snapshot=input_dir / "snapshot.npz",
            bbo_path=input_dir / "bbo.npz",
            trades_path=input_dir / "trades.npz",
            manifest_path=input_dir / "manifest.json",
            old_orders=old_orders,
            output_dir=out,
            queue_model=queue_model,
            power_n=power_n,
            tick_size=tick_size,
            lot_size=lot_size,
            quote_ttl_ms=quote_ttl_ms,
        )

    output_root.mkdir(parents=True, exist_ok=True)
    matrix_json = {
        str(k): v for k, v in summaries.items()
    }
    (output_root / "hbt_r0_matrix.json").write_text(
        json.dumps(matrix_json, indent=2, sort_keys=True) + "\n"
    )
    report = render_report(
        summaries,
        queue_model=queue_model,
        power_n=power_n,
        source_data=input_dir,
        old_output_dir=old_output_dir,
    )
    (output_root / "HBT_R0_PARITY_REPORT.md").write_text(report + "\n")
    return summaries


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run 50/100/250ms frozen R8.1 HBT parity matrix."
    )
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--old-output-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--tick-size", type=float, required=True)
    p.add_argument("--lot-size", type=float, required=True)
    p.add_argument("--quote-ttl-ms", type=int, default=5000)
    p.add_argument(
        "--queue-model",
        choices=("trade_only", "risk_adverse", "power_prob"),
        default="trade_only",
    )
    p.add_argument("--power-n", type=float, default=1.0)
    return p


def main() -> None:
    args = build_parser().parse_args()
    summaries = run_matrix(
        input_dir=args.input_dir,
        old_output_dir=args.old_output_dir,
        output_root=args.output,
        tick_size=args.tick_size,
        lot_size=args.lot_size,
        quote_ttl_ms=args.quote_ttl_ms,
        queue_model=args.queue_model,
        power_n=args.power_n,
    )
    print(
        json.dumps(
            {
                str(k): {
                    "entries": v["entries"],
                    "old_fill_rate": v["old_fill_rate"],
                    "hbt_fill_rate": v["hbt_fill_rate"],
                    "fill_path_confusion": v["fill_path_confusion"],
                }
                for k, v in summaries.items()
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
