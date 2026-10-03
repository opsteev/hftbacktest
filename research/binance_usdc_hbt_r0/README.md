# HBT-R0 — Binance USDC selective-maker engine parity

This directory is intentionally narrow. It does **not** search for a new maker strategy and does
not use Hyperliquid as a venue. It audits the frozen Binance USDC maker execution path from
`opsteev/yue_maker@research/usdc-mm-0.1` inside HftBacktest.

## What R0 isolates

The legacy yue_maker virtual fill model:

- enters behind the full displayed best-level quantity;
- advances queue ahead only with qualifying same-price aggressive trades;
- ignores cancellations ahead;
- declares a fill once queue ahead reaches zero;
- declares trade-through fills from qualifying aggressive trades beyond the resting price;
- keeps an order fillable between cancel request and cancel-effective time.

HftBacktest's stock `RiskAdverseQueueModel` is close but not identical: a displayed depth
decrease can truncate estimated front queue. R0 therefore adds `TradeOnlyQueueModel` as a strict
legacy-parity queue model. Stock `RiskAdverseQueueModel` and `PowerProbQueueModel(n)` remain the
next robustness layer, not tuning choices.

The stock HftBacktest exchange model can also fill when the opposite best crosses a resting order
based on depth updates. The legacy virtual engine only used aggressive trades for its trade-through
proof. R0 preserves that HBT behavior and reports such fills separately instead of hiding the
difference.

## Data adapter

`convert_yue_capture.py` reads the immutable yue_maker `raw.jsonl.zst` capture and produces:

- `snapshot.npz` — synchronized SOLUSDC depth snapshot;
- `feed.npz` — HBT depth/trade feed;
- `bbo.npz` — original SOLUSDC bookTicker series for markouts;
- `trades.npz` — original SOLUSDC aggTrades for fill-trigger diagnostics;
- `manifest.json` — synchronization and row-count audit.

For R0, HBT `exch_ts` and `local_ts` are both the original integer `recv_wall_ns`. This is
deliberate: the first audit isolates queue/fill semantics from feed-latency calibration. No
nanosecond timestamp is routed through Python float.

The adapter also fuses bookTicker best price/quantity into the HBT L2 stream. Legacy entries were
priced from bookTicker; using only the slower `depth@100ms` stream would otherwise create false
GTX rejects and false crossing fills at entry.

## Frozen order schedules

Existing R4 runs already write detailed `r4_orders_*.csv` files.

R8.1 previously wrote only `r8_1_summary.json`. A non-semantic export patch was added to
`yue_maker@research/usdc-mm-0.1`; rerunning R8.1 now also creates:

- `r8_1_orders_hysteresis_50ms.csv`
- `r8_1_orders_hysteresis_100ms.csv`
- `r8_1_orders_hysteresis_250ms.csv`

The R8.1 state machine itself is unchanged.

## Preferred path: reuse the original 20260929T034418Z raw capture

From a parent directory containing both repositories:

```bash
find ./yue_maker -path '*20260929T034418Z/raw.jsonl.zst' -print
```

If that returns the old capture:

```bash
cd yue_maker
git checkout research/usdc-mm-0.1
git pull

RAW=/absolute/path/to/20260929T034418Z/raw.jsonl.zst
OLD_OUT=research/results/hbt_r0_old_r8_1

python -m yue_maker.r8_hysteresis "$RAW" \
  --output "$OLD_OUT" \
  --target SOLUSDC \
  --leader SOLUSDT \
  --spot-fx USDCUSDT \
  --fx-usdc BTCUSDC \
  --fx-usdt BTCUSDT
```

Then:

```bash
cd ../hftbacktest
git checkout research/binance-usdc-hbt-r0
git pull

python -m pip install -e py-hftbacktest
python -m pip install zstandard orjson

RAW=/absolute/path/to/20260929T034418Z/raw.jsonl.zst
HBT_DATA=research/results/binance_usdc_hbt_r0/data

python research/binance_usdc_hbt_r0/convert_yue_capture.py "$RAW" \
  --output "$HBT_DATA" \
  --symbol SOLUSDC
```

As of 2026-10-03, Binance USDⓈ-M exchange information reports SOLUSDC price tick `0.01` and
quantity step `0.01`. Use the captured/historical contract filters if they differ.

Run the complete strict R0 matrix in one command:

```bash
python research/binance_usdc_hbt_r0/run_r0_matrix.py \
  --input-dir "$HBT_DATA" \
  --old-output-dir ../yue_maker/research/results/hbt_r0_old_r8_1 \
  --output research/results/binance_usdc_hbt_r0/trade_only \
  --queue-model trade_only \
  --tick-size 0.01 \
  --lot-size 0.01
```

This runs the frozen 50/100/250 ms hysteresis schedules and writes
`HBT_R0_PARITY_REPORT.md` plus the detailed per-order CSV/JSON outputs.

For a single schedule, use `replay_frozen_orders.py` directly. Only after the strict R0 mismatch
classes are understood should the same frozen schedules be replayed with stock HBT queue models:

```bash
# HBT-R1 sensitivity, not parameter fitting
--queue-model risk_adverse
--queue-model power_prob --power-n 1
--queue-model power_prob --power-n 2
--queue-model power_prob --power-n 3
```

## If the original raw capture is gone

Collect one new immutable sample in yue_maker, then replay **that exact same file** through the old
engine and HBT:

```bash
cd ../yue_maker
python -m yue_maker.collector \
  --output data/binance_usdm \
  --hours 1

# Use the session path printed by collector.
RAW=/absolute/path/to/new/session/raw.jsonl.zst

python -m yue_maker.r8_hysteresis "$RAW" \
  --output research/results/hbt_r0_old_r8_1
```

Then run the converter and HBT replay above against the same `RAW`.

## R0 outputs and decision discipline

`replay_frozen_orders.py` writes `hbt_r0_orders.csv` and `hbt_r0_summary.json`. The first report
is deliberately about execution, not Sharpe or optimized PnL.

Primary fields:

- old/HBT fill rate;
- both-fill / old-only / HBT-only / neither;
- fill Jaccard and overall fill-path agreement;
- matched fill-time delta;
- same-price trade / trade-through / depth-cross-or-book-update fill trigger;
- old/HBT cancel-pending fills;
- 100 ms / 1 s / 5 s markout, both whole-population and matched-fill views;
- cumulative buy/sell fill-unit inventory path.

Do not choose a queue model because it produces the best economics. First account for the mismatch
classes. If the strict trade-only run disagrees materially, inspect why before proceeding to R1.
Only after R0/R1 show that queue20, stale cancellation, and resilience improvements survive
reasonable fill assumptions should the project proceed to one-sided inventory cycling.
