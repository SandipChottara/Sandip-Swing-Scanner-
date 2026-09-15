# Swing Scanner — Reversal + Continuation

A second, separate scanner. One scan per day, every stock classified as
**Reversal** (downtrend turning up) or **Continuation** (uptrend resuming) —
mutually exclusive by definition, so one pass covers both.

## Design principle

**Price structure is the trigger. Indicators only confirm.**
Mandatory conditions gate (fail = not a candidate at all); confirmations score.
Price structure carries the majority of both 100-point models, so nothing can
rank highly on indicators alone.

Everything is **ATR-normalised** — thresholds adapt to each stock's own
volatility rather than applying one percentage to a ₹50 stock and a ₹5,000 one.

## Setup (same pattern as the momentum screener)

1. New **public** GitHub repo, e.g. `swing-scanner`
2. Upload everything, keeping structure:
   ```
   structure.py
   modules.py
   runner.py
   requirements.txt
   my-full-nse-universe.csv     <- your screener.in export
   .github/workflows/scan.yml
   docs/index.html
   docs/manifest.json
   ```
3. **Settings → Pages** → Deploy from branch → `main` → `/docs`
4. **Actions → Run Swing Scanner → Run workflow**
5. Open the Pages URL on your phone → **⋮ → Add to Home screen**

Runs automatically weekdays at 18:50 IST.

## Light fundamental gate — deliberate

Only exclusions: market cap ≥ ₹2,000 Cr, 20-day turnover ≥ ₹5 Cr, pledge ≤ 30%.

**No ROE or growth thresholds.** A stock 30% off its high that is only now
turning usually has weak trailing numbers — that is the nature of the setup.
Screening on fundamentals would remove exactly the reversals we're hunting.
(The cap is also lower than the momentum screener's ₹20,000 Cr, since reversals
are often found in beaten-down mid-caps.)

## Validation

`python test_modules.py` — six synthetic charts with known structure, covering
both modules and the main false-positive risks (weak higher low, too-deep
pullback, extended stock, ranging chop). Run it after any change.

## Known limitation — read this

**Risk/reward carries little weight in the score** (2/100 in continuation, none
in reversal, as specified). A structurally perfect setup can therefore score A+
while being a poor trade — during testing a 90.3-scoring continuation had a
0.08:1 R:R and an 18% stop distance.

Rather than silently changing your weights, poor R:R and wide stops are flagged
in red on each card. **Read those warnings before acting on a high score.**

## Chart patterns

Detected from the same ATR-filtered pivots the structure engine uses, so a
pattern can never be built on noise already rejected elsewhere:

Symmetrical / Ascending / Descending Triangle · Falling & Rising Wedge ·
Bull Flag & Pennant · Rectangle · Double Bottom · Double Top

Patterns are **descriptive, not a trigger**. A flag on a stock with no higher
low is still not a trade. They add context to a setup that already passed the
structural gates, and the defining trendlines are drawn on the chart.

## Charts

Each card renders an inline SVG candlestick chart (last 90 bars) showing
support/resistance zones, the trigger level, the stop, 20/50 DMA, HL/LL/HH/LH
pivot labels, and the pattern trendlines. No external service, no images
committed to the repo.

## Backtest

`python backtest.py`, or the **Run Backtest** workflow (manual only — it is
heavy). Walk-forward over 12 months: each weekly scan sees only bars up to that
date, using the same `scan()` the live runner calls. Exits use each setup's own
structure-based stop and targets, checked against daily High/Low.

Results land in `docs/backtest.json`, split by Reversal vs Continuation, with a
Nifty 50 comparison.

**Read the caveats it prints.** No brokerage/STT/slippage is modelled, and the
universe is today's Nifty 500 — so delisted or dropped stocks are absent, which
flatters the numbers (survivorship bias).

## Not yet built

- Track record / live P&L history for this scanner
- Bearish reversal (uptrend → downtrend) — you scoped this to bullish only
