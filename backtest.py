"""
SWING SCANNER — 12-month walk-forward backtest.

METHOD, and its honest limits:

  * Walk forward week by week over the last 12 months. At each date the scanner
    sees ONLY bars up to that date -- the same scan() the live runner calls, on
    a truncated frame. No look-ahead.
  * Entries: any stock scoring >= min_score that week, capped at max_new per
    week (matching how you'd actually trade it).
  * Exits use the setup's OWN structure-based levels -- stop below the higher
    low, targets at the next resistance zones -- not a fixed percentage.
    Checked against daily High/Low, so intraday stop-outs are caught.
  * A position is held up to max_hold_days (default 90, matching the 2-3 month
    swing horizon) and force-closed at the end of the window.

  LIMITS WORTH KNOWING:
  - Universe is today's Nifty 500, so there is survivorship bias: stocks that
    were delisted or dropped from the index over the year are absent, which
    flatters results.
  - No brokerage, STT, slippage or impact cost. On a strategy with this many
    trades those are material.
  - Structure-based stops vary in width; equal-weighting each trade therefore
    implies unequal risk per trade. Per-trade stats are the honest read;
    the compounded figure assumes sequential full-capital deployment.
"""
import json, os, sys, traceback
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

from modules import scan
from runner import universe, fundamentals, GATE

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "docs", "backtest.json")

BT = {
    "months": 12,
    "warmup_bars": 160,     # history the scanner needs before the window starts
    "rebalance_weekday": 0, # Monday
    "min_score": 65,
    "max_new_per_week": 3,
    "max_hold_days": 90,
}


class Pos:
    def __init__(self, sym, setup, date, entry, stop, t1, t2, score):
        self.sym, self.setup, self.entry_date = sym, setup, date
        self.entry, self.stop, self.t1, self.t2, self.score = entry, stop, t1, t2, score
        self.half = False
        self.closed = False
        self.exit_date = self.exit_price = self.reason = None
        self.pnl = None

    def check(self, date, hi, lo, close, max_days):
        if self.closed:
            return
        if (date - self.entry_date).days >= max_days:
            return self.close(date, close, "Max holding period")
        if not self.half:
            if lo <= self.stop:
                return self.close(date, self.stop, "Stop loss")
            if hi >= self.t1:
                self.half = True
                self.stop = self.entry          # to breakeven on the rest
        else:
            if lo <= self.stop:
                return self.close(date, self.stop, "Target 1 hit, rest at breakeven",
                                  blended=(self.t1, self.stop))
            if hi >= self.t2:
                return self.close(date, self.t2, "Target 2", blended=(self.t1, self.t2))

    def close(self, date, price, reason, blended=None):
        self.closed = True
        self.exit_date, self.exit_price, self.reason = date, price, reason
        if blended:
            a, b = blended
            self.pnl = round(((a - self.entry) / self.entry * 100 +
                              (b - self.entry) / self.entry * 100) / 2, 2)
        else:
            self.pnl = round((price - self.entry) / self.entry * 100, 2)


def run():
    print("=" * 66)
    print("  SWING SCANNER — 12-MONTH WALK-FORWARD BACKTEST")
    print("=" * 66)

    syms = universe()
    if not syms:
        raise RuntimeError("Nifty 500 list unavailable")
    fund = fundamentals()
    print(f"\n  Universe: {len(syms)} stocks")

    end = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    start = end - timedelta(days=BT["months"] * 31)
    print(f"  Window: {start.date()} to {end.date()}")
    print("  Downloading price history (once per stock)...\n")

    hist = {}
    for i, s in enumerate(syms):
        try:
            df = yf.Ticker(f"{s}.NS").history(period="2y", interval="1d", auto_adjust=True)
            if df is None or len(df) < BT["warmup_bars"] + 60:
                continue
            df = df.dropna(subset=["Close", "Volume"])
            df.index = pd.to_datetime(df.index).tz_localize(None)
            turn = float(df["Close"].iloc[-1] * df["Volume"].tail(20).mean()) / 1e7
            if turn < GATE["min_turnover_cr"]:
                continue
            g = fund.get(s) if fund else None
            if g and g.get("mcap") == g.get("mcap") and g.get("mcap") is not None \
               and g["mcap"] < GATE["min_mcap_cr"]:
                continue
            hist[s] = df
        except Exception:
            pass
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(syms)} — {len(hist)} usable")
    print(f"\n  {len(hist)} stocks with usable history")
    if not hist:
        raise RuntimeError("no price history downloaded")

    days = sorted({d for df in hist.values() for d in df.index if start <= d <= end})
    rebal = [d for d in days if d.weekday() == BT["rebalance_weekday"]]
    print(f"  {len(days)} trading days, {len(rebal)} weekly scan dates\n")

    open_pos, closed = {}, []
    for di, date in enumerate(days):
        # mark open positions against today's bar
        for sym, p in list(open_pos.items()):
            df = hist.get(sym)
            if df is None or date not in df.index:
                continue
            row = df.loc[date]
            p.check(date, float(row["High"]), float(row["Low"]), float(row["Close"]),
                    BT["max_hold_days"])
            if p.closed:
                closed.append(p)
                del open_pos[sym]

        if date not in rebal:
            continue

        # scan with ONLY data up to this date
        cands = []
        for sym, df in hist.items():
            if sym in open_pos:
                continue
            sub = df[df.index <= date]
            if len(sub) < BT["warmup_bars"]:
                continue
            try:
                r = scan(sym, sub.reset_index(drop=True))
            except Exception:
                continue
            if r.get("qualified") and r.get("confirmed") and r["score"] >= BT["min_score"]:
                cands.append(r)
        cands.sort(key=lambda r: -r["score"])

        for r in cands[:BT["max_new_per_week"]]:
            sym = r["symbol"]
            if sym in open_pos or date not in hist[sym].index:
                continue
            entry = float(hist[sym].loc[date, "Close"])
            if r["stop"] >= entry:
                continue
            open_pos[sym] = Pos(sym, r["setup"], date, entry, r["stop"],
                                r["target1"], r["target2"], r["score"])

        if di % 40 == 0:
            print(f"  {date.date()} — {len(open_pos)} open, {len(closed)} closed")

    last = days[-1]
    for sym, p in open_pos.items():
        df = hist.get(sym)
        if df is not None and last in df.index:
            p.close(last, float(df.loc[last, "Close"]), "Backtest end")
            closed.append(p)

    def stats(trades):
        rs = [t.pnl for t in trades if t.pnl is not None]
        if not rs:
            return {"trades": 0}
        wins = [r for r in rs if r > 0]
        eq = 100.0
        for r in rs:
            eq *= (1 + r / 100)
        reasons = {}
        holds = []
        for t in trades:
            reasons[t.reason] = reasons.get(t.reason, 0) + 1
            if t.exit_date:
                holds.append((t.exit_date - t.entry_date).days)
        return {
            "trades": len(rs),
            "win_rate": round(len(wins) / len(rs) * 100, 1),
            "avg": round(float(np.mean(rs)), 2),
            "avg_win": round(float(np.mean(wins)), 2) if wins else 0,
            "avg_loss": round(float(np.mean([r for r in rs if r <= 0])), 2) if len(wins) < len(rs) else 0,
            "best": round(max(rs), 2), "worst": round(min(rs), 2),
            "compounded": round(eq - 100, 2),
            "avg_hold_days": round(float(np.mean(holds)), 1) if holds else 0,
            "exit_reasons": reasons,
        }

    allst = stats(closed)
    rev = stats([t for t in closed if t.setup == "Reversal"])
    con = stats([t for t in closed if t.setup == "Continuation"])

    # benchmark
    bench = None
    try:
        b = yf.Ticker("^NSEI").history(start=start, end=end + timedelta(days=1), auto_adjust=True)
        if len(b) > 2:
            bench = round((float(b["Close"].iloc[-1]) / float(b["Close"].iloc[0]) - 1) * 100, 2)
    except Exception:
        pass

    print("\n" + "=" * 66)
    print("  RESULTS")
    print("=" * 66)
    for label, st in (("ALL", allst), ("REVERSAL", rev), ("CONTINUATION", con)):
        print(f"\n  {label}")
        if not st.get("trades"):
            print("    no trades")
            continue
        print(f"    Trades {st['trades']} | Win rate {st['win_rate']}% | Avg hold {st['avg_hold_days']}d")
        print(f"    Avg/trade {st['avg']}% | Avg win {st['avg_win']}% | Avg loss {st['avg_loss']}%")
        print(f"    Compounded {st['compounded']}%")
        print(f"    Exits: {st['exit_reasons']}")
    if bench is not None:
        print(f"\n  Nifty 50 buy & hold over the same window: {bench}%")
    print("\n  NOTE: no brokerage/STT/slippage modelled; universe is today's")
    print("  Nifty 500 so survivorship bias flatters these numbers.")
    print("=" * 66 + "\n")

    payload = {
        "generated_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        "window": {"from": str(start.date()), "to": str(end.date())},
        "config": BT, "benchmark_nifty50": bench,
        "all": allst, "reversal": rev, "continuation": con,
        "caveats": [
            "No brokerage, STT or slippage modelled — material at this trade count.",
            "Universe is today's Nifty 500, so delisted/dropped stocks are absent (survivorship bias).",
            "Structure-based stops vary in width, so equal-weighting implies unequal risk per trade.",
        ],
        "trades": [{"symbol": t.sym, "setup": t.setup,
                    "entry_date": str(t.entry_date.date()), "entry": round(t.entry, 2),
                    "exit_date": str(t.exit_date.date()) if t.exit_date else None,
                    "exit": round(t.exit_price, 2) if t.exit_price else None,
                    "reason": t.reason, "pnl_pct": t.pnl, "score": t.score}
                   for t in sorted(closed, key=lambda x: x.entry_date, reverse=True)[:200]],
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1, default=str)
    print(f"  Saved {OUT}\n")


if __name__ == "__main__":
    try:
        run()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
