"""
SWING SCANNER — cloud runner. One pass, two classifications.

Light fundamental gate only (market cap, liquidity, pledge). Deliberately no
ROE/growth thresholds: a stock 30% off its high that is only now turning will
usually have weak trailing numbers -- screening those out would remove exactly
the reversals we're hunting.
"""
import json, os, sys, io, re, traceback
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf
import requests

from modules import scan
from structure import CFG

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "docs", "data.json")
FUND = os.path.join(HERE, "my-full-nse-universe.csv")
N500 = "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv"
HDRS = {"User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                       "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"),
        "Accept-Language": "en-IN,en;q=0.9", "Referer": "https://www.nseindia.com/"}

GATE = {"min_mcap_cr": 2000.0, "min_turnover_cr": 5.0, "max_pledge": 30.0}
TOP_N = 20


def universe():
    s = requests.Session(); s.headers.update(HDRS)
    try: s.get("https://www.nseindia.com/", timeout=15)
    except Exception: pass
    for a in range(3):
        try:
            print(f"  Nifty 500, attempt {a+1}/3 ... ", end="", flush=True)
            r = s.get(N500, timeout=20)
            print(f"HTTP {r.status_code}")
            if r.status_code == 200 and len(r.content) > 1000:
                df = pd.read_csv(io.StringIO(r.content.decode("utf-8", "ignore")))
                df.columns = [c.strip() for c in df.columns]
                col = next((c for c in df.columns if c.lower() == "symbol"), None)
                if col:
                    return df[col].dropna().astype(str).str.strip().tolist()
        except Exception as e:
            print(f"failed ({e})")
    return None


def fundamentals():
    if not os.path.exists(FUND):
        return None
    raw = pd.read_csv(FUND, encoding="utf-8-sig")
    raw.columns = [str(c).strip() for c in raw.columns]
    def f(*kw):
        for c in raw.columns:
            lc = re.sub(r"[^a-z0-9 ]", " ", c.lower())
            if all(k in lc for k in kw): return c
        return None
    sym = f("nse", "code") or f("symbol") or f("name")
    if not sym: return None
    mc, pl = f("market", "capitalization") or f("market", "cap"), f("pledge")
    out = pd.DataFrame({"symbol": raw[sym].astype(str).str.strip()})
    out["mcap"] = pd.to_numeric(raw[mc], errors="coerce") if mc else np.nan
    out["pledge"] = pd.to_numeric(raw[pl], errors="coerce") if pl else np.nan
    return out.dropna(subset=["symbol"]).set_index("symbol").to_dict("index")


def main():
    print("=" * 64); print("  SWING SCANNER - Reversal + Continuation"); print("=" * 64)
    notes = []

    print("\n[1/3] Universe...")
    syms = universe()
    if not syms:
        raise RuntimeError("Nifty 500 list unavailable")
    print(f"  {len(syms)} stocks")
    fund = fundamentals()
    if fund is None:
        notes.append("Fundamentals file missing - market-cap and pledge filters skipped "
                     "(liquidity filter still applied).")

    print("\n[2/3] Scanning...")
    rev, con, fails, n = [], [], {}, 0
    for i, sym in enumerate(syms):
        try:
            df = yf.Ticker(f"{sym}.NS").history(period="1y", interval="1d", auto_adjust=True)
            if df is None or len(df) < 130: continue
            df = df.dropna(subset=["Close", "Volume"])
            n += 1
            turn = float(df["Close"].iloc[-1] * df["Volume"].tail(20).mean()) / 1e7
            if turn < GATE["min_turnover_cr"]:
                fails["0_liquidity"] = fails.get("0_liquidity", 0) + 1; continue
            g = fund.get(sym) if fund else None
            if g:
                mc, pg = g.get("mcap"), g.get("pledge")
                if mc is not None and mc == mc and mc < GATE["min_mcap_cr"]:
                    fails["0_mcap"] = fails.get("0_mcap", 0) + 1; continue
                if pg is not None and pg == pg and pg > GATE["max_pledge"]:
                    fails["0_pledge"] = fails.get("0_pledge", 0) + 1; continue

            res = scan(sym, df)
            if res.get("qualified"):
                res["turnover_cr"] = round(turn, 1)
                res["mcap_cr"] = round(g["mcap"]) if g and g.get("mcap") == g.get("mcap") else None
                (rev if res["setup"] == "Reversal" else con).append(res)
            else:
                k = res.get("failed_at", "?"); fails[k] = fails.get(k, 0) + 1
        except Exception:
            continue
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(syms)} - {len(rev)} reversal, {len(con)} continuation")

    rev.sort(key=lambda r: -r["score"]); con.sort(key=lambda r: -r["score"])
    print(f"\n  Scanned {n}: {len(rev)} reversal, {len(con)} continuation")
    print(f"  Rejections: {dict(sorted(fails.items()))}")

    print("\n[3/3] Writing...")
    payload = {
        "generated_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        "scanned": n, "notes": notes,
        "stage_fails": dict(sorted(fails.items())),
        "reversal": rev[:TOP_N], "continuation": con[:TOP_N],
        "counts": {"reversal": len(rev), "continuation": len(con)},
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1, ensure_ascii=False, default=str)
    print(f"  Wrote {OUT}")


if __name__ == "__main__":
    try: main()
    except Exception:
        traceback.print_exc(); sys.exit(1)
