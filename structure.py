"""
SWING SCANNER — shared structure engine.

Everything both modules need: ATR-filtered swing pivots, LH/LL/HH/HL
labelling, and ATR-clustered support/resistance zones with strength scores.

Design principles (per spec):
  * PRICE STRUCTURE IS THE TRIGGER. Indicators only confirm.
  * Zones, never exact prices. Grouped by ATR, not a fixed %.
  * ATR normalises everything so the scanner adapts to each stock's
    volatility instead of applying one threshold to a Rs 50 stock and a
    Rs 5000 stock alike.
"""

import numpy as np
import pandas as pd

CFG = {
    "lookback_days":        90,    # primary 60-90 day daily-chart context
    "pivot_bars":            2,    # 5-candle pivot = 2 left / 2 right
    "atr_period":           14,
    "pivot_min_atr":       1.5,    # a swing must move >=1.5 ATR to count (noise filter)
    "zone_atr":            1.2,    # pivots within 1.2 ATR cluster into one zone
    "near_support_pct":    3.0,    # "at support" = within 3%
    "hl_min_atr":          0.5,    # HL must sit >=0.5 ATR above the prior LL
    "hl_bounce_atr":       1.0,    # HL must bounce >=1 ATR to prove buyers stepped in
    "vol_strong":          1.5,    # breakout volume vs 20d avg
    "vol_good":            1.2,
    "max_ext_above_20dma": 10.0,   # reject continuation buys this far above 20 DMA
    "min_rr":              1.5,
    "max_ext_above_trigger": 12.0,  # reject reversal entries this far past the breakout    # minimum acceptable risk:reward
}


# ── indicators ───────────────────────────────────────────────────────────

def atr(df, n=14):
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, min_periods=n, adjust=False).mean()


def rsi(s, n=14):
    d = s.diff()
    g = d.clip(lower=0).ewm(alpha=1/n, min_periods=n, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(alpha=1/n, min_periods=n, adjust=False).mean()
    return (100 - 100 / (1 + g / l.replace(0, np.nan))).fillna(50)


def sma(s, n):
    return s.rolling(n).mean()


# ── 1. swing pivots, ATR-filtered ────────────────────────────────────────

def find_pivots(df, cfg=CFG):
    """
    5-candle pivots (2 left / 2 right), then filtered: a pivot only counts if
    the move away from it is >= pivot_min_atr. This is what stops the scanner
    drowning in dozens of tiny highs and lows on a noisy chart.

    Returns (lows, highs) as lists of dicts: {i, price, atr}
    """
    k = cfg["pivot_bars"]
    a = atr(df, cfg["atr_period"])
    H, L = df["High"].values, df["Low"].values
    n = len(df)
    raw_lo, raw_hi = [], []

    for i in range(k, n - k):
        av = a.iloc[i]
        if not av or av != av:
            continue
        if L[i] < L[i-k:i].min() and L[i] < L[i+1:i+k+1].min():
            raw_lo.append({"i": i, "price": float(L[i]), "atr": float(av)})
        if H[i] > H[i-k:i].max() and H[i] > H[i+1:i+k+1].max():
            raw_hi.append({"i": i, "price": float(H[i]), "atr": float(av)})

    def significant(pivots, is_low):
        """Keep a pivot only if price moved >=1.5 ATR away from it afterwards."""
        out = []
        for p in pivots:
            fwd = df.iloc[p["i"]: min(n, p["i"] + 25)]
            if len(fwd) < 3:
                continue
            move = (fwd["High"].max() - p["price"]) if is_low else (p["price"] - fwd["Low"].min())
            if move >= cfg["pivot_min_atr"] * p["atr"]:
                out.append(p)
        return out

    return significant(raw_lo, True), significant(raw_hi, False)


# ── 2. LH / LL / HH / HL labelling ───────────────────────────────────────

def label_pivots(lows, highs):
    """
    Tag each pivot LL/HL and LH/HH.

    Compares against the last pivot that DIFFERED MEANINGFULLY (>0.3 ATR), not
    simply the one immediately before. On choppy charts two near-identical lows
    would otherwise get labelled HL/LL essentially at random, corrupting the
    trend state that everything downstream depends on.
    """
    def tag(pivots, up_label, down_label):
        ref = None
        for p in pivots:
            if ref is None:
                p["label"] = "—"
                ref = p
                continue
            diff = p["price"] - ref["price"]
            if abs(diff) < 0.3 * p["atr"]:
                p["label"] = "="            # too close to call -- equal swing
            else:
                p["label"] = up_label if diff > 0 else down_label
                ref = p
        return pivots
    return tag(lows, "HL", "LL"), tag(highs, "HH", "LH")


def trend_state(lows, highs):
    """
    'down'  = at least 2 LH/LL pairs recently
    'up'    = at least 2 HH/HL pairs recently
    'range' = neither
    """
    ll = sum(1 for p in lows[-4:] if p.get("label") == "LL")
    lh = sum(1 for p in highs[-4:] if p.get("label") == "LH")
    hh = sum(1 for p in highs[-4:] if p.get("label") == "HH")
    hl = sum(1 for p in lows[-4:] if p.get("label") == "HL")
    if ll >= 1 and lh >= 2 or (ll >= 2 and lh >= 1):
        return "down"
    if hh >= 2 and hl >= 1 or (hh >= 1 and hl >= 2):
        return "up"
    return "range"


# ── 3. S/R zones, ATR-clustered, 9-point strength score ──────────────────

def build_zones(pivots, df, kind, cfg=CFG):
    """
    Cluster pivots within zone_atr of each other into zones, then score each
    0-9 per the spec:
        reactions 1/2/3+        -> +1/+2/+3
        strong bounce (or rejection) -> +1
        above-average volume at the reaction -> +1
        role reversal (old resistance->support, or vice versa) -> +2
        aligns with 50 DMA      -> +1
        aligns with 200 DMA     -> +1
    """
    if not pivots:
        return []
    a_med = float(atr(df, cfg["atr_period"]).median())
    band = cfg["zone_atr"] * a_med

    zones = []
    for p in sorted(pivots, key=lambda x: x["price"]):
        for z in zones:
            if abs(p["price"] - z["price"]) <= band:
                z["pivots"].append(p)
                z["price"] = float(np.mean([q["price"] for q in z["pivots"]]))
                break
        else:
            zones.append({"price": float(p["price"]), "pivots": [p]})

    vol = df["Volume"]; avg_vol = float(vol.mean())
    d50 = sma(df["Close"], 50); d200 = sma(df["Close"], min(200, max(20, len(df)//2)))
    close = df["Close"].values
    n = len(df)
    out = []

    for z in zones:
        touches = len(z["pivots"])
        score = min(3, touches)                      # 1/2/3+ reactions

        # quality of each reaction -- measured, because repeated touches with
        # ever-weaker bounces actually mean the level is WEAKENING, not holding
        bounces = []
        for p in z["pivots"]:
            fwd = df.iloc[p["i"]: min(n, p["i"] + 20)]
            if len(fwd) < 2:
                continue
            if kind == "support":
                bounces.append((fwd["High"].max() - p["price"]) / p["price"] * 100)
            else:
                bounces.append((p["price"] - fwd["Low"].min()) / p["price"] * 100)
        best_bounce = max(bounces) if bounces else 0.0
        if best_bounce >= 5:
            score += 1

        # volume at the reactions
        rvols = []
        for p in z["pivots"]:
            w = vol.iloc[max(0, p["i"]-1): p["i"]+3]
            if len(w):
                rvols.append(float(w.mean()) / avg_vol if avg_vol else 0)
        if rvols and max(rvols) >= 1.2:
            score += 1

        # Role reversal, properly: the level must have (a) REJECTED price from
        # the other side at least twice, (b) been BROKEN through, and only then
        # (c) been retested from this side. A loose "price was mostly below"
        # proxy produced both false positives and misses.
        first_i = min(p["i"] for p in z["pivots"])
        lvl = z["price"]
        role_rev = False
        if first_i > 15:
            hist = close[:first_i]
            if kind == "support":
                # was resistance: repeatedly capped BELOW it, then closed above
                caps = int(((hist > lvl * 0.985) & (hist < lvl * 1.005)).sum())
                broke = bool((hist > lvl * 1.02).any())
                below_before = float((hist < lvl).mean()) > 0.5
                role_rev = caps >= 2 and broke and below_before
            else:
                # was support: repeatedly held ABOVE it, then closed below
                floors = int(((hist < lvl * 1.015) & (hist > lvl * 0.995)).sum())
                broke = bool((hist < lvl * 0.98).any())
                above_before = float((hist > lvl).mean()) > 0.5
                role_rev = floors >= 2 and broke and above_before
        if role_rev:
            score += 2

        # MA alignment
        if d50.notna().iloc[-1] and abs(z["price"] - float(d50.iloc[-1])) / z["price"] < 0.03:
            score += 1
        if d200.notna().iloc[-1] and abs(z["price"] - float(d200.iloc[-1])) / z["price"] < 0.03:
            score += 1

        score = min(9, score)
        band_pct = band / z["price"] * 100
        out.append({
            "price": round(z["price"], 2),
            "low": round(z["price"] - band, 2),
            "high": round(z["price"] + band, 2),
            "touches": touches,
            "score": score,
            "grade": ("Strong" if score >= 7 else "Good" if score >= 5 else
                      "Weak" if score >= 3 else "Ignore"),
            "best_bounce_pct": round(best_bounce, 1),
            "bounces": [round(b, 1) for b in bounces],
            "weakening": bool(len(bounces) >= 2 and bounces[-1] < bounces[0] * 0.5),
            "last_i": max(p["i"] for p in z["pivots"]),
            "first_i": min(p["i"] for p in z["pivots"]),
            "band_pct": round(band_pct, 2),
        })
    return sorted(out, key=lambda z: -z["score"])


def zone_near(zones, price, pct):
    c = [z for z in zones if abs(price - z["price"]) / z["price"] * 100 <= pct]
    return max(c, key=lambda z: z["score"]) if c else None


def zone_below(zones, price):
    c = [z for z in zones if z["price"] < price]
    return max(c, key=lambda z: z["price"]) if c else None


def zone_above(zones, price):
    c = [z for z in zones if z["price"] > price]
    return min(c, key=lambda z: z["price"]) if c else None
