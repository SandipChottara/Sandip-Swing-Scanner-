"""
CHART PATTERN DETECTION — triangles, flags, wedges, double bottoms, rectangles.

Patterns are detected from the SAME ATR-filtered pivots the rest of the scanner
uses, so a pattern can never be built on noise that the structure engine already
rejected.

Each pattern returns the trendlines that define it, so the app can plot them.

IMPORTANT: patterns are DESCRIPTIVE, not a separate trigger. They add context to
a setup that already passed the structural gates — a flag on a stock with no
higher low is still not a trade.
"""
import numpy as np


def _fit(points):
    """Least-squares line through [(i, price)]. Returns (slope, intercept, r2)."""
    if len(points) < 2:
        return None
    x = np.array([p[0] for p in points], float)
    y = np.array([p[1] for p in points], float)
    if len(np.unique(x)) < 2:
        return None
    slope, inter = np.polyfit(x, y, 1)
    pred = slope * x + inter
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(slope), float(inter), float(r2)


def _norm_slope(slope, price, atr_v):
    """Slope per bar, expressed in ATR — comparable across stocks."""
    return (slope / atr_v) if atr_v else 0.0


def detect_patterns(df, lows, highs, atr_v, lookback=60):
    """
    Returns a list of patterns, best first. Each has:
      name, type (bullish/bearish/neutral), confidence 0-1, plus the lines
      needed to draw it: {upper:[x1,y1,x2,y2], lower:[...]}
    """
    n = len(df)
    cut = n - lookback
    lo = [p for p in lows if p["i"] >= cut]
    hi = [p for p in highs if p["i"] >= cut]
    close = df["Close"].values
    vol = df["Volume"].values
    out = []

    if len(lo) < 2 or len(hi) < 2:
        return out

    fl = _fit([(p["i"], p["price"]) for p in lo])
    fh = _fit([(p["i"], p["price"]) for p in hi])
    if not fl or not fh:
        return out
    sl_lo, ic_lo, r2_lo = fl
    sl_hi, ic_hi, r2_hi = fh
    last_price = float(close[-1])

    ns_lo = _norm_slope(sl_lo, last_price, atr_v)
    ns_hi = _norm_slope(sl_hi, last_price, atr_v)
    x1, x2 = min(p["i"] for p in lo + hi), n - 1

    def lines():
        return {"upper": [x1, round(sl_hi * x1 + ic_hi, 2), x2, round(sl_hi * x2 + ic_hi, 2)],
                "lower": [x1, round(sl_lo * x1 + ic_lo, 2), x2, round(sl_lo * x2 + ic_lo, 2)]}

    FLAT = 0.05     # |slope| below this (ATR/bar) counts as flat
    fit_q = (r2_lo + r2_hi) / 2

    # ── converging = triangle family ─────────────────────────────────────
    converging = (sl_hi < 0 and sl_lo > 0)
    if converging and fit_q > 0.3:
        out.append({"name": "Symmetrical Triangle", "type": "neutral",
                    "confidence": round(min(1, fit_q), 2), "lines": lines(),
                    "note": "Converging highs and lows — a breakout is pending; direction follows the break."})
    elif abs(ns_hi) < FLAT and sl_lo > 0 and fit_q > 0.3:
        out.append({"name": "Ascending Triangle", "type": "bullish",
                    "confidence": round(min(1, fit_q), 2), "lines": lines(),
                    "note": "Flat resistance with rising lows — buyers paying up into a fixed ceiling."})
    elif abs(ns_lo) < FLAT and sl_hi < 0 and fit_q > 0.3:
        out.append({"name": "Descending Triangle", "type": "bearish",
                    "confidence": round(min(1, fit_q), 2), "lines": lines(),
                    "note": "Flat support with falling highs — sellers pressing into a fixed floor."})
    elif abs(ns_hi) < FLAT and abs(ns_lo) < FLAT and fit_q > 0.35:
        out.append({"name": "Rectangle / Range", "type": "neutral",
                    "confidence": round(min(1, fit_q), 2), "lines": lines(),
                    "note": "Horizontal range — accumulation or distribution, unresolved."})

    # ── wedges: both lines same direction, converging ────────────────────
    if sl_hi < 0 and sl_lo < 0 and sl_lo < sl_hi and fit_q > 0.3:
        out.append({"name": "Falling Wedge", "type": "bullish",
                    "confidence": round(min(1, fit_q), 2), "lines": lines(),
                    "note": "Both boundaries falling but converging — downside momentum fading; typically resolves up."})
    if sl_hi > 0 and sl_lo > 0 and sl_hi < sl_lo and fit_q > 0.3:
        out.append({"name": "Rising Wedge", "type": "bearish",
                    "confidence": round(min(1, fit_q), 2), "lines": lines(),
                    "note": "Both boundaries rising but converging — upside momentum fading; often resolves down."})

    # ── flag / pennant: sharp pole, then a tight counter-drift ───────────
    pole_win = 20
    if n > pole_win + 12:
        seg = close[max(0, n - lookback):n - 12]
        if len(seg) > 5:
            pole = (seg.max() - seg.min()) / atr_v if atr_v else 0
            recent = close[-12:]
            drift = (recent[-1] - recent[0]) / atr_v if atr_v else 0
            tight = (recent.max() - recent.min()) / atr_v if atr_v else 99
            rose = seg[-1] > seg[0]
            # volume should contract during the flag -- that's what separates a
            # healthy consolidation from genuine distribution
            v_pole = float(vol[max(0, n - lookback):n - 12].mean())
            v_flag = float(vol[-12:].mean())
            contracted = v_flag < v_pole if v_pole else False
            if pole >= 3 and tight <= 2.5 and rose and -1.5 <= drift <= 0.5:
                conf = 0.55 + (0.25 if contracted else 0) + min(0.2, pole / 25)
                out.append({"name": "Bull Flag" if drift < -0.2 else "Pennant",
                            "type": "bullish", "confidence": round(min(1, conf), 2),
                            "lines": {"upper": [n - 12, round(float(recent.max()), 2), x2, round(float(recent.max()), 2)],
                                      "lower": [n - 12, round(float(recent.min()), 2), x2, round(float(recent.min()), 2)]},
                            "note": ("Sharp advance then a tight, low-volume drift — classic continuation."
                                     if contracted else
                                     "Sharp advance then a tight drift, but volume has not contracted — less reliable.")})

    # ── double bottom: two similar lows with a peak between ──────────────
    if len(lo) >= 2:
        a, b = lo[-2], lo[-1]
        if abs(a["price"] - b["price"]) <= 0.8 * atr_v and (b["i"] - a["i"]) >= 8:
            mids = [h for h in hi if a["i"] < h["i"] < b["i"]]
            if mids:
                neck = max(mids, key=lambda h: h["price"])
                depth = (neck["price"] - min(a["price"], b["price"])) / atr_v
                if depth >= 1.5:
                    out.append({"name": "Double Bottom", "type": "bullish",
                                "confidence": round(min(1, 0.5 + depth / 10), 2),
                                "lines": {"upper": [a["i"], round(neck["price"], 2), x2, round(neck["price"], 2)],
                                          "lower": [a["i"], round(a["price"], 2), b["i"], round(b["price"], 2)]},
                                "note": f"Two lows at roughly the same level with a peak between — neckline ₹{neck['price']:.2f} is the trigger."})

    # ── double top ───────────────────────────────────────────────────────
    if len(hi) >= 2:
        a, b = hi[-2], hi[-1]
        if abs(a["price"] - b["price"]) <= 0.8 * atr_v and (b["i"] - a["i"]) >= 8:
            mids = [l for l in lo if a["i"] < l["i"] < b["i"]]
            if mids:
                neck = min(mids, key=lambda l: l["price"])
                depth = (max(a["price"], b["price"]) - neck["price"]) / atr_v
                if depth >= 1.5:
                    out.append({"name": "Double Top", "type": "bearish",
                                "confidence": round(min(1, 0.5 + depth / 10), 2),
                                "lines": {"upper": [a["i"], round(a["price"], 2), b["i"], round(b["price"], 2)],
                                          "lower": [a["i"], round(neck["price"], 2), x2, round(neck["price"], 2)]},
                                "note": f"Two highs at roughly the same level — breakdown below ₹{neck['price']:.2f} confirms."})

    return sorted(out, key=lambda p: -p["confidence"])[:3]
