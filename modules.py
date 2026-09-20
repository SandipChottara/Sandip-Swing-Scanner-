"""
The two setup modules, each scored out of 100 exactly as specified.

Mandatory conditions GATE (fail = not a candidate at all).
Confirmations SCORE (they rank the survivors).
Price structure carries the majority of the score in both models, so a stock
can never rank highly on indicators alone.
"""
import numpy as np
import pandas as pd

from patterns import detect_patterns
from structure import (
    CFG, atr, rsi, sma, find_pivots, label_pivots, trend_state,
    build_zones, zone_near, zone_below, zone_above,
)

# ── 100-point weights, per spec ──────────────────────────────────────────
REV_W = {"downtrend": 10, "support": 15, "ll_at_support": 10, "higher_low": 20,
         "resistance": 10, "breakout": 15, "volume": 10, "rsi": 5, "ma": 5}
CON_W = {"uptrend": 15, "trend_strength": 10, "impulse": 10, "pullback": 15,
         "support": 15, "higher_low": 15, "breakout": 10, "volume": 5,
         "rsi": 3, "rr": 2}




def _chart_series(df, lows, highs, bars=90):
    """
    Compact OHLC + pivots + MAs for the app to draw an inline SVG chart.
    Trimmed to the last `bars` and rounded, to keep data.json small.
    """
    d = df.tail(bars)
    off = len(df) - len(d)
    c = d["Close"]
    d20 = c.rolling(20).mean(); d50 = c.rolling(50).mean()
    def clean(sr):
        return [None if v != v else round(float(v), 2) for v in sr]
    return {
        "o": clean(d["Open"]), "h": clean(d["High"]),
        "l": clean(d["Low"]),  "c": clean(c),
        "v": [int(x) for x in d["Volume"].fillna(0)],
        "dates": (list(d["_date"]) if "_date" in d.columns
                  else [str(x)[:10] for x in d.index]),
        "dma20": clean(d20), "dma50": clean(d50),
        "offset": off,
        "pivot_lows":  [{"x": p["i"] - off, "y": round(p["price"], 2), "lab": p.get("label", "")}
                        for p in lows if p["i"] >= off],
        "pivot_highs": [{"x": p["i"] - off, "y": round(p["price"], 2), "lab": p.get("label", "")}
                        for p in highs if p["i"] >= off],
    }


def _rr_flags(rr, risk_pct, cfg):
    """
    R:R is only 2/100 in the continuation model and absent from reversal, so a
    structurally perfect setup can score A+ while being untradeable. Scoring
    stays as specified; these flags surface the problem instead of hiding it.
    """
    w = []
    if rr is None:
        w.append("Risk/reward could not be calculated")
    elif rr < 1:
        w.append(f"Poor risk/reward ({rr}:1) — reward is smaller than the risk")
    elif rr < cfg["min_rr"]:
        w.append(f"Below-target risk/reward ({rr}:1, want {cfg['min_rr']}:1+)")
    if risk_pct and risk_pct > 12:
        w.append(f"Wide stop — {risk_pct}% below entry; size the position down")
    return w


def _grade(s):
    return ("A+ candidate" if s >= 80 else "Strong candidate" if s >= 70 else
            "Watchlist" if s >= 60 else "Ignore")


def _vol_ratio(df, i, span=3):
    """Volume at bar i (best of i..i+2) vs the 20 bars BEFORE it."""
    v = df["Volume"]
    prior = v.iloc[max(0, i - 20):i]
    if not len(prior) or not float(prior.mean()):
        return None
    w = v.iloc[i:min(len(v), i + span)]
    return round(float(w.max()) / float(prior.mean()), 2)


def _leg_vol(df, a, b):
    """Average volume over a price leg, relative to the whole window."""
    seg = df["Volume"].iloc[a:b + 1]
    allv = float(df["Volume"].mean())
    return round(float(seg.mean()) / allv, 2) if len(seg) and allv else None


# ═════════════════════════════════════════════════════════════════════════
#  MODULE 1 — TREND REVERSAL
#  Downtrend -> Support -> LL -> HL -> LH breakout -> Volume -> RSI
# ═════════════════════════════════════════════════════════════════════════

def scan_reversal(sym, df, lows, highs, supports, resistances, cfg=CFG):
    def no(stage, why=""):
        return {"symbol": sym, "setup": "Reversal", "qualified": False,
                "failed_at": stage, "detail": why}

    n = len(df)
    last = float(df["Close"].iloc[-1])
    a_now = float(atr(df, cfg["atr_period"]).iloc[-1])
    r = rsi(df["Close"])

    # ── M1: established downtrend (LH + LL) ──────────────────────────────
    if trend_state(lows, highs) != "down":
        return no("1_downtrend", "no established LH+LL downtrend")

    # ── M4: the Higher Low — most recent swing low, labelled HL ──────────
    if len(lows) < 2:
        return no("4_higher_low", "not enough swing lows")
    hl, prior_ll = lows[-1], lows[-2]
    if hl["label"] != "HL":
        return no("4_higher_low", "latest swing low is not a higher low")

    # Test 1: meaningfully above the prior LL (ATR-normalised)
    gap_atr = (hl["price"] - prior_ll["price"]) / hl["atr"]
    if gap_atr < cfg["hl_min_atr"]:
        return no("4_higher_low", f"HL only {gap_atr:.2f} ATR above prior LL")

    # Test 2: meaningful bounce off the HL — proves buyers actually stepped in
    fwd = df.iloc[hl["i"]:]
    bounce_atr = (float(fwd["High"].max()) - hl["price"]) / hl["atr"] if len(fwd) else 0
    if bounce_atr < cfg["hl_bounce_atr"]:
        return no("4_higher_low", f"bounce off HL only {bounce_atr:.2f} ATR")

    # ── M3: the LL must have happened AT a real support zone ─────────────
    # The zone must have been ESTABLISHED BEFORE this low arrived at it --
    # otherwise the test is circular, since the LL is itself one of the pivots
    # that formed the zone. Require a touch predating the LL by a clear margin.
    cand = [z for z in supports
            if abs(prior_ll["price"] - z["price"]) / z["price"] * 100 <= cfg["near_support_pct"] * 2
            and z.get("first_i") is not None and z["first_i"] < prior_ll["i"] - 5]
    if not cand:
        return no("3_support", "no support zone that existed BEFORE this low")
    sup = max(cand, key=lambda z: z["score"])
    if sup["score"] < 4:
        return no("3_support", f"support too weak (score {sup['score']}/9)")

    # ── M5: the key Lower High — the LH formed BEFORE this LL/HL structure,
    # not simply the highest high in the window ─────────────────────────
    key_lh = None
    for h in reversed(highs):
        if h["i"] < hl["i"] and h.get("label") in ("LH", "—"):
            key_lh = h
            break
    if key_lh is None:
        return no("5_resistance", "no key Lower High identified")

    res_zone = zone_near(resistances, key_lh["price"], 3.0)

    # ── M6: has price CLOSED above that LH? ──────────────────────────────
    after = df.iloc[hl["i"]:]
    broke = after["Close"] > key_lh["price"]
    confirmed = bool(broke.any())
    bo_i = hl["i"] + int(np.argmax(broke.values)) if confirmed else None

    # Tiering per spec -- an unconfirmed setup is still worth surfacing,
    # just labelled honestly rather than presented as a buy.
    if not confirmed:
        near_pct = (key_lh["price"] - last) / last * 100
        tier = "Early reversal" if near_pct < 8 else "Watch"
    else:
        tier = "Confirmed reversal"

    vr = _vol_ratio(df, bo_i) if confirmed else None
    if confirmed and vr and vr >= cfg["vol_strong"]:
        tier = "Strong reversal"

    # Extension guard: continuation rejects stocks that have run too far, but
    # reversal had no equivalent -- a stock already 25% past its breakout still
    # qualified, which is exactly how the risk/reward gets ruined.
    if confirmed:
        ext = (last - key_lh["price"]) / key_lh["price"] * 100
        if ext > cfg["max_ext_above_trigger"]:
            return no("7_extended", f"{ext:.0f}% above the breakout level — entry would be chasing")

        # FRESHNESS. days_since_breakout was reported but never constrained, so
        # a breakout from 15 days ago still surfaced as a new pick today -- you
        # were buying the pullback after the move, not the move. A stock can sit
        # only slightly above the trigger yet be well past its momentum.
        age = len(df) - 1 - bo_i
        if age > cfg["max_days_since_breakout"]:
            return no("7_stale", f"breakout was {age} days ago — the move has already happened")

    # ── selling pressure weakening across the two declines ───────────────
    v_ll = _leg_vol(df, max(0, prior_ll["i"] - 15), prior_ll["i"])
    v_hl = _leg_vol(df, max(0, hl["i"] - 15), hl["i"])
    weakening = bool(v_ll and v_hl and v_hl < v_ll)

    # ── RSI: divergence + reclaim ────────────────────────────────────────
    rsi_ll, rsi_hl, rsi_now = float(r.iloc[prior_ll["i"]]), float(r.iloc[hl["i"]]), float(r.iloc[-1])

    # TRUE bullish divergence needs price to make a LOWER low while RSI makes a
    # HIGHER low. That can never happen between the LL and the HL (the HL is
    # higher by definition), so look for it earlier in the downtrend -- between
    # the last two genuinely lower lows. The previous check was dead code.
    diverg, div_pair = False, None
    lls = [p for p in lows if p["i"] <= prior_ll["i"] and p.get("label") in ("LL", "—")]
    for a, b in zip(lls, lls[1:]):
        if b["price"] < a["price"] and float(r.iloc[b["i"]]) > float(r.iloc[a["i"]]):
            diverg = True
            div_pair = {"low1": round(a["price"], 2), "low2": round(b["price"], 2),
                        "rsi1": round(float(r.iloc[a["i"]]), 1),
                        "rsi2": round(float(r.iloc[b["i"]]), 1)}
    improving = rsi_hl > rsi_ll

    # ── MAs ──────────────────────────────────────────────────────────────
    d20, d50 = sma(df["Close"], 20), sma(df["Close"], 50)
    above20 = bool(last > float(d20.iloc[-1])) if d20.notna().iloc[-1] else False
    above50 = bool(last > float(d50.iloc[-1])) if d50.notna().iloc[-1] else False
    d20_rising = bool(float(d20.iloc[-1]) > float(d20.iloc[-6])) if d20.notna().iloc[-1] else False

    # ── SCORE ────────────────────────────────────────────────────────────
    c = {}
    c["downtrend"] = REV_W["downtrend"]
    c["support"] = REV_W["support"] * (sup["score"] / 9)
    prox = abs(prior_ll["price"] - sup["price"]) / sup["price"] * 100
    c["ll_at_support"] = REV_W["ll_at_support"] * max(0, 1 - prox / 6)
    c["higher_low"] = (REV_W["higher_low"] * 0.5 * min(1, gap_atr / 1.5)
                       + REV_W["higher_low"] * 0.3 * min(1, bounce_atr / 2.5)
                       + REV_W["higher_low"] * 0.2 * (1 if weakening else 0))
    c["resistance"] = REV_W["resistance"] * ((res_zone["score"] / 9) if res_zone else 0.4)
    c["breakout"] = REV_W["breakout"] * (1.0 if confirmed else 0.0)
    c["volume"] = (REV_W["volume"] * min(1, (vr - 0.8) / 1.0) if vr else
                   REV_W["volume"] * 0.2 if weakening else 0)
    c["rsi"] = (REV_W["rsi"] * 0.5 * (1 if diverg else 0.5 if improving else 0)
                + REV_W["rsi"] * 0.5 * (1 if rsi_now >= 50 else 0.6 if rsi_now >= 40 else 0))
    c["ma"] = REV_W["ma"] * (0.5 * above20 + 0.3 * d20_rising + 0.2 * above50)
    total = sum(c.values())

    # ── trade levels: structure-based, never a fixed % ───────────────────
    stop = round(min(hl["price"], sup["price"]) * 0.99, 2)
    nxt = zone_above(resistances, last)
    t1 = nxt["price"] if nxt else round(last + 3 * a_now, 2)
    further = [z for z in resistances if z["price"] > t1]
    t2 = min(further, key=lambda z: z["price"])["price"] if further else round(last + 5 * a_now, 2)
    risk = last - stop
    rr = round((t1 - last) / risk, 2) if risk > 0 else None

    return {
        "symbol": sym, "setup": "Reversal", "qualified": True,
        "tier": tier, "confirmed": confirmed,
        "score": round(total, 1), "grade": _grade(total),
        "cmp": round(last, 2), "atr": round(a_now, 2),
        "structure": {
            "prior_ll": round(prior_ll["price"], 2), "prior_ll_i": prior_ll["i"],
            "higher_low": round(hl["price"], 2), "higher_low_i": hl["i"],
            "key_lh": round(key_lh["price"], 2), "key_lh_i": key_lh["i"],
            "hl_gap_atr": round(gap_atr, 2), "bounce_atr": round(bounce_atr, 2),
            "breakout_i": bo_i,
            "days_since_breakout": (len(df) - 1 - bo_i) if bo_i is not None else None,
        },
        "support": sup, "resistance": res_zone,
        "trigger_level": round(key_lh["price"], 2),
        "volume": {"breakout_ratio": vr, "decline_vol_ll": v_ll,
                   "decline_vol_hl": v_hl, "selling_weakening": weakening},
        "rsi": {"now": round(rsi_now, 1), "at_ll": round(rsi_ll, 1),
                "at_hl": round(rsi_hl, 1), "divergence": diverg,
                "divergence_detail": div_pair, "improving": bool(improving)},
        "ma": {"above_20dma": above20, "above_50dma": above50, "dma20_rising": d20_rising},
        "stop": stop, "target1": t1, "target2": t2, "rr": rr,
        "risk_pct": round((last - stop) / last * 100, 1),
        "reward_pct": round((t1 - last) / last * 100, 1),
        "warnings": _rr_flags(rr, round((last - stop) / last * 100, 1), cfg),
        "components": {k: round(v, 1) for k, v in c.items()},
        "patterns": detect_patterns(df, lows, highs, a_now),
        "chart": _chart_series(df, lows, highs),
    }


# ═════════════════════════════════════════════════════════════════════════
#  MODULE 2 — TREND CONTINUATION
#  Uptrend -> impulse -> controlled pullback -> support -> HL -> HH breakout
# ═════════════════════════════════════════════════════════════════════════

def scan_continuation(sym, df, lows, highs, supports, resistances, cfg=CFG):
    def no(stage, why=""):
        return {"symbol": sym, "setup": "Continuation", "qualified": False,
                "failed_at": stage, "detail": why}

    n = len(df)
    last = float(df["Close"].iloc[-1])
    a_now = float(atr(df, cfg["atr_period"]).iloc[-1])
    r = rsi(df["Close"])

    # ── M1: established uptrend (HH + HL) ────────────────────────────────
    if trend_state(lows, highs) != "up":
        return no("1_uptrend", "no established HH+HL uptrend")

    # ── M2: trend quality — 20 DMA > 50 DMA, 50 rising ───────────────────
    d20, d50 = sma(df["Close"], 20), sma(df["Close"], 50)
    if not (d20.notna().iloc[-1] and d50.notna().iloc[-1]):
        return no("2_trend_quality", "insufficient MA history")
    v20, v50 = float(d20.iloc[-1]), float(d50.iloc[-1])
    if v20 <= v50:
        return no("2_trend_quality", "20 DMA not above 50 DMA")
    d50_rising = bool(v50 > float(d50.iloc[-11])) if len(d50) > 11 else False

    # ── M6: the new Higher Low (the pullback low) ────────────────────────
    if len(lows) < 2 or len(highs) < 1:
        return no("6_higher_low", "not enough pivots")
    hl, prev_hl = lows[-1], lows[-2]
    if hl["label"] != "HL":
        return no("6_higher_low", "pullback low is not a higher low")

    # ── M3: meaningful prior impulse (>=2 ATR) ───────────────────────────
    prior_hh = None
    for h in reversed(highs):
        if h["i"] < hl["i"]:
            prior_hh = h
            break
    if prior_hh is None:
        return no("3_impulse", "no prior swing high before the pullback")
    impulse = prior_hh["price"] - prev_hl["price"]
    impulse_atr = impulse / hl["atr"] if hl["atr"] else 0
    if impulse_atr < 2:
        return no("3_impulse", f"prior advance only {impulse_atr:.1f} ATR")

    # ── M4: controlled pullback, 20-61.8% of the impulse ─────────────────
    pull = prior_hh["price"] - hl["price"]
    pull_pct = pull / impulse * 100 if impulse > 0 else 100
    if hl["price"] <= prev_hl["price"]:
        return no("4_pullback", "pullback broke the previous HL — structure damaged")
    if pull_pct > 61.8:
        return no("4_pullback", f"pullback {pull_pct:.0f}% of impulse — too deep")

    # ── M5: support at the pullback low ──────────────────────────────────
    sup = zone_near(supports, hl["price"], cfg["near_support_pct"] * 2)
    near_20dma = abs(hl["price"] - v20) / hl["price"] * 100 <= 4
    if sup is None and not near_20dma:
        return no("5_support", "pullback low not at support or the 20 DMA")

    # ── M7: breakout of the previous HH ──────────────────────────────────
    after = df.iloc[hl["i"]:]
    broke = after["Close"] > prior_hh["price"]
    confirmed = bool(broke.any())
    bo_i = hl["i"] + int(np.argmax(broke.values)) if confirmed else None
    vr = _vol_ratio(df, bo_i) if confirmed else None

    if confirmed:
        tier = "Strong continuation" if vr and vr >= cfg["vol_strong"] else "Confirmed continuation"
    else:
        gap = (prior_hh["price"] - last) / last * 100
        tier = "Coiling" if gap < 6 else "Watch"

    # ── M12: not excessively extended above the 20 DMA ───────────────────
    ext = (last - v20) / v20 * 100
    if ext > cfg["max_ext_above_20dma"]:
        return no("12_extended", f"{ext:.0f}% above the 20 DMA — poor risk/reward here")

    # Freshness -- same reasoning as reversal: a confirmed breakout from two
    # weeks ago is history, not a signal.
    if confirmed:
        age = len(df) - 1 - bo_i
        if age > cfg["max_days_since_breakout"]:
            return no("12_stale", f"breakout was {age} days ago — already played out")

    # pullback volume should CONTRACT vs the impulse (profit-taking, not distribution)
    v_imp = _leg_vol(df, prev_hl["i"], prior_hh["i"])
    v_pull = _leg_vol(df, prior_hh["i"], hl["i"])
    contracted = bool(v_imp and v_pull and v_pull < v_imp)

    rsi_now = float(r.iloc[-1]); rsi_hl = float(r.iloc[hl["i"]])
    rsi_ok = rsi_now >= 50 or (rsi_now >= 40 and rsi_now > rsi_hl)

    # ── trade levels ─────────────────────────────────────────────────────
    stop_ref = min(hl["price"], sup["price"]) if sup else hl["price"]
    stop = round(stop_ref * 0.99, 2)
    nxt = zone_above(resistances, max(last, prior_hh["price"]))
    t1 = nxt["price"] if nxt else round(prior_hh["price"] + 2 * a_now, 2)
    further = [z for z in resistances if z["price"] > t1]
    t2 = min(further, key=lambda z: z["price"])["price"] if further else round(t1 + 3 * a_now, 2)
    risk = last - stop
    rr = round((t1 - last) / risk, 2) if risk > 0 else None

    # ── SCORE ────────────────────────────────────────────────────────────
    c = {}
    c["uptrend"] = CON_W["uptrend"]
    c["trend_strength"] = CON_W["trend_strength"] * (0.6 + 0.4 * (1 if d50_rising else 0))
    c["impulse"] = CON_W["impulse"] * min(1, impulse_atr / 4)
    # pullback depth: 20-40% ideal, 40-50% good, deeper needs more proof
    depth = (1.0 if 20 <= pull_pct <= 40 else 0.8 if pull_pct <= 50 else
             0.55 if pull_pct <= 61.8 else 0.2)
    if pull_pct < 20:
        depth = 0.6                      # too shallow = little real consolidation
    c["pullback"] = CON_W["pullback"] * depth
    c["support"] = CON_W["support"] * ((sup["score"] / 9) if sup else 0.5)
    c["higher_low"] = CON_W["higher_low"] * min(1, (hl["price"] - prev_hl["price"]) / max(1e-9, hl["atr"]) / 1.5)
    c["breakout"] = CON_W["breakout"] * (1.0 if confirmed else 0.0)
    c["volume"] = (CON_W["volume"] * min(1, (vr - 0.8) / 1.0) if vr else
                   CON_W["volume"] * 0.3 if contracted else 0)
    c["rsi"] = CON_W["rsi"] * (1 if rsi_ok else 0.3)
    c["rr"] = CON_W["rr"] * (1 if (rr and rr >= cfg["min_rr"]) else 0.3)
    total = sum(c.values())

    return {
        "symbol": sym, "setup": "Continuation", "qualified": True,
        "tier": tier, "confirmed": confirmed,
        "score": round(total, 1), "grade": _grade(total),
        "cmp": round(last, 2), "atr": round(a_now, 2),
        "structure": {
            "prev_hl": round(prev_hl["price"], 2),
            "prior_hh": round(prior_hh["price"], 2), "prior_hh_i": prior_hh["i"],
            "higher_low": round(hl["price"], 2), "higher_low_i": hl["i"],
            "impulse_atr": round(impulse_atr, 1),
            "pullback_pct_of_impulse": round(pull_pct, 1),
            "ext_above_20dma": round(ext, 1),
            "breakout_i": bo_i,
            "days_since_breakout": (len(df) - 1 - bo_i) if bo_i is not None else None,
        },
        "support": sup, "resistance": zone_near(resistances, prior_hh["price"], 3.0),
        "trigger_level": round(prior_hh["price"], 2),
        "volume": {"breakout_ratio": vr, "impulse_vol": v_imp,
                   "pullback_vol": v_pull, "pullback_contracted": contracted},
        "rsi": {"now": round(rsi_now, 1), "at_hl": round(rsi_hl, 1), "healthy": bool(rsi_ok)},
        "ma": {"dma20": round(v20, 2), "dma50": round(v50, 2),
               "dma20_above_50": True, "dma50_rising": d50_rising},
        "stop": stop, "target1": t1, "target2": t2, "rr": rr,
        "risk_pct": round((last - stop) / last * 100, 1),
        "reward_pct": round((t1 - last) / last * 100, 1),
        "warnings": _rr_flags(rr, round((last - stop) / last * 100, 1), cfg),
        "components": {k: round(v, 1) for k, v in c.items()},
        "patterns": detect_patterns(df, lows, highs, a_now),
        "chart": _chart_series(df, lows, highs),
    }


# ═════════════════════════════════════════════════════════════════════════

def scan(sym, df, cfg=CFG):
    """
    One pass per stock. A stock is in a downtrend or an uptrend, not both, so
    the two modules are mutually exclusive -- whichever matches is returned.
    """
    if df is None or len(df) < 120:
        return {"symbol": sym, "qualified": False, "failed_at": "data"}
    df = df.tail(cfg["lookback_days"] + 60).copy()
    # Keep the real calendar dates before resetting the index -- the charting
    # library needs YYYY-MM-DD, and reset_index(drop=True) would otherwise
    # leave only integer positions.
    try:
        df["_date"] = [str(x)[:10] for x in df.index]
    except Exception:
        df["_date"] = [str(i) for i in range(len(df))]
    df = df.reset_index(drop=True)

    lows, highs = find_pivots(df, cfg)
    lows, highs = label_pivots(lows, highs)
    if len(lows) < 2 or len(highs) < 2:
        return {"symbol": sym, "qualified": False, "failed_at": "pivots",
                "detail": "not enough significant swings"}

    supports = build_zones(lows, df, "support", cfg)
    resistances = build_zones(highs, df, "resistance", cfg)

    st = trend_state(lows, highs)
    if st == "down":
        return scan_reversal(sym, df, lows, highs, supports, resistances, cfg)
    if st == "up":
        return scan_continuation(sym, df, lows, highs, supports, resistances, cfg)
    return {"symbol": sym, "qualified": False, "failed_at": "1_trend",
            "detail": "no clear trend — ranging"}
