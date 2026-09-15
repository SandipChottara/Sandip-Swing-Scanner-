"""Validation: synthetic charts with KNOWN structure, both modules."""
import numpy as np, pandas as pd
from modules import scan

def mk(closes, vols=None, seed=3, wig=0.006):
    rng = np.random.RandomState(seed)
    c = np.asarray(closes, float); n = len(c)
    v = np.full(n, 1e6) if vols is None else np.asarray(vols, float)
    return pd.DataFrame({
        "Open": c,
        "High": c * (1 + np.abs(rng.normal(0, wig, n))),
        "Low":  c * (1 - np.abs(rng.normal(0, wig, n))),
        "Close": c, "Volume": v,
    }, index=pd.bdate_range(end="2026-09-04", periods=n))

def L(a, b, n): return list(np.linspace(a, b, n))

def pad(c, start=None, n=70):
    """Lead-in bars so charts clear the 120-bar minimum, mildly drifting."""
    start = c[0] if start is None else start
    return L(start*1.02, c[0], n) + c

# ── REVERSAL: LH 160 -> LL 100 -> LH 130 -> LL 102 -> HL 112 -> break 130
def rev_good():
    c = pad(L(180,160,12)+L(160,100,20)+L(100,130,14)+L(130,102,16)+
         L(102,124,12)+L(124,112,10)+L(112,138,14)+L(138,136,8), 190)
    v = ([1e6]*70+[1e6]*12+[1.8e6]*20+[1.1e6]*14+[0.9e6]*16+
         [1.0e6]*12+[0.7e6]*10+[1.0e6]*6+[2.2e6]*8+[1.2e6]*8)
    return mk(c, v[:len(c)])

# ── REVERSAL but weak HL (barely above prior low) -> should reject
def rev_weak_hl():
    c = pad(L(180,160,12)+L(160,100,20)+L(100,130,14)+L(130,102,16)+
         L(102,124,12)+L(124,102.4,10)+L(102.4,118,14)+L(118,116,8))
    return mk(c, [1e6]*len(c))

# ── CONTINUATION: HL 100 -> HH 140 -> HL 126 -> break 140
def con_good():
    c = pad(L(90,100,10)+L(100,140,22)+L(140,126,14)+L(126,152,18)+L(152,150,8), 85)
    v = [1e6]*70+[1e6]*10+[1.6e6]*22+[0.7e6]*14+[1.0e6]*10+[2.1e6]*8+[1.1e6]*8
    return mk(c, v[:len(c)])

# ── CONTINUATION but pullback too deep (>61.8%) -> reject
def con_deep():
    c = pad(L(90,100,10)+L(100,140,22)+L(140,108,16)+L(108,132,18)+L(132,130,8), 85)
    return mk(c, [1e6]*len(c))

# ── CONTINUATION but extended far above 20 DMA -> reject
def con_extended():
    c = pad(L(90,100,10)+L(100,140,22)+L(140,130,10)+L(130,190,26), 85)
    return mk(c, [1e6]*len(c))

# ── Ranging chop -> neither
def ranging():
    c = []
    for _ in range(12): c += L(100,110,7)+L(110,100,7)
    return mk(c, [1e6]*len(c))

CASES = [
    ("REV_GOOD",  rev_good,     "Reversal",     True,  "textbook reversal"),
    ("REV_WEAKHL",rev_weak_hl,  None,           False, "HL too close to prior LL"),
    ("CON_GOOD",  con_good,     "Continuation", True,  "textbook continuation"),
    ("CON_DEEP",  con_deep,     None,           False, "pullback deeper than 61.8%"),
    ("CON_EXT",   con_extended, None,           False, "extended far above 20 DMA"),
    ("RANGING",   ranging,      None,           False, "no trend"),
]

if __name__ == "__main__":
    print("="*76); print("  SWING SCANNER — VALIDATION (Reversal + Continuation)"); print("="*76)
    ok = 0
    for name, fn, want_setup, want_q, desc in CASES:
        res = scan(name, fn())
        got_q = res.get("qualified", False)
        got_setup = res.get("setup")
        good = (got_q == want_q) and (not want_q or got_setup == want_setup)
        ok += good
        print(f"\n[{'PASS' if good else 'FAIL'}] {name:11} {desc}")
        if got_q:
            s = res["structure"]
            print(f"       {got_setup} | {res['tier']} | score {res['score']} ({res['grade']})")
            print(f"       trigger ₹{res['trigger_level']} | stop ₹{res['stop']} | "
                  f"T1 ₹{res['target1']} | R:R {res['rr']}")
            if res.get("support"):
                sp = res["support"]
                print(f"       support ₹{sp['price']} [{sp['low']}-{sp['high']}] "
                      f"{sp['touches']} touches, {sp['score']}/9 {sp['grade']}")
            print(f"       components {res['components']}")
        else:
            print(f"       rejected at {res.get('failed_at')} — {res.get('detail','')}")
    print("\n"+"="*76); print(f"  {ok}/{len(CASES)} behaved as expected"); print("="*76)
