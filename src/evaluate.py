import numpy as np
import pandas as pd

MA_SHORT = 50
MA_LONG = 200

def ma_signal(px: pd.DataFrame) -> pd.Series:
    ma50 = px["Close"].rolling(MA_SHORT).mean()
    ma200 = px["Close"].rolling(MA_LONG).mean()
    return (ma50 > ma200).astype(float).rename("ma_50_200")

def backtest(px, sig, cost_bps=2.0):
    """sig[t] in {0,1}, decided at the close of t -> position held over t+1."""
    ret = px["Close"].pct_change().reindex(sig.index)
    pos = sig.shift(1)
    turn = pos.diff().abs().fillna(pos.abs())
    gross = pos * ret
    return pd.DataFrame({"pos": pos, "ret": ret, "gross": gross,
                         "net": gross - turn * cost_bps / 1e4,
                         "turn": turn}).dropna()

def metrics(r, name=""):
    eq = (1 + r).cumprod(); n = len(r)
    cagr = eq.iloc[-1] ** (252 / n) - 1
    dd = (eq / eq.cummax() - 1).min()
    return {"strategy": name, "TotalRet": eq.iloc[-1] - 1, "CAGR": cagr,
            "Vol": r.std() * np.sqrt(252),
            "Sharpe": r.mean() / r.std() * np.sqrt(252),
            "MaxDD": dd, "days": n}
