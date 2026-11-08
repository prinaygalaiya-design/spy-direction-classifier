import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier

HORIZON = 5
RF = dict(n_estimators=500, max_depth=4, min_samples_leaf=50, max_features="sqrt")

def make_features(px: pd.DataFrame) -> pd.DataFrame:
    c, v = px["Close"], px["Volume"]
    r = np.log(c).diff()                       # known at today's close
    f = pd.DataFrame(index=px.index)
    f["ret_1"]      = r                        # immediate momentum
    f["ret_5"]      = r.rolling(5).sum()       # weekly momentum
    f["ret_20"]     = r.rolling(20).sum()      # medium-term momentum
    f["ma_dist_20"] = c / c.rolling(20).mean() - 1
    f["ma_dist_50"] = c / c.rolling(50).mean() - 1
    f["vol_10"]     = r.rolling(10).std() * np.sqrt(252)
    f["vol_20"]     = r.rolling(20).std() * np.sqrt(252)
    f["volu_rel"]   = v / v.rolling(20).mean()
    f["hl_range"]   = (px["High"] - px["Low"]) / c
    return f

def make_label(px: pd.DataFrame, horizon: int = HORIZON) -> pd.Series:
    fwd = px["Close"].shift(-horizon) / px["Close"] - 1
    return (fwd > 0).astype(float).where(fwd.notna())

def purged_split(idx, bounds, horizon=HORIZON):
    """bounds = {'train': ('2005-01-01','2016-12-31'), ...}"""
    out = {}
    for name, (s, e) in bounds.items():
        pos = np.flatnonzero((idx >= pd.Timestamp(s)) & (idx <= pd.Timestamp(e)))
        if len(pos) > horizon:
            pos = pos[:-horizon]               # purge the tail
        out[name] = idx[pos]
    return out

def make_model():
    return RandomForestClassifier(random_state=0, n_jobs=-1, **RF)
