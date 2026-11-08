# SPY 5-Day Direction — Version 1 Spec

Scope: the smallest complete version you can defend line by line. Walk-forward, transaction-cost
sensitivity, permutation tests and cross-asset features are deferred to v2/v3.

**Research question:** Can a small set of price and volume features predict whether SPY closes
higher five trading days later, and does any predictive ability survive into a long/flat strategy
that beats simple benchmarks?

---

## 1. Environment

```bash
mkdir -p ~/projects/spy-direction && cd ~/projects/spy-direction
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install "yfinance>=0.2.40" pandas numpy scikit-learn matplotlib jupyterlab pyarrow
pip freeze > requirements.txt
git init && printf ".venv/\ndata/\n__pycache__/\n.ipynb_checkpoints/\n" > .gitignore
```

```
spy-direction/
├── data/raw/
├── src/{data.py, pipeline.py, evaluate.py}
├── notebooks/01_explore.ipynb
├── results/
├── run.py
└── README.md
```

---

## 2. Data

`src/data.py`:

```python
from pathlib import Path
import pandas as pd, yfinance as yf

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

def load(ticker="SPY", start="2005-01-01", refresh=False) -> pd.DataFrame:
    path = RAW / f"{ticker}.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    df = yf.Ticker(ticker).history(start=start, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No data for {ticker}")
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index = df.index.normalize(); df.index.name = "Date"
    df = df[["Open", "High", "Low", "Close", "Volume"]].astype("float64")
    df.to_parquet(path)
    return df
```

`auto_adjust=True` is required — unadjusted prices contain fake overnight gaps on every dividend
date and the model will learn to "predict" them.

Sanity checks before modelling anything:

```python
spy = load()
print(spy.shape, spy.index.min(), spy.index.max())
print(spy.isna().sum(), spy.index.duplicated().sum())   # both must be 0
r = spy["Close"].pct_change()
print(r.describe()); print(r.abs().nlargest(10))
spy["Close"].plot(logy=True)
```

Largest single-day moves should be recognisable: ~-12% on 2020-03-16, ~-9% on 2008-10-15,
~-7% on 2008-09-29. If your biggest move is 40%, the data is broken. Fix it now.

---

## 3. Features and label

Nine features. Every one is a ratio, return or rolling statistic — never a raw price level, or the
model learns which decade it is in rather than a repeatable relationship.

`src/pipeline.py`:

```python
import numpy as np, pandas as pd

HORIZON = 5

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
```

**Governing rule:** a feature dated day *t* may only use information that existed at the close of
day *t*. `shift(-n)` appears exactly once in the whole codebase, in `make_label`. Verify with
`grep -rn "shift(-" src/` — one hit, or you have a bug.

No full-sample z-scores, no scaler fit on all the data, no `.rolling()` with `center=True`.

Then:

```python
X = make_features(spy); y = make_label(spy)
d = X.join(y.rename("y")).dropna()
X, y = d.drop(columns="y"), d["y"]
print(X.shape, "base rate:", round(y.mean(), 4))     # expect ~0.55-0.57
```

Write that base rate down. It is the number you have to beat, not 50%.

**Worth knowing about your own label:** with h=5, Monday's and Tuesday's labels share four of their
five days. Your labels are heavily autocorrelated, so your effective sample size is roughly a fifth
of your row count. That's the reason for the purge in step 4, and it's also why you shouldn't
over-trust any single accuracy figure to three decimal places.

---

## 4. The purged chronological split

| Block | Period | Use |
|---|---|---|
| Train | 2005 → 2016 | fit the model |
| Validation | 2017 → 2020 | choose the threshold |
| Test | 2021 → latest | answer the question, once |

Drop the last five trading rows of each block. Without that, the final training label depends on
prices inside the validation period.

```python
def purged_split(idx, bounds, horizon=HORIZON):
    """bounds = {'train': ('2005-01-01','2016-12-31'), ...}"""
    out = {}
    for name, (s, e) in bounds.items():
        pos = np.flatnonzero((idx >= pd.Timestamp(s)) & (idx <= pd.Timestamp(e)))
        if len(pos) > horizon:
            pos = pos[:-horizon]               # purge the tail
        out[name] = idx[pos]
    return out
```

I ran this on a 2005–2026 series: it produces a 10-calendar-day gap between train and validation and
an 8-day gap between validation and test. Print your own gaps and put them in the README.

The one-sentence defence: *"my target uses the following five trading days, so I removed the last
five observations of each block to stop a training label incorporating prices from the next block."*

**Then don't open the test set until everything is frozen.** Every time you look at test results and
adjust something, you convert test into validation and your reported number becomes optimistic.

**Do not refit on train+validation.** Fit the model once on 2005–2016, use 2017–2020 only to pick
the threshold, then open the test set. Refitting is standard practice and would use four extra years
of data, but keeping the train-only model makes the experiment cleaner to state: *one model trained
once, one strategy decision made on validation, everything frozen, then the test set opened.* Note
in the README that you chose not to refit and why.

---

## 5. Baselines first

Compute these before you look at any model output:

1. **Always up** — accuracy equals the base rate, and the strategy is buy-and-hold.
2. **Buy and hold SPY** — return benchmark.
3. **MA rule** — long when 50-day MA > 200-day MA, else cash. Same one-bar lag, same costs.

The MA rule matters because without it the comparison is just "ML versus doing nothing." Beating
buy-and-hold is one claim; beating a two-line technical rule is a different and harder one.

---

## 6. Model

One Random Forest, restrained:

```python
from sklearn.ensemble import RandomForestClassifier

RF = dict(n_estimators=500, max_depth=4, min_samples_leaf=50, max_features="sqrt")
m = RandomForestClassifier(random_state=0, n_jobs=-1, **RF)
m.fit(X.loc[sp["train"]], y.loc[sp["train"]])
```

Defaults (`max_depth=None`) will hit ~100% training accuracy and produce noise out of sample.
Financial data has a signal-to-noise ratio around 1:100; shallow trees and large leaves are the
whole point. Understand and be able to explain: what a decision tree splits on, why single trees
overfit, why bagging + feature subsampling reduces variance, what `max_depth` and
`min_samples_leaf` do, and why financial data has a poor signal-to-noise ratio.

**No `class_weight` in v1.** A 56/44 split is mild imbalance, and `class_weight="balanced_subsample"`
recentres the model scores so that 0.50 no longer means anything interpretable — which makes a fixed
threshold grid arbitrary. I measured this: on the same data, balancing shifted the mean validation
score from 0.521 to 0.499 and cut exposure at a 0.55 threshold from 24% to 11%. Leave it out, handle
the imbalance through the threshold (step 8), and say so in the README.

No grid search in v1.

---

## 7. Classification results

```python
from sklearn.metrics import roc_auc_score, confusion_matrix

for split in ("val", "test"):
    ix = sp[split]
    p = pd.Series(m.predict_proba(X.loc[ix])[:, 1], index=ix)
    print(split, "acc", round(((p > .5).astype(float) == y.loc[ix]).mean(), 4),
          "| always-up", round(y.loc[ix].mean(), 4),
          "| AUC", round(roc_auc_score(y.loc[ix], p), 4))
```

Reading it:
- **AUC is the honest signal measure** — imbalance-insensitive. 0.50 is nothing, 0.52–0.54 is a
  plausible weak edge, above 0.60 on daily index data means leakage. Go find it.
- **Accuracy below the base rate** means worse than always-long at the classification task.
- **Print `p.describe()`.** If probabilities all sit between 0.48 and 0.52, your threshold choice is
  slicing noise and the resulting strategy will trade constantly for no reason.
- **Break AUC down by year.** AUC 0.58 in 2020 and 0.49 everywhere else is one crisis, not an edge.

---

## 8. Threshold — chosen on validation, frozen before test

**Define thresholds as exposure targets, not as absolute score values.**

Random Forest scores from `predict_proba()` are not calibrated probabilities, and on this kind of
data they cluster tightly around the base rate. I measured a validation-score standard deviation of
0.039, which makes a fixed grid of 0.50 / 0.55 / 0.60 badly placed: at 0.60 the model was long
**1.6% of days**, so that candidate is effectively dead and the grid is really a choice between two
options. Absolute cut values also move with the base rate and with any modelling change, so they
aren't comparable across versions.

Instead, target exposure directly:

```python
pv = pd.Series(m.predict_proba(X.loc[sp["val"]])[:, 1], index=sp["val"])

for target in (0.70, 0.50, 0.30):          # fraction of days long
    cut = pv.quantile(1 - target)          # a frozen NUMERIC value
    ...
```

Compute the cut on the **validation** score distribution, freeze that number, and apply the same
number to the test set. Do not recompute the quantile on test — that would use test-period
information. In my run the exposures transferred cleanly: 70% / 50% / 30% on validation became
71% / 51% / 31% on test.

This also sidesteps the calibration problem entirely. You're using the score only as a *ranking*,
which is exactly what it's good for and exactly what AUC measures.

**Pre-registered selection rule, written down before you run it:**

> Choose the exposure target with the highest validation-period **net** Sharpe. If two are within
> 0.05 Sharpe of each other, take the higher exposure.

The tiebreak matters because with ~1,000 validation days these Sharpes are noisy, and differences
smaller than that aren't real. State the rule in your README before the results table so a reader
can see it wasn't chosen afterwards.

**Wording, in the README and out loud.** Don't write "the model estimated a 60% probability that SPY
would rise." Write "the model produced a score of 0.60 for the positive class, which placed that day
in the top 30% of the validation score distribution." The classifier ranks days; it does not emit
calibrated probabilities. Probability calibration (`CalibratedClassifierCV`, reliability diagrams) is
a good v3 extension, not a v1 requirement.

---

## 9. Backtest

`src/evaluate.py`:

```python
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
```

`pos = sig.shift(1)` is the most important line in the project. Without it you trade on the close
using the close, which is time travel and roughly doubles your Sharpe out of nowhere.

Long/flat only. Shorting SPY brings borrow costs, margin and constraints you aren't modelling.

**Describe the holding period accurately.** You are not holding each prediction for five days. Each
day produces a fresh five-day forecast and the position is set from the latest one, so the correct
description is: *"the classifier forecasts five-day direction, while the strategy is rebalanced
daily using the most recent five-day forecast."* That's a legitimate design — building genuine
overlapping five-day trades is a v2 refinement — but if you describe it as "I held each signal for a
week" an interviewer will read your code and find otherwise.

**Costs stay in v1.** They're three lines and turnover is usually what kills the strategy —
reporting a gross-only headline in v1 means your first result is systematically flattering and you
may draw the wrong conclusion. Report gross and net side by side; the gap between them is the
transaction-cost story.

**State the cost convention precisely.** The code charges `cost_bps` per unit of turnover, and a
0 → 1 → 0 cycle is two units. So `cost_bps=2.0` means **2bp per transaction, 4bp for a complete
entry-and-exit round trip** — not 2bp round trip. If you want 2bp round trip, pass `cost_bps=1.0`.
Either is a reasonable assumption for SPY; just write down which one you used.

Print annualised turnover:

```python
print("turnover/yr:", round(bt["turn"].sum() / len(bt) * 252, 1))
```

---

## 10. The comparison table

Include exposure and volatility. Not optional.

| | RF net | RF gross | MA rule | Buy & hold |
|---|---|---|---|---|
| Accuracy vs base rate | | — | — | — |
| AUC | | — | — | — |
| Exposure (% days long) | | | | 100% |
| Total return | | | | |
| CAGR | | | | |
| Annualised vol | | | | |
| Sharpe | | | | |
| Max drawdown | | | | |
| Turnover / yr | | | — | ~0 |

**Why exposure and vol belong in the table.** A strategy that sits in cash most of the time has low
volatility by construction, and Sharpe rewards that. In my synthetic run the strategy showed Sharpe
0.58 against buy-and-hold's 0.08 — with realised vol of 5.4% versus 17.6%, 10% exposure, and an AUC
of 0.5028, i.e. **no predictive signal whatsoever**. Comparing Sharpe across those two things is
apples to oranges. If you want a like-for-like return comparison, scale the strategy's returns to
buy-and-hold's volatility and compare CAGR at matched vol. At minimum, show both vols next to the
Sharpes so a reader can see it.

Then four plots, saved to `results/` at dpi=150 with labelled axes:

1. Equity curves, log scale, both series on one axis
2. Drawdown curves below, shared x-axis
3. Rolling 252-day Sharpe for both
4. Per-calendar-year returns, strategy vs buy-and-hold, as a bar chart

Plot 4 matters: 2021–2026 is only about 250 independent 5-day observations. The aggregate result
will be dominated by whether the model happened to be flat during 2022. Show the year-by-year
breakdown rather than letting one regime carry the headline.

---

## 11. The two smoke detectors (in v1)

### 11a. Synthetic random data

Generate a random walk with no signal, push it through the identical pipeline:

```python
rng = np.random.default_rng(1)
dates = pd.bdate_range("2005-01-03", "2026-09-01"); n = len(dates)
c = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.011, n)))
fake = pd.DataFrame({"Open": c, "Close": c,
                     "High": c * (1 + abs(rng.normal(0, .004, n))),
                     "Low":  c * (1 - abs(rng.normal(0, .004, n))),
                     "Volume": rng.lognormal(18, .3, n)}, index=dates)
```

**Get the pass criterion right.** The test is **AUC ≈ 0.50 on the synthetic test block** — not
"the strategy loses to buy-and-hold." When I ran it, AUC came out at 0.5028 (correct: no signal)
while the strategy still beat buy-and-hold on Sharpe, because it was flat 90% of the time and the
random walk drifted sideways. If you use return metrics as the pass criterion you will pass a broken
pipeline or fail a working one. AUC is the criterion; the return metrics are the illustration.

### 11b. Leakage canary

```python
X_cheat = X.copy()
X_cheat["CHEAT"] = (spy["Close"].shift(-5) / spy["Close"] - 1).reindex(X.index)
```

Refit, rerun. On random data with zero signal, adding that one column took my result to **CAGR 68%,
Sharpe 4.2, max drawdown -4.6%**. Memorise that shape. If your real results ever look like it, you
have a `shift` pointing the wrong way.

Both tests go in the README under "pipeline validation," with the numbers. The defence is one
sentence each and it's stronger than asserting you were careful:

> *"Before trusting the result I ran the pipeline on data I knew contained no predictive
> relationship, and separately injected a future-looking feature to confirm the metrics responded
> as expected."*

---

## 12. Build order

1. Download SPY, sanity checks, price plot
2. Nine features + 5-day label; print the base rate
3. Purged train/val/test split; print the boundary gaps
4. Three baselines
5. Fit the Random Forest
6. Accuracy + AUC on validation, `p.describe()`, per-year AUC
7. Write the selection rule down; evaluate exposure targets {70%, 50%, 30%} on validation; freeze the cut
8. Signal → `shift(1)` → gross and net backtest on test
9. Comparison table with exposure and vol; four plots
10. Synthetic-data run (pass criterion: AUC ≈ 0.50)
11. Leakage canary
12. README; `run.py` reproduces everything end to end
13. Freeze v1

Roughly six sessions of 2–3 hours.

---

## 13. README structure

1. **Summary with the conclusion first**, numbers included.
2. Data, period, sanity checks.
3. Features table and the lag rule.
4. Split diagram, purge, and the never-touch-test policy.
5. Baselines.
6. Model and why the hyperparameters are restrained.
7. Threshold protocol — exposure targets, selection rule stated before results, cut frozen from validation.
8. Results: table + four figures + per-year breakdown.
9. Pipeline validation: 11a and 11b with numbers.
10. Limitations: single asset, single split, overlapping labels reduce effective sample size,
    no slippage model, rf rate assumed zero, exposure mismatch vs buy-and-hold.
11. Next steps: walk-forward with embargo, probability calibration, volatility regime conditioning,
    cross-sectional version.
12. Reproduction: `pip install -r requirements.txt && python run.py`.

A negative result stated precisely is the target outcome, not a fallback:

> *Nine price and volume features gave out-of-sample AUC of 0.5xx on five-day SPY direction.
> Accuracy of 55.x% was below the 56.x% always-long base rate. The long/flat strategy returned
> x% CAGR at y% volatility against z% / w% for buy-and-hold, with exposure of only n% of days. The
> features do not contain stable directional information at this horizon.*

---

## Pitfalls checklist

- [ ] `grep -rn "shift(-" src/` → exactly one hit, in `make_label`
- [ ] No `train_test_split`, `KFold`, or `shuffle=True` anywhere
- [ ] Five-row purge applied at every block boundary; gaps printed
- [ ] Nothing fit on the full dataset (scalers, means, z-scores)
- [ ] `auto_adjust=True`
- [ ] `pos = sig.shift(1)`
- [ ] Costs applied to turnover; turnover reported
- [ ] Base rate reported beside every accuracy figure
- [ ] Exposure and annualised vol in the comparison table
- [ ] Threshold defined as an exposure target, cut value frozen from validation
- [ ] Selection rule written down before validation results were seen
- [ ] Cost convention stated (per transaction vs round trip)
- [ ] Model trained once on 2005-2016; no refit on train+validation
- [ ] Scores described as model scores, not calibrated probabilities
- [ ] Synthetic run gives AUC ≈ 0.50
- [ ] Leakage canary gives an absurd result
- [ ] Per-year breakdown shown, not just the aggregate
- [ ] Every figure has axis labels

---

## Deferred

**v2:** expanding-window walk-forward with embargo, transaction-cost sensitivity sweep, volatility
regime split (high vs low trailing vol), rolling feature importance.

**v3:** permutation null distribution, multiple seeds, horizon and threshold sensitivity, permutation
feature importance, risk-free-adjusted Sharpe.

**v4:** cross-asset features, or — better — drop the timing problem entirely and do the
cross-sectional version: rank a universe of stocks daily, long the top decile. Far more independent
observations, and closer to what equity quant desks actually do.