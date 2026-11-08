from pathlib import Path
import os
import tempfile

mpl_config = Path(tempfile.gettempdir()) / "spy_direction_mplconfig"
mpl_config.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(mpl_config))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.data import load
from src.evaluate import MA_LONG, MA_SHORT, backtest, ma_signal, metrics
from src.pipeline import RF, make_features, make_label, make_model, purged_split


FROZEN_CUTOFF = 0.540488139218
SELECTED_TARGET = 0.70
TARGETS = (0.70, 0.50, 0.30)
COST_BPS = 2.0
RESULTS = Path("results")


def build_xy(px):
    X = make_features(px)
    y = make_label(px)
    d = X.join(y.rename("y")).dropna()
    return d.drop(columns="y"), d["y"]


def build_splits(idx):
    bounds = {
        "train": ("2005-01-01", "2016-12-31"),
        "val": ("2017-01-01", "2020-12-31"),
        "test": ("2021-01-01", str(idx.max().date())),
    }
    return purged_split(idx, bounds)


def fit_train_model(X, y, sp):
    model = make_model()
    model.fit(X.loc[sp["train"]], y.loc[sp["train"]])
    return model


def equity(r):
    return (1 + r).cumprod()


def drawdown(r):
    eq = equity(r)
    return eq / eq.cummax() - 1


def rolling_sharpe(r):
    return r.rolling(252).mean() / r.rolling(252).std() * np.sqrt(252)


def yearly_returns(r):
    return (1 + r).groupby(r.index.year).prod() - 1


def validation_selection(px, p_val):
    rows = []
    for target in TARGETS:
        cut = p_val.quantile(1 - target)
        sig = (p_val > cut).astype(float)
        bt = backtest(px, sig, cost_bps=COST_BPS)
        m = metrics(bt["net"], name=f"RF net {target:.0%}")
        rows.append({
            "candidate_target_exposure": target,
            "cutoff": cut,
            "actual_validation_exposure": bt["pos"].mean(),
            "net_sharpe": m["Sharpe"],
            "net_cagr": m["CAGR"],
            "maxdd": m["MaxDD"],
            "turnover_per_year": bt["turn"].sum() / len(bt) * 252,
        })

    out = pd.DataFrame(rows)
    max_sharpe = out["net_sharpe"].max()
    eligible = out[max_sharpe - out["net_sharpe"] <= 0.05]
    selected = eligible.sort_values("candidate_target_exposure", ascending=False).iloc[0]

    if selected["candidate_target_exposure"] != SELECTED_TARGET:
        raise RuntimeError(
            f"Validation rule selected {selected['candidate_target_exposure']}, "
            f"expected frozen target {SELECTED_TARGET}."
        )
    if abs(selected["cutoff"] - FROZEN_CUTOFF) > 1e-12:
        raise RuntimeError(
            f"Reproduced cutoff {selected['cutoff']:.12f} differs from "
            f"frozen cutoff {FROZEN_CUTOFF:.12f}."
        )

    return out, selected


def comparison_table(test_base_rate, test_acc, test_auc, bt_rf, bt_ma, bh_ret):
    rf_net = metrics(bt_rf["net"], "RF net")
    rf_gross = metrics(bt_rf["gross"], "RF gross")
    ma = metrics(bt_ma["net"], "MA rule")
    bh = metrics(bh_ret, "Buy & hold")

    turn_rf = bt_rf["turn"].sum() / len(bt_rf) * 252
    turn_ma = bt_ma["turn"].sum() / len(bt_ma) * 252

    return pd.DataFrame({
        "RF net": [
            f"{test_acc:.6f} vs {test_base_rate:.6f}",
            test_auc,
            bt_rf["pos"].mean(),
            rf_net["TotalRet"],
            rf_net["CAGR"],
            rf_net["Vol"],
            rf_net["Sharpe"],
            rf_net["MaxDD"],
            turn_rf,
        ],
        "RF gross": [
            "",
            "",
            bt_rf["pos"].mean(),
            rf_gross["TotalRet"],
            rf_gross["CAGR"],
            rf_gross["Vol"],
            rf_gross["Sharpe"],
            rf_gross["MaxDD"],
            turn_rf,
        ],
        "MA rule": [
            "",
            "",
            bt_ma["pos"].mean(),
            ma["TotalRet"],
            ma["CAGR"],
            ma["Vol"],
            ma["Sharpe"],
            ma["MaxDD"],
            turn_ma,
        ],
        "Buy & hold": [
            "",
            "",
            1.0,
            bh["TotalRet"],
            bh["CAGR"],
            bh["Vol"],
            bh["Sharpe"],
            bh["MaxDD"],
            0.0,
        ],
    }, index=[
        "Accuracy vs TEST base rate",
        "ROC-AUC",
        "Exposure (% days long)",
        "Total return",
        "CAGR",
        "Annualised volatility",
        "Sharpe",
        "Maximum drawdown",
        "Turnover / year",
    ])


def make_plots(rf_net, bh_ret):
    RESULTS.mkdir(exist_ok=True)

    rf_eq = equity(rf_net)
    bh_eq = equity(bh_ret)
    rf_dd = drawdown(rf_net)
    bh_dd = drawdown(bh_ret)
    rf_rs = rolling_sharpe(rf_net)
    bh_rs = rolling_sharpe(bh_ret)
    yearly = pd.DataFrame({
        "RF net": yearly_returns(rf_net),
        "Buy and hold SPY": yearly_returns(bh_ret),
    }).loc[[2021, 2022, 2023, 2024, 2025, 2026]]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(rf_eq.index, rf_eq, label="RF net strategy")
    ax.plot(bh_eq.index, bh_eq, label="Buy and hold SPY")
    ax.set_yscale("log")
    ax.set_title("TEST Equity Curves")
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth of $1")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "equity_curves.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(rf_dd.index, rf_dd, label="RF net drawdown")
    ax.plot(bh_dd.index, bh_dd, label="Buy and hold drawdown")
    ax.set_title("TEST Drawdowns")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "drawdowns.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(rf_rs.index, rf_rs, label="RF net")
    ax.plot(bh_rs.index, bh_rs, label="Buy and hold SPY")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title("TEST Rolling 252-Day Sharpe")
    ax.set_xlabel("Date")
    ax.set_ylabel("Rolling 252-Day Sharpe")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "rolling_sharpe.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    yearly.plot(kind="bar", ax=ax)
    ax.set_title("TEST Calendar-Year Returns")
    ax.set_xlabel("Year")
    ax.set_ylabel("Return")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "yearly_returns.png", dpi=150)
    plt.close(fig)

    return yearly


def synthetic_data():
    rng = np.random.default_rng(1)
    dates = pd.bdate_range("2005-01-03", "2026-09-01")
    n = len(dates)
    c = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.011, n)))
    return pd.DataFrame({
        "Open": c,
        "Close": c,
        "High": c * (1 + abs(rng.normal(0, 0.004, n))),
        "Low": c * (1 - abs(rng.normal(0, 0.004, n))),
        "Volume": rng.lognormal(18, 0.3, n),
    }, index=dates)


def synthetic_smoke_and_canary():
    fake = synthetic_data()
    X_clean, y_clean = build_xy(fake)
    sp_clean = build_splits(X_clean.index)

    clean_model = fit_train_model(X_clean, y_clean, sp_clean)
    p_clean = pd.Series(
        clean_model.predict_proba(X_clean.loc[sp_clean["test"]])[:, 1],
        index=sp_clean["test"],
    )
    clean_auc = roc_auc_score(y_clean.loc[sp_clean["test"]], p_clean)

    X0 = make_features(fake)
    y0 = make_label(fake)
    X_cheat0 = X0.copy()
    X_cheat0["CHEAT"] = (fake["Close"].shift(-5) / fake["Close"] - 1).reindex(X_cheat0.index)
    d_cheat = X_cheat0.join(y0.rename("y")).dropna()
    X_cheat = d_cheat.drop(columns="y")
    y_cheat = d_cheat["y"]
    sp_cheat = build_splits(X_cheat.index)

    cheat_model = fit_train_model(X_cheat, y_cheat, sp_cheat)
    p_cheat = pd.Series(
        cheat_model.predict_proba(X_cheat.loc[sp_cheat["test"]])[:, 1],
        index=sp_cheat["test"],
    )
    cheat_auc = roc_auc_score(y_cheat.loc[sp_cheat["test"]], p_cheat)

    return clean_auc, cheat_auc


def main():
    RESULTS.mkdir(exist_ok=True)

    spy = load()
    X, y = build_xy(spy)
    sp = build_splits(X.index)
    model = fit_train_model(X, y, sp)

    p_val = pd.Series(model.predict_proba(X.loc[sp["val"]])[:, 1], index=sp["val"])
    val_auc = roc_auc_score(y.loc[sp["val"]], p_val)
    selection, selected = validation_selection(spy, p_val)
    selection.to_csv(RESULTS / "validation_selection.csv", index=False)

    p_test = pd.Series(model.predict_proba(X.loc[sp["test"]])[:, 1], index=sp["test"])
    test_pred_050 = (p_test > 0.5).astype(float)
    test_base_rate = y.loc[sp["test"]].mean()
    test_acc = (test_pred_050 == y.loc[sp["test"]]).mean()
    test_auc = roc_auc_score(y.loc[sp["test"]], p_test)

    sig_test = (p_test >= FROZEN_CUTOFF).astype(float)
    bt_rf = backtest(spy, sig_test, cost_bps=COST_BPS)
    sig_ma = ma_signal(spy).reindex(sp["test"])
    bt_ma = backtest(spy, sig_ma, cost_bps=COST_BPS)

    common = bt_rf.index.intersection(bt_ma.index)
    bt_rf = bt_rf.loc[common]
    bt_ma = bt_ma.loc[common]
    bh_ret = spy["Close"].pct_change().reindex(common)
    if not bh_ret.notna().all():
        raise RuntimeError("Buy-and-hold returns are not aligned cleanly.")

    headline = comparison_table(test_base_rate, test_acc, test_auc, bt_rf, bt_ma, bh_ret)
    headline.to_csv(RESULTS / "headline_metrics.csv")
    make_plots(bt_rf["net"].rename("RF net"), bh_ret.rename("Buy and hold SPY"))

    clean_auc, cheat_auc = synthetic_smoke_and_canary()

    rf_net = metrics(bt_rf["net"], "RF net")
    bh = metrics(bh_ret, "Buy & hold")
    ma = metrics(bt_ma["net"], "MA rule")

    print("SPY 5-Day Direction v1 summary")
    print(f"validation AUC: {val_auc:.6f}")
    print(f"selected exposure target: {selected['candidate_target_exposure']:.2f}")
    print(f"frozen cutoff: {FROZEN_CUTOFF:.12f}")
    print(f"TEST base rate: {test_base_rate:.6f}")
    print(f"TEST accuracy: {test_acc:.6f}")
    print(f"TEST AUC: {test_auc:.6f}")
    print(f"RF net CAGR: {rf_net['CAGR']:.6f}")
    print(f"RF net Sharpe: {rf_net['Sharpe']:.6f}")
    print(f"buy-and-hold CAGR: {bh['CAGR']:.6f}")
    print(f"buy-and-hold Sharpe: {bh['Sharpe']:.6f}")
    print(f"MA CAGR: {ma['CAGR']:.6f}")
    print(f"MA Sharpe: {ma['Sharpe']:.6f}")
    print(f"clean synthetic AUC: {clean_auc:.6f}")
    print(f"CHEAT synthetic AUC: {cheat_auc:.6f}")


if __name__ == "__main__":
    main()
