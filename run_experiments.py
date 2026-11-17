#!/usr/bin/env python3
"""
Forecasting monthly UPI transaction volume and value
====================================================
Reproduces every number, table and figure used in the paper (paper/).

    python run_experiments.py

What it does
------------
1. Loads and checks data/upi_monthly_cleaned.csv (NPCI UPI product statistics).
2. EDA figure (volume, value, banks).
3. Over-fitting experiment: polynomial *trend* models of degree 1-8 on the time index,
   raw target vs log target, trained on 2018-2025 and tested on the last 12 months.
4. Main comparison: Linear, Polynomial(+degree CV), Ridge, Lasso on lag features,
   each fitted on the raw target AND on the log target, against two baselines
   (persistence, 12-month-drift). Chronological split; hyper-parameters chosen by
   forward-chaining (TimeSeriesSplit) CV on the training years only.
5. Evaluation: MAPE, RMSE, MAE, R2, sMAPE; one-step-ahead and 12-month recursive.
6. Recursive forecast of the next 12 months (Nov 2026 - Oct 2027).
7. Writes results/results.json, results/tables/*.csv and results/figures/*.png.
"""

import argparse
import json
import platform
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import sklearn

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ------------------------------------------------------------------ configuration
SEED = 42                    # no model here is stochastic, but we fix it anyway
START_DATE = "2018-01-01"    # modelling starts after the 2016-17 launch phase
TEST_MONTHS = 12             # last 12 months = test set
FORECAST_MONTHS = 12
BANK_TREND_WINDOW = 24       # months used to estimate the slope of the bank count
CV_SPLITS = 5                # five expanding-window folds, each validated on 12 months
MAX_TREND_DEGREE = 8

TARGETS = {
    "volume_mn": {"name": "Volume", "short": "volume", "unit": "million transactions", "plot_div": 1000.0,
                  "plot_unit": "billion"},
    "value_cr": {"name": "Value", "short": "value", "unit": "Rs crore", "plot_div": 1e5,
                 "plot_unit": "Rs lakh crore"},
}
SPACES = ("raw", "log")

ALL_FEATURES = [
    "lag_1", "lag_2", "lag_3", "lag_6", "lag_12",
    "roll_mean_3", "roll_mean_6", "roll_mean_12",
    "growth_lag_1", "banks_lag_1",
    "year", "month", "quarter", "days_in_month", "month_sin", "month_cos",
]
POLY_FEATURES = ["lag_1", "lag_12"]
MODEL_NAMES = ["Linear", "Polynomial", "Ridge", "Lasso"]
COLORS = {"raw": "#d95f02", "log": "#1b9e77", "actual": "black", "drift": "#7570b3"}


# ------------------------------------------------------------------ helpers
def to_space(x, space):
    return np.log(x) if space == "log" else x


def from_space(x, space):
    return np.exp(np.clip(x, None, 50)) if space == "log" else x


def load_data(path):
    df = pd.read_csv(path, parse_dates=["date"])[["date", "banks", "volume_mn", "value_cr"]]
    df = df.sort_values("date").reset_index(drop=True)
    checks = {
        "rows": int(len(df)),
        "first_month": df["date"].min().strftime("%Y-%m"),
        "last_month": df["date"].max().strftime("%Y-%m"),
        "missing_cells": int(df.isna().sum().sum()),
        "duplicate_months": int(df["date"].duplicated().sum()),
        "all_month_starts": bool((df["date"].dt.day == 1).all()),
        "continuous_months": bool((df["date"].diff().dropna().dt.days.between(28, 31)).all()),
        "zero_volume_months": int((df["volume_mn"] == 0).sum()),
        "months_banks_decreased": int((df["banks"].diff().dropna() < 0).sum()),
    }
    assert checks["missing_cells"] == 0 and checks["duplicate_months"] == 0 and checks["continuous_months"]
    return df, checks


def metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    e = y_true - y_pred
    return {
        "MAPE": float(np.mean(np.abs(e) / np.abs(y_true)) * 100),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
        "sMAPE": float(np.mean(2 * np.abs(e) / (np.abs(y_true) + np.abs(y_pred))) * 100),
    }


def make_features(df, target, space):
    """Lag / rolling / calendar features for day t built only from months <= t-1."""
    d = df[["date", "banks", target]].copy()
    ys = to_space(d[target].where(d[target] > 0), space)      # zeros (pre-launch) -> NaN
    bs = to_space(d["banks"].astype(float), space)
    d["y_s"] = ys
    for k in (1, 2, 3, 6, 12):
        d[f"lag_{k}"] = ys.shift(k)
    for w in (3, 6, 12):
        d[f"roll_mean_{w}"] = ys.shift(1).rolling(w).mean()
    d["growth_lag_1"] = (ys.shift(1) - ys.shift(2)) if space == "log" else (ys.shift(1) / ys.shift(2) - 1)
    d["banks_lag_1"] = bs.shift(1)
    d["year"] = d["date"].dt.year
    d["month"] = d["date"].dt.month
    d["quarter"] = d["date"].dt.quarter
    d["days_in_month"] = d["date"].dt.days_in_month
    d["month_sin"] = np.sin(2 * np.pi * d["month"] / 12)
    d["month_cos"] = np.cos(2 * np.pi * d["month"] / 12)
    return d.replace([np.inf, -np.inf], np.nan)


def split(feat):
    data = feat[feat["date"] >= START_DATE].dropna(subset=ALL_FEATURES + ["y_s"]).reset_index(drop=True)
    return data, data.iloc[:-TEST_MONTHS], data.iloc[-TEST_MONTHS:]


def mape_scorer(space):
    def f(y_true, y_pred):
        a, p = from_space(np.asarray(y_true), space), from_space(np.asarray(y_pred), space)
        return float(np.mean(np.abs(a - p) / np.abs(a)) * 100)
    return make_scorer(f, greater_is_better=False)


def model_specs():
    def wrap(steps):
        return TransformedTargetRegressor(regressor=Pipeline(steps), transformer=StandardScaler())
    return {
        "Linear": dict(est=wrap([("scaler", StandardScaler()), ("model", LinearRegression())]),
                       grid={}, features=ALL_FEATURES),
        "Polynomial": dict(est=wrap([("scaler", StandardScaler()),
                                     ("poly", PolynomialFeatures(include_bias=False)),
                                     ("scaler2", StandardScaler()), ("model", LinearRegression())]),
                           grid={"regressor__poly__degree": [2, 3]}, features=POLY_FEATURES),
        "Ridge": dict(est=wrap([("scaler", StandardScaler()), ("model", Ridge())]),
                      grid={"regressor__model__alpha": np.logspace(-3, 3, 25)}, features=ALL_FEATURES),
        "Lasso": dict(est=wrap([("scaler", StandardScaler()), ("model", Lasso(max_iter=200000))]),
                      grid={"regressor__model__alpha": np.logspace(-4, 0, 25)}, features=ALL_FEATURES),
    }


def forecast_banks(hist, dates):
    """Future bank count: last observed value + slope of a line fitted to the last 24 months."""
    h = hist.tail(BANK_TREND_WINDOW)
    slope = LinearRegression().fit(np.arange(len(h)).reshape(-1, 1), h["banks"].values).coef_[0]
    return h["banks"].iloc[-1] + slope * np.arange(1, len(dates) + 1)


def recursive_forecast(model, features, hist, target, space, future_dates, future_banks):
    """Predict t+1, feed it back as history, rebuild features, predict t+2, ..."""
    d = hist[["date", "banks", target]].copy()
    out = []
    for dt, b in zip(future_dates, future_banks):
        d = pd.concat([d, pd.DataFrame({"date": [dt], "banks": [b], target: [np.nan]})], ignore_index=True)
        row = make_features(d, target, space).iloc[[-1]][features]
        p = float(from_space(model.predict(row)[0], space))
        d.loc[d.index[-1], target] = p
        out.append(p)
    return np.array(out)


# ------------------------------------------------------------------ figures
def set_style():
    plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300, "font.size": 8, "axes.titlesize": 8.5,
                         "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False,
                         "axes.spines.right": False, "font.family": "DejaVu Sans"})


def fig_eda(df, path):
    d = df[df["date"] >= "2016-07-01"]
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.3))
    ax[0].semilogy(d["date"], d["volume_mn"], color="#1f77b4", lw=1.6)
    ax[0].set(title="(a) Volume (log scale)", ylabel="million transactions")
    ax[1].semilogy(d["date"], d["value_cr"], color="#2ca02c", lw=1.6)
    ax[1].set(title="(b) Value (log scale)", ylabel="Rs crore")
    ax[2].plot(df["date"], df["banks"], color="#ff7f0e", lw=1.6)
    ax[2].set(title="(c) Banks live on UPI", ylabel="number of banks")
    for a in ax:
        a.xaxis.set_major_locator(mdates.YearLocator(2))
        a.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        a.axvline(pd.Timestamp(START_DATE), color="grey", ls=":", lw=1)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def trend_overfit_experiment(df, target):
    """Polynomial in the time index only (no lags): degree 1..8, raw vs log target."""
    d = df[df["date"] >= START_DATE].reset_index(drop=True)
    d["t"] = np.arange(len(d))
    train, test = d.iloc[:-TEST_MONTHS], d.iloc[-TEST_MONTHS:]
    u_scale = float(train["t"].max())
    res = {"raw": {}, "log": {}}
    curves = {}
    for space in SPACES:
        ytr = to_space(train[target].values, space)
        for deg in range(1, MAX_TREND_DEGREE + 1):
            pipe = Pipeline([("poly", PolynomialFeatures(deg, include_bias=False)), ("lr", LinearRegression())])
            Xtr, Xte = (train[["t"]] / u_scale).values, (test[["t"]] / u_scale).values
            pipe.fit(Xtr, ytr)
            ptr, pte = from_space(pipe.predict(Xtr), space), from_space(pipe.predict(Xte), space)
            res[space][deg] = {"train_MAPE": metrics(train[target], ptr)["MAPE"],
                               "test_MAPE": metrics(test[target], pte)["MAPE"]}
            curves[(space, deg)] = pte
    return res, curves, train, test


def fig_overfit(res, curves, train, test, target, path):
    cfg = TARGETS[target]
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7))
    degs = list(range(1, MAX_TREND_DEGREE + 1))
    for space, ls in [("raw", "-"), ("log", "-")]:
        ax[0].plot(degs, [res[space][k]["train_MAPE"] for k in degs], ls="--", marker="o", ms=3,
                   color=COLORS[space], label=f"{space} target: train")
        ax[0].plot(degs, [res[space][k]["test_MAPE"] for k in degs], ls=ls, marker="s", ms=3,
                   color=COLORS[space], label=f"{space} target: test")
    ax[0].set_yscale("log")
    ax[0].set(xlabel="polynomial degree of the time index", ylabel="MAPE (%)",
              title="(a) Trend-only polynomial: error vs degree")
    ax[0].legend(ncol=1)
    hist = train.tail(24)
    ax[1].plot(hist["date"], hist[target] / cfg["plot_div"], "k-", lw=1.4, label="actual (train)")
    ax[1].plot(test["date"], test[target] / cfg["plot_div"], "k-o", ms=2.5, lw=1.4, label="actual (test)")
    for (space, deg), c, ls in [(("raw", 1), COLORS["raw"], "-"), (("raw", 6), COLORS["raw"], ":"),
                                (("log", 1), COLORS["log"], "-"), (("log", 6), COLORS["log"], ":")]:
        ax[1].plot(test["date"], curves[(space, deg)] / cfg["plot_div"], color=c, ls=ls, lw=1.4,
                   label=f"{space}, degree {deg}")
    ax[1].set(title="(b) Extrapolation over the 12 test months", ylabel=f"{cfg['name']} ({cfg['plot_unit']})")
    ax[1].xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
    ax[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    ax[1].tick_params(axis="x", rotation=30)
    ax[1].legend(fontsize=6.5)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


MODEL_COLORS = {"Linear": "#1f77b4", "Polynomial": "#ff7f0e", "Ridge": "#2ca02c", "Lasso": "#d62728"}


def fig_test(all_res, path):
    """Rows: raw-target models / log-target models. Columns: volume / value. 12-month recursive."""
    fig, ax = plt.subplots(2, 2, figsize=(7.2, 5.0), sharex="col")
    for j, target in enumerate(TARGETS):
        cfg, r = TARGETS[target], all_res[target]
        test = r["_test"]
        for i, space in enumerate(SPACES):
            a = ax[i, j]
            a.plot(test["date"], test[target] / cfg["plot_div"], "k-o", ms=3, lw=1.8, label="actual")
            for name in MODEL_NAMES:
                a.plot(test["date"], r["_rec"][(space, name)] / cfg["plot_div"], color=MODEL_COLORS[name],
                       lw=1.3, label=name)
            a.plot(test["date"], r["_rec"]["Drift"] / cfg["plot_div"], color="#7570b3", ls="--", lw=1.1, label="drift")
            a.plot(test["date"], r["_rec"]["Persistence"] / cfg["plot_div"], color="grey", ls=":", lw=1.1,
                   label="persistence")
            a.set(title=f"({'abcd'[i * 2 + j]}) {cfg['name']}, {space}-target models",
                  ylabel=f"{cfg['name']} ({cfg['plot_unit']})")
            a.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
            a.tick_params(axis="x", rotation=30)
    ax[0, 0].legend(fontsize=6, loc="upper left", ncol=2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_forecast(all_res, df, path):
    fig, ax = plt.subplots(2, 2, figsize=(7.2, 5.0), sharex="col")
    for j, target in enumerate(TARGETS):
        cfg, r = TARGETS[target], all_res[target]
        h = df[df["date"] >= "2023-01-01"]
        fc = r["_forecast"]
        for i, space in enumerate(SPACES):
            a = ax[i, j]
            a.plot(h["date"], h[target] / cfg["plot_div"], "k-", lw=1.6, label="actual")
            for name in MODEL_NAMES:
                a.plot(fc["date"], fc[f"{name}_{space}"] / cfg["plot_div"], color=MODEL_COLORS[name], lw=1.4,
                       label=name)
            a.axvline(df["date"].max(), color="grey", ls=":", lw=1)
            a.set(title=f"({'abcd'[i * 2 + j]}) {cfg['name']}, {space}-target models",
                  ylabel=f"{cfg['name']} ({cfg['plot_unit']})")
            a.xaxis.set_major_locator(mdates.YearLocator())
            a.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax[0, 0].legend(fontsize=6.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ------------------------------------------------------------------ main experiment
def run_target(df, target, tab_dir):
    cfg = TARGETS[target]
    out = {"models": {}, "baselines": {}}
    internal = {"_rec": {}, "_one": {}}
    orig = df.set_index("date")[target]

    train_end = None
    for space in SPACES:
        feat = make_features(df, target, space)
        data, train, test = split(feat)
        train_end = train["date"].max()
        out["n_usable"] = int(len(data))
        out["n_train"] = int(len(train))
        out["n_test"] = int(len(test))
        out["train_range"] = [train["date"].min().strftime("%Y-%m"), train["date"].max().strftime("%Y-%m")]
        out["test_range"] = [test["date"].min().strftime("%Y-%m"), test["date"].max().strftime("%Y-%m")]
        y_test = orig.loc[test["date"]].values
        hist = df[df["date"] < test["date"].min()]
        fut_banks = forecast_banks(hist, test["date"])
        internal["_test"] = df[df["date"].isin(test["date"])].reset_index(drop=True)

        cv = TimeSeriesSplit(n_splits=CV_SPLITS, test_size=12)
        for name, spec in model_specs().items():
            gs = GridSearchCV(spec["est"], spec["grid"], cv=cv, scoring=mape_scorer(space), n_jobs=1,
                              error_score=-1e6)
            gs.fit(train[spec["features"]], train["y_s"])
            best = gs.best_estimator_
            p_one = from_space(best.predict(test[spec["features"]]), space)
            p_rec = recursive_forecast(best, spec["features"], hist, target, space, test["date"], fut_banks)
            entry = {
                "params": {k.split("__")[-1]: float(v) for k, v in gs.best_params_.items()},
                "cv_MAPE": float(-gs.best_score_),
                "onestep": metrics(y_test, p_one),
                "recursive": metrics(y_test, p_rec),
            }
            if name == "Lasso":
                coefs = best.regressor_.named_steps["model"].coef_
                entry["lasso_kept"] = [f for f, c in zip(ALL_FEATURES, coefs) if c != 0]
                entry["lasso_n_features"] = len(ALL_FEATURES)
            if name in ("Linear", "Ridge", "Lasso"):
                entry["coef"] = {f: float(c) for f, c in
                                 zip(ALL_FEATURES, best.regressor_.named_steps["model"].coef_)}
            out["models"][f"{name}|{space}"] = entry
            internal["_one"][(space, name)] = p_one
            internal["_rec"][(space, name)] = p_rec

            # final refit on ALL usable data for the genuine out-of-sample forecast
            fut_dates = pd.date_range(df["date"].max() + pd.offsets.MonthBegin(1), periods=FORECAST_MONTHS,
                                      freq="MS")
            final = clone(spec["est"]).set_params(**gs.best_params_).fit(data[spec["features"]], data["y_s"])
            internal.setdefault("_fc", {})[(space, name)] = recursive_forecast(
                final, spec["features"], df, target, space, fut_dates, forecast_banks(df, fut_dates))
            internal["_fut_dates"] = fut_dates

    # ---- baselines (computed on original scale)
    test_dates = internal["_test"]["date"]
    prev = orig.shift(1).loc[test_dates].values
    drift_one = prev * (prev / orig.shift(13).loc[test_dates].values) ** (1 / 12)
    last = orig.loc[train_end]
    last12 = orig.shift(12).loc[train_end]
    h = np.arange(1, TEST_MONTHS + 1)
    drift_rec = last * (last / last12) ** (h / 12)
    pers_rec = np.repeat(last, TEST_MONTHS)
    y_test = orig.loc[test_dates].values
    for name, one, rec in [("Persistence", prev, pers_rec), ("Drift", drift_one, drift_rec)]:
        out["baselines"][name] = {"onestep": metrics(y_test, one), "recursive": metrics(y_test, rec)}
        internal["_one"][name] = one
        internal["_rec"][name] = rec

    # ---- selection by CV-MAPE (training data only; the test set is never used to choose)
    cv_scores = {k: v["cv_MAPE"] for k, v in out["models"].items()}
    out["selected_by_cv"] = min(cv_scores, key=cv_scores.get)
    out["best_per_space"] = {s: min((k for k in cv_scores if k.endswith(f"|{s}")), key=cv_scores.get).split("|")[0]
                             for s in SPACES}

    # ---- tables
    rows = []
    for k, v in out["models"].items():
        m, s = k.split("|")
        for stage in ("onestep", "recursive"):
            rows.append({"model": m, "target_space": s, "stage": stage, "cv_MAPE": v["cv_MAPE"], **v[stage]})
    for k, v in out["baselines"].items():
        for stage in ("onestep", "recursive"):
            rows.append({"model": k, "target_space": "-", "stage": stage, "cv_MAPE": np.nan, **v[stage]})
    pd.DataFrame(rows).to_csv(tab_dir / f"metrics_{cfg['short']}.csv", index=False)

    pred = pd.DataFrame({"date": test_dates.dt.strftime("%Y-%m").values, "actual": y_test})
    for key in internal["_one"]:
        tag = key if isinstance(key, str) else f"{key[1]}_{key[0]}"
        pred[f"{tag}_onestep"] = internal["_one"][key]
        pred[f"{tag}_recursive"] = internal["_rec"][key]
    pred.to_csv(tab_dir / f"test_predictions_{cfg['short']}.csv", index=False)

    fc = pd.DataFrame({"date": internal["_fut_dates"]})
    for (space, name), p in internal["_fc"].items():
        fc[f"{name}_{space}"] = p
    fc["banks_assumed"] = forecast_banks(df, internal["_fut_dates"])
    fc.assign(date=fc["date"].dt.strftime("%Y-%m")).to_csv(tab_dir / f"forecast_{cfg['short']}.csv", index=False)
    sel_m, sel_s = out["selected_by_cv"].split("|")
    out["forecast_selected"] = {"model": out["selected_by_cv"],
                                "first": float(fc[f"{sel_m}_{sel_s}"].iloc[0]),
                                "last": float(fc[f"{sel_m}_{sel_s}"].iloc[-1]),
                                "values": [float(x) for x in fc[f"{sel_m}_{sel_s}"]],
                                "dates": [d.strftime("%Y-%m") for d in fc["date"]]}
    out["forecast_all"] = {f"{n}|{sp}": {"first": float(fc[f"{n}_{sp}"].iloc[0]), "last": float(fc[f"{n}_{sp}"].iloc[-1])}
                           for sp in SPACES for n in MODEL_NAMES}
    out["forecast_range_last"] = {"min": float(fc.filter(regex="_(raw|log)$").iloc[-1].min()),
                                  "max": float(fc.filter(regex="_(raw|log)$").iloc[-1].max())}

    internal["_forecast"] = fc
    return out, internal


def eda_summary(df):
    d = df[df["date"] >= START_DATE].reset_index(drop=True)
    s = {}
    for t in TARGETS:
        v = d[t]
        mom = v.pct_change() * 100
        s[t] = {"first": float(v.iloc[0]), "last": float(v.iloc[-1]), "min": float(v.min()),
                "max": float(v.max()), "mean": float(v.mean()), "std": float(v.std()),
                "growth_factor": float(v.iloc[-1] / v.iloc[0]),
                "yoy_last_pct": float((v.iloc[-1] / v.iloc[-13] - 1) * 100),
                "yoy_first_pct": float((v.iloc[12] / v.iloc[0] - 1) * 100),
                "mean_mom_last12_pct": float(mom.tail(12).mean()),
                "months_with_decline": int((mom < 0).sum()),
                "months_total": int(len(mom) - 1),
                "largest_mom_drop_pct": float(mom.min())}
    s["banks"] = {"first": int(d["banks"].iloc[0]), "last": int(d["banks"].iloc[-1]),
                  "added_last12": int(d["banks"].iloc[-1] - d["banks"].iloc[-13])}
    h = df.tail(BANK_TREND_WINDOW)
    s["bank_slope_per_month"] = float(LinearRegression().fit(np.arange(len(h)).reshape(-1, 1),
                                                             h["banks"].values).coef_[0])
    v = d.set_index("date")["volume_mn"]
    ratio = (v / v.rolling(12, center=True).mean()).dropna()
    s["seasonal_ratio_volume"] = {int(m): float(x) for m, x in ratio.groupby(ratio.index.month).mean().items()}
    s["corr_logvol_logbanks"] = float(np.corrcoef(np.log(d["volume_mn"]), np.log(d["banks"]))[0, 1])
    s["avg_ticket_first"] = float(d["value_cr"].iloc[0] * 10 / d["volume_mn"].iloc[0])      # Rs per txn
    s["avg_ticket_last"] = float(d["value_cr"].iloc[-1] * 10 / d["volume_mn"].iloc[-1])
    return s


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(here / "data" / "upi_monthly_cleaned.csv"))
    ap.add_argument("--out", default=str(here / "results"))
    a = ap.parse_args()
    out_dir = Path(a.out)
    fig_dir, tab_dir = out_dir / "figures", out_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(SEED)
    set_style()

    df, checks = load_data(a.data)
    print("data checks:", checks)
    results = {"config": {"seed": SEED, "start_date": START_DATE, "test_months": TEST_MONTHS,
                          "forecast_months": FORECAST_MONTHS, "cv_splits": CV_SPLITS,
                          "n_features": len(ALL_FEATURES), "poly_features": POLY_FEATURES,
                          "python": platform.python_version(), "sklearn": sklearn.__version__,
                          "pandas": pd.__version__, "numpy": np.__version__},
               "data_checks": checks, "eda": eda_summary(df)}

    fig_eda(df, fig_dir / "fig1_eda.png")

    over, curves, tr, te = trend_overfit_experiment(df, "volume_mn")
    results["overfit_volume"] = {s: {str(k): v for k, v in d.items()} for s, d in over.items()}
    over_v, _, _, _ = trend_overfit_experiment(df, "value_cr")
    results["overfit_value"] = {s: {str(k): v for k, v in d.items()} for s, d in over_v.items()}
    fig_overfit(over, curves, tr, te, "volume_mn", fig_dir / "fig2_overfit.png")

    all_res = {}
    for t in TARGETS:
        print(f"\n=== {TARGETS[t]['name']} ===")
        res, internal = run_target(df, t, tab_dir)
        results[t] = res
        all_res[t] = {**res, **internal}
        for k, v in res["models"].items():
            print(f"{k:<18} cv_MAPE={v['cv_MAPE']:.2f}  one-step MAPE={v['onestep']['MAPE']:.2f}  "
                  f"recursive MAPE={v['recursive']['MAPE']:.2f}  params={v['params']}")
        for k, v in res["baselines"].items():
            print(f"{k:<18} one-step MAPE={v['onestep']['MAPE']:.2f}  recursive MAPE={v['recursive']['MAPE']:.2f}")
        print("selected by CV:", res["selected_by_cv"], "| best per space:", res["best_per_space"])

    fig_test(all_res, fig_dir / "fig3_test.png")
    fig_forecast(all_res, df, fig_dir / "fig4_forecast.png")

    comb = pd.DataFrame({"date": all_res["volume_mn"]["_forecast"]["date"].dt.strftime("%Y-%m")})
    for t in TARGETS:
        f = all_res[t]["_forecast"].drop(columns=["date", "banks_assumed"])
        comb = pd.concat([comb, f.add_prefix(f"{TARGETS[t]['short']}_")], axis=1)
    comb["banks_assumed"] = all_res["volume_mn"]["_forecast"]["banks_assumed"].values
    comb.to_csv(tab_dir / "forecast_combined.csv", index=False)

    (out_dir / "results.json").write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    print(f"\nDone. Results in {out_dir}")


if __name__ == "__main__":
    main()
