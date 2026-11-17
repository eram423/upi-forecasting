# Forecasting monthly UPI transaction volume and value

Machine Learning (CSA) application project, B.Sc. (H) Computer Science, Ramanujan College, University of Delhi.
Problem 37: forecast monthly UPI transaction volume and value with Linear, Polynomial, Ridge and Lasso regression,
on the raw and the log target, using a chronological train/test split, and discuss over-fitting on exponential growth.

**Author:** Eram Alameen (Roll No. 24075570020)

## What this repository contains

| Path | Content |
|---|---|
| `run_experiments.py` | The whole experiment: data checks, EDA figure, over-fitting experiment, model training and tuning, evaluation, forecasting. Writes everything in `results/`. |
| `prepare_data.py` | Converts the raw NPCI workbooks in `data/raw/` (numbers stored as text, e.g. `1,33,460.72`) into the cleaned CSV, with checks for duplicate months and gaps. |
| `data/raw/` | The 11 original NPCI yearly workbooks (FY2016-17 to FY2026-27), unmodified. |
| `data/upi_monthly_cleaned.csv` | Monthly NPCI UPI statistics: banks live, volume (million), value (Rs crore), Apr 2016 - Oct 2026 (127 rows). Source: https://www.npci.org.in/product/upi/product-statistics |
| `results/results.json` | Every number quoted in the paper (metrics, tuned parameters, forecasts, data checks). |
| `results/tables/` | `metrics_*.csv`, `test_predictions_*.csv`, `forecast_*.csv`, `forecast_combined.csv` |
| `results/figures/` | `fig1_eda.png` ... `fig4_forecast.png` (the figures used in the paper) |
| `paper/` | The short communication as `.docx` and `.pdf`, and `build_paper.js` which builds it from `results.json`. |

## How to run

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows;  on macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python prepare_data.py          # optional: rebuilds data/upi_monthly_cleaned.csv from data/raw/
python run_experiments.py       # about a minute; regenerates everything in results/
```

Tested with Python 3.12.3, scikit-learn 1.9.1, pandas 3.0.6,
NumPy 2.5.4. Random seed 42 (no model here is stochastic, so results are deterministic).

To rebuild the paper (optional, needs Node.js): `cd paper && npm install docx && node build_paper.js`.
The build script reads `results/results.json` and stops with an error if a claim in the text is no longer supported by the numbers.

## Method in brief

* **Window:** modelling starts Jan 2018 (after the 2016-17 launch phase); earlier months only supply lags. 106 usable months.
* **Split:** train Jan 2018 - Oct 2025 (94 months), test Nov 2025 - Oct 2026 (12 months). No shuffling.
* **Features (16):** target lags 1, 2, 3, 6, 12; rolling means of the previous 3, 6, 12 months; last month's growth;
  last month's number of banks; year, month, quarter, days in month, sin/cos of month. All use data up to month t-1 only.
* **Targets:** raw, and natural log (predictions are converted back before any metric is computed).
* **Models:** Linear; Polynomial (degree 2 or 3 on `lag_1`, `lag_12`); Ridge; Lasso. Features and target standardised inside a pipeline fitted on training data only.
* **Tuning:** `GridSearchCV` + `TimeSeriesSplit` (5 expanding-window folds, each validated on 12 months), scored by MAPE on the original scale. The test set is never used for selection.
* **Baselines:** persistence (last month) and drift (last 12 months' average growth).
* **Evaluation:** MAPE, RMSE, MAE, R2, sMAPE; one-step-ahead and 12-month recursive.
* **Over-fitting experiment:** trend-only polynomials of degree 1-8 on the time index, raw vs log target.

## Headline results (test period Nov 2025 - Oct 2026)

**Volume**

| Model / target | CV MAPE % | One-step MAPE % | 12-month recursive MAPE % |
|---|---|---|---|
| Linear / raw | 12.99 | 1.13 | 1.26 |
| Linear / log | 8.09 | 2.03 | 4.34 |
| Polynomial / raw | 7.07 | 0.86 | 0.87 |
| Polynomial / log | 30.20 | 2.06 | 2.69 |
| Ridge / raw | 3.87 | 1.62 | 1.48 |
| Ridge / log | 6.86 | 5.28 | 10.47 |
| Lasso / raw | 3.36 | 3.54 | 4.16 |
| Lasso / log | 4.98 | 2.22 | 3.18 |
| Persistence baseline | - | 3.76 | 8.99 |
| Drift baseline | - | 3.78 | 3.05 |

**Value**

| Model / target | CV MAPE % | One-step MAPE % | 12-month recursive MAPE % |
|---|---|---|---|
| Linear / raw | 10.12 | 1.35 | 1.74 |
| Linear / log | 13.37 | 2.37 | 6.31 |
| Polynomial / raw | 11.78 | 4.31 | 3.42 |
| Polynomial / log | 5.15 | 2.32 | 1.96 |
| Ridge / raw | 4.05 | 1.72 | 1.53 |
| Ridge / log | 7.08 | 4.71 | 8.23 |
| Lasso / raw | 3.63 | 3.80 | 7.59 |
| Lasso / log | 4.62 | 4.94 | 16.17 |
| Persistence baseline | - | 3.77 | 6.45 |
| Drift baseline | - | 3.81 | 2.67 |

Main findings (details and caveats in the paper):

1. Polynomial trends in time alone over-fit: training error keeps falling with degree while test error is erratic and, on the log target, reaches thousands of percent at degree 8.
2. The log transform did not help here, because growth has slowed since the launch years.
3. Raw-target Linear, Ridge and (for volume) Polynomial models beat both baselines in the 12-month recursive test.
4. The Lasso model chosen by cross-validation was **not** the best on the test set.

## Limitations

Only 12 test months from a single period; polynomial feature set fixed after a preliminary run showed a 3-feature version diverging
(so its test scores are slightly optimistic); future bank counts are a straight-line assumption; no prediction intervals;
no external events (policy, festivals, outages) are modelled. See Section 5 of the paper.

## Use of AI tools

The code, figures and first draft of the paper were produced with the help of an AI assistant (Claude, Anthropic) from the author's instructions.
See the declaration in the paper.
