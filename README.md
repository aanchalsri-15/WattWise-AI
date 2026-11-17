# WattWise AI — Household Electricity Consumption Forecasting

**Capstone project | AI product lifecycle | Electrical Engineering**

WattWise AI forecasts the next day's average household active power (kW) from historical electricity-use measurements. It includes a reproducible data pipeline, chronological model evaluation, a Streamlit dashboard, and tests.

> **Integrity note:** No model scores are pre-filled. Run `python train.py` to generate `reports/metrics.json` and actual test predictions. Never submit invented metrics.

## Problem statement
Household energy users may find it difficult to anticipate near-term electricity demand. This project frames next-day average active power forecasting as supervised time-series regression using the UCI Individual Household Electric Power Consumption dataset.

## What the project does
1. Downloads the official UCI dataset if it is not already in `data/`.
2. Parses the semicolon-separated file and treats `?` as missing.
3. Aggregates minute-level active-power readings into daily means.
4. Creates lag, trailing-average, and calendar features using only prior observations.
5. Compares a persistence baseline with a Random Forest Regressor.
6. Uses a chronological 80/20 split and reports MAE, RMSE, and R².
7. Saves a model, metrics, predictions, and feature importances.
8. Provides a Streamlit dashboard.

## Dataset
- **Name:** Individual Household Electric Power Consumption
- **Source:** UCI Machine Learning Repository, dataset 235
- **DOI:** https://doi.org/10.24432/C58K54
- **Coverage:** December 2006–November 2010; one-minute measurements; 2,075,259 rows
- **License:** CC BY 4.0

Dataset page: https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption

## Repository structure
```
WattWise-AI/
├── app.py
├── train.py
├── requirements.txt
├── README.md
├── LICENSE
├── data/                 # raw dataset is not committed
├── models/               # generated model is not committed
├── reports/
│   ├── EXECUTIVE_SUMMARY.md
│   ├── PROJECT_METHODOLOGY.md
│   └── FIVE_SLIDE_PITCH.md
├── src/pipeline.py
└── tests/test_pipeline.py
```

## Run locally (Python 3.10+ recommended)
```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python train.py
python -m pytest
streamlit run app.py
```
The first training run downloads the UCI archive (internet required). Alternatively, place `household_power_consumption.txt` in `data/`.

## Evaluation design
- **Target:** daily mean `Global_active_power` in kW for day t.
- **Features:** lags 1, 2, 7, 14, 28; trailing 7- and 28-day means; weekday, month, weekend.
- **Leakage prevention:** lag and rolling features are shifted to use only information before the forecast day.
- **Split:** earliest 80% train, latest 20% test; no random shuffle.
- **Models:** persistence baseline and Random Forest.
- **Metrics:** MAE and RMSE (kW), R².

## Limitations
The dataset represents one household in Sceaux, France, from 2006–2010. It may not generalize to present-day homes, other regions, tariffs, weather, or appliance mixes. The project predicts average household active power, not appliance-level use, electricity bills, or grid peaks. It is a student prototype, not a utility-grade system.

## Submission checklist
- [ ] Run `python train.py` and record actual MAE, RMSE, R².
- [ ] Run `python -m pytest`.
- [ ] Test `streamlit run app.py`.
- [ ] Add institution name to the executive summary.
- [ ] Insert measured scores into the executive summary and presentation.
- [ ] Deploy the dashboard if required and include its URL.

## Dataset citation
Hebrail, G., & Berard, A. (2006). *Individual Household Electric Power Consumption* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C58K54. CC BY 4.0.
